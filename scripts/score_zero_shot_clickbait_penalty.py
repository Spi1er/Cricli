#!/usr/bin/env python3
"""Score zero-shot generated headlines with the clickbait penalty critic."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_zero_shot_100.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_zero_shot_scored_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_zero_shot_profile.md"


class TitleDataset(Dataset):
    def __init__(self, titles: list[str], tokenizer, max_length: int) -> None:
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(self.titles[index], truncation=True, max_length=self.max_length)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def clean_generated_title(value: object) -> str:
    title = clean_text(value)
    title = re.sub(r"^(headline|title)\s*:\s*", "", title, flags=re.IGNORECASE).strip()
    title = title.strip("\"'` ")
    # A generated headline should not end like a sentence unless punctuation is
    # semantically useful. Keep question/exclamation marks for diagnostics.
    title = re.sub(r"\.$", "", title).strip()
    return title


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_device(device: str) -> torch.device:
    if device == "auto":
        return infer_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()):
            raise RuntimeError("Requested --device mps, but this PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but torch.backends.mps.is_available() is False.")
        torch.ones(1, device="mps")
    if device not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Unsupported device: {device}")
    return torch.device(device)


def score_titles(titles: list[str], model_path: Path, batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = TitleDataset(titles, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def build_report(df: pd.DataFrame, output: Path, threshold: float) -> str:
    delta = df["zero_shot_clickbait_penalty"] - df["original_clickbait_penalty"]
    improved = (delta < 0).mean()
    worsened = (delta > 0).mean()

    category_profile = (
        df.groupby("category")
        .agg(
            rows=("seed_id", "size"),
            original_mean=("original_clickbait_penalty", "mean"),
            zero_shot_mean=("zero_shot_clickbait_penalty", "mean"),
            mean_delta=("clickbait_penalty_delta", "mean"),
            original_clickbait_rate=("original_predicted_clickbait", "mean"),
            zero_shot_clickbait_rate=("zero_shot_predicted_clickbait", "mean"),
        )
        .sort_values("mean_delta")
    )

    biggest_reductions = df.sort_values("clickbait_penalty_delta").head(12)
    biggest_increases = df.sort_values("clickbait_penalty_delta", ascending=False).head(12)

    lines = [
        "# Zero-Shot Headline Clickbait Penalty Profile",
        "",
        f"- Output: `{output}`",
        f"- Rows: {len(df):,}",
        f"- Threshold: {threshold:.2f}",
        f"- Original mean penalty: {df['original_clickbait_penalty'].mean():.4f}",
        f"- Zero-shot mean penalty: {df['zero_shot_clickbait_penalty'].mean():.4f}",
        f"- Mean delta (zero-shot - original): {delta.mean():.4f}",
        f"- Median delta: {delta.median():.4f}",
        f"- Improved rows: {improved:.2%}",
        f"- Worsened rows: {worsened:.2%}",
        f"- Original predicted clickbait rate: {df['original_predicted_clickbait'].mean():.2%}",
        f"- Zero-shot predicted clickbait rate: {df['zero_shot_predicted_clickbait'].mean():.2%}",
        "",
        "## Category Profile",
        "",
        "| Category | Rows | Original mean | Zero-shot mean | Mean delta | Original rate | Zero-shot rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for category, row in category_profile.iterrows():
        lines.append(
            f"| {category} | {int(row['rows'])} | {row['original_mean']:.4f} | "
            f"{row['zero_shot_mean']:.4f} | {row['mean_delta']:.4f} | "
            f"{row['original_clickbait_rate']:.2%} | {row['zero_shot_clickbait_rate']:.2%} |"
        )

    lines.extend(["", "## Biggest Penalty Reductions", "", "| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |", "| ---: | ---: | ---: | --- | --- | --- |"])
    for row in biggest_reductions.itertuples(index=False):
        original = str(row.original_title).replace("|", "\\|")
        generated = str(row.zero_shot_title_clean).replace("|", "\\|")
        lines.append(
            f"| {row.clickbait_penalty_delta:.4f} | {row.original_clickbait_penalty:.4f} | "
            f"{row.zero_shot_clickbait_penalty:.4f} | {row.category} | {original} | {generated} |"
        )

    lines.extend(["", "## Biggest Penalty Increases", "", "| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |", "| ---: | ---: | ---: | --- | --- | --- |"])
    for row in biggest_increases.itertuples(index=False):
        original = str(row.original_title).replace("|", "\\|")
        generated = str(row.zero_shot_title_clean).replace("|", "\\|")
        lines.append(
            f"| {row.clickbait_penalty_delta:.4f} | {row.original_clickbait_penalty:.4f} | "
            f"{row.zero_shot_clickbait_penalty:.4f} | {row.category} | {original} | {generated} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative deltas mean the API zero-shot headline is less clickbait-like according to the fine-tuned DistilBERT penalty critic. This does not evaluate faithfulness or attractiveness; those need separate critics or human/LLM-judge evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    device = validate_device(args.device)
    df = pd.read_csv(args.input)
    df["zero_shot_title_clean"] = df["zero_shot_title"].map(clean_generated_title)

    missing = df["zero_shot_title_clean"].eq("")
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} rows are missing zero_shot_title. Finish generation first.")

    print(f"Scoring {len(df):,} zero-shot titles on {device}...")
    probs = score_titles(df["zero_shot_title_clean"].tolist(), args.model, args.batch_size, args.max_length, device)

    df["zero_shot_clickbait_penalty"] = probs
    df["zero_shot_predicted_clickbait"] = (df["zero_shot_clickbait_penalty"] >= args.threshold).astype(int)
    df["clickbait_penalty_delta"] = df["zero_shot_clickbait_penalty"] - df["original_clickbait_penalty"]
    df["clickbait_penalty_reduced"] = df["clickbait_penalty_delta"] < 0
    df["clickbait_threshold"] = args.threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    report = build_report(df, args.output, args.threshold)
    args.report.write_text(report, encoding="utf-8")

    metadata = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(args.output),
        "report": str(args.report),
        "rows": len(df),
        "threshold": args.threshold,
        "original_mean_penalty": float(df["original_clickbait_penalty"].mean()),
        "zero_shot_mean_penalty": float(df["zero_shot_clickbait_penalty"].mean()),
        "mean_delta": float(df["clickbait_penalty_delta"].mean()),
        "original_clickbait_rate": float(df["original_predicted_clickbait"].mean()),
        "zero_shot_clickbait_rate": float(df["zero_shot_predicted_clickbait"].mean()),
        "reduced_rate": float(df["clickbait_penalty_reduced"].mean()),
        "device": str(device),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", metadata_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
