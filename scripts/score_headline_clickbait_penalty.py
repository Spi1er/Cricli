#!/usr/bin/env python3
"""Score MIND headline pool with the DistilBERT clickbait penalty critic.

Input:
  data/processed/mind_headline_pool_sample.csv

Output:
  data/processed/mind_headline_pool_with_clickbait_penalty.csv
  data/processed/clickbait_penalty_profile.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mind_headline_pool_sample.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mind_headline_pool_with_clickbait_penalty.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "clickbait_penalty_profile.md"


class TitleOnlyDataset(Dataset):
    def __init__(self, titles: list[str], tokenizer, max_length: int) -> None:
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            self.titles[index],
            truncation=True,
            max_length=self.max_length,
        )


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


def score_titles(
    titles: list[str],
    model_path: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = TitleOnlyDataset(titles, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probabilities.append(probs)

    return np.concatenate(probabilities)


def build_report(df: pd.DataFrame, output_path: Path, model_path: Path, threshold: float) -> str:
    high = df[df["clickbait_penalty"] >= threshold].copy()
    category_profile = (
        df.groupby("category", dropna=False)
        .agg(
            rows=("title", "size"),
            mean_penalty=("clickbait_penalty", "mean"),
            median_penalty=("clickbait_penalty", "median"),
            high_penalty_rate=("predicted_clickbait", "mean"),
        )
        .sort_values(["mean_penalty", "rows"], ascending=[False, False])
        .head(15)
    )

    top_examples = df.sort_values("clickbait_penalty", ascending=False).head(12)
    low_examples = df.sort_values("clickbait_penalty", ascending=True).head(8)

    lines = [
        "# MIND Headline Clickbait Penalty Profile",
        "",
        f"- Input rows: {len(df):,}",
        f"- Model: `{model_path}`",
        f"- Output: `{output_path}`",
        f"- Decision threshold: {threshold:.2f}",
        f"- Mean clickbait penalty: {df['clickbait_penalty'].mean():.4f}",
        f"- Median clickbait penalty: {df['clickbait_penalty'].median():.4f}",
        f"- Predicted clickbait titles: {int(df['predicted_clickbait'].sum()):,} ({df['predicted_clickbait'].mean():.2%})",
        "",
        "## Category Profile",
        "",
        "| Category | Rows | Mean penalty | Median penalty | High-penalty rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for category, row in category_profile.iterrows():
        lines.append(
            f"| {category} | {int(row['rows']):,} | {row['mean_penalty']:.4f} | "
            f"{row['median_penalty']:.4f} | {row['high_penalty_rate']:.2%} |"
        )

    lines.extend(["", "## Highest Penalty Examples", "", "| Penalty | Category | Title |", "| ---: | --- | --- |"])
    for row in top_examples.itertuples(index=False):
        title = str(row.title).replace("|", "\\|")
        lines.append(f"| {row.clickbait_penalty:.4f} | {row.category} | {title} |")

    lines.extend(["", "## Lowest Penalty Examples", "", "| Penalty | Category | Title |", "| ---: | --- | --- |"])
    for row in low_examples.itertuples(index=False):
        title = str(row.title).replace("|", "\\|")
        lines.append(f"| {row.clickbait_penalty:.4f} | {row.category} | {title} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use `clickbait_penalty` as a negative reward component. This score estimates whether a headline has clickbait-style wording; it does not directly measure factuality or audience alignment.",
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
    titles = df["title"].fillna("").astype(str).tolist()

    print(f"Scoring {len(titles):,} titles on {device}...")
    penalties = score_titles(titles, args.model, args.batch_size, args.max_length, device)

    df["clickbait_penalty"] = penalties
    df["predicted_clickbait"] = (df["clickbait_penalty"] >= args.threshold).astype(int)
    df["clickbait_model"] = args.model.name
    df["clickbait_threshold"] = args.threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    report = build_report(df, args.output, args.model, args.threshold)
    args.report.write_text(report, encoding="utf-8")

    metadata = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(args.output),
        "report": str(args.report),
        "rows": len(df),
        "threshold": args.threshold,
        "mean_clickbait_penalty": float(df["clickbait_penalty"].mean()),
        "median_clickbait_penalty": float(df["clickbait_penalty"].median()),
        "predicted_clickbait_rate": float(df["predicted_clickbait"].mean()),
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
