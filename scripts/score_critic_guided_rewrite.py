#!/usr/bin/env python3
"""Score critic-guided rewrites and compare against original/zero-shot titles."""

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_critic_guided_100.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_critic_guided_scored_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_critic_guided_profile.md"


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


def clean_title(value: object) -> str:
    title = clean_text(value)
    title = re.sub(r"^(headline|title|rewritten headline)\s*:\s*", "", title, flags=re.IGNORECASE).strip()
    title = title.strip("\"'` ")
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


def build_report(df: pd.DataFrame, target: pd.DataFrame, output: Path, threshold: float) -> str:
    target_zero_delta = target["rewritten_clickbait_penalty"] - target["zero_shot_clickbait_penalty"]
    target_original_delta = target["rewritten_clickbait_penalty"] - target["original_clickbait_penalty"]

    lines = [
        "# Critic-Guided Rewrite Clickbait Penalty Profile",
        "",
        f"- Output: `{output}`",
        f"- Total rows: {len(df):,}",
        f"- Rewrite target threshold: {threshold:.2f}",
        f"- Rewritten target rows: {len(target):,}",
        "",
        "## Full 100-Row Comparison",
        "",
        f"- Original mean penalty: {df['original_clickbait_penalty'].mean():.4f}",
        f"- Zero-shot mean penalty: {df['zero_shot_clickbait_penalty'].mean():.4f}",
        f"- Final mean penalty: {df['final_clickbait_penalty'].mean():.4f}",
        f"- Original clickbait rate: {df['original_predicted_clickbait'].mean():.2%}",
        f"- Zero-shot clickbait rate: {df['zero_shot_predicted_clickbait'].mean():.2%}",
        f"- Final clickbait rate: {df['final_predicted_clickbait'].mean():.2%}",
        "",
        "## Rewritten Target Rows Only",
        "",
        f"- Target zero-shot mean penalty: {target['zero_shot_clickbait_penalty'].mean():.4f}",
        f"- Target rewritten mean penalty: {target['rewritten_clickbait_penalty'].mean():.4f}",
        f"- Mean delta vs zero-shot: {target_zero_delta.mean():.4f}",
        f"- Median delta vs zero-shot: {target_zero_delta.median():.4f}",
        f"- Rows improved vs zero-shot: {(target_zero_delta < 0).mean():.2%}",
        f"- Rows below threshold after rewrite: {(target['rewritten_clickbait_penalty'] < threshold).mean():.2%}",
        f"- Mean delta vs original: {target_original_delta.mean():.4f}",
        "",
        "## Rewritten Examples",
        "",
        "| Seed | Category | Zero-shot penalty | Rewrite penalty | Delta | Zero-shot title | Rewritten title |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]

    examples = target.sort_values("rewrite_vs_zero_shot_delta")
    for row in examples.itertuples(index=False):
        zero = str(row.zero_shot_title_clean).replace("|", "\\|")
        rewritten = str(row.rewritten_title_clean).replace("|", "\\|")
        lines.append(
            f"| {row.seed_id} | {row.category} | {row.zero_shot_clickbait_penalty:.4f} | "
            f"{row.rewritten_clickbait_penalty:.4f} | {row.rewrite_vs_zero_shot_delta:.4f} | "
            f"{zero} | {rewritten} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report measures whether critic-guided rewriting reduces the clickbait penalty for the subset of zero-shot headlines that remained above the threshold. It only evaluates clickbait style, not factual faithfulness or audience preference.",
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
    df["rewritten_title_clean"] = df["rewritten_title"].map(clean_title)

    target_mask = df["zero_shot_clickbait_penalty"] >= args.threshold
    missing_rewrites = target_mask & df["rewritten_title_clean"].eq("")
    if missing_rewrites.any():
        raise SystemExit(f"{int(missing_rewrites.sum())} target rows are missing rewritten_title.")

    print(f"Scoring {int(target_mask.sum()):,} rewritten titles on {device}...")
    rewritten_probs = np.full(len(df), np.nan, dtype=np.float32)
    rewritten_probs[target_mask.to_numpy()] = score_titles(
        df.loc[target_mask, "rewritten_title_clean"].tolist(),
        args.model,
        args.batch_size,
        args.max_length,
        device,
    )

    df["rewritten_clickbait_penalty"] = rewritten_probs
    df["rewritten_predicted_clickbait"] = np.where(
        df["rewritten_clickbait_penalty"].notna(),
        (df["rewritten_clickbait_penalty"] >= args.threshold).astype(int),
        np.nan,
    )
    df["rewrite_vs_zero_shot_delta"] = df["rewritten_clickbait_penalty"] - df["zero_shot_clickbait_penalty"]
    df["rewrite_vs_original_delta"] = df["rewritten_clickbait_penalty"] - df["original_clickbait_penalty"]
    df["final_title"] = np.where(target_mask, df["rewritten_title_clean"], df["zero_shot_title_clean"])
    df["final_clickbait_penalty"] = np.where(
        target_mask,
        df["rewritten_clickbait_penalty"],
        df["zero_shot_clickbait_penalty"],
    )
    df["final_predicted_clickbait"] = (df["final_clickbait_penalty"] >= args.threshold).astype(int)

    target = df[target_mask].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    report = build_report(df, target, args.output, args.threshold)
    args.report.write_text(report, encoding="utf-8")

    metadata = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(args.output),
        "report": str(args.report),
        "total_rows": len(df),
        "target_rows": int(target_mask.sum()),
        "threshold": args.threshold,
        "target_zero_shot_mean_penalty": float(target["zero_shot_clickbait_penalty"].mean()),
        "target_rewritten_mean_penalty": float(target["rewritten_clickbait_penalty"].mean()),
        "target_mean_delta_vs_zero_shot": float(target["rewrite_vs_zero_shot_delta"].mean()),
        "target_below_threshold_rate": float((target["rewritten_clickbait_penalty"] < args.threshold).mean()),
        "final_mean_penalty": float(df["final_clickbait_penalty"].mean()),
        "final_clickbait_rate": float(df["final_predicted_clickbait"].mean()),
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
