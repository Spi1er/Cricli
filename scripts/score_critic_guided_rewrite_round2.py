#!/usr/bin/env python3
"""Score second-round critic-guided rewrites and build final comparison."""

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_100.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_scored_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_profile.md"


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
    lines = [
        "# Round-2 Critic-Guided Rewrite Profile",
        "",
        f"- Output: `{output}`",
        f"- Total rows: {len(df):,}",
        f"- Round-2 target rows: {len(target):,}",
        f"- Threshold: {threshold:.2f}",
        "",
        "## Full 100-Row Comparison",
        "",
        "| Stage | Mean penalty | Clickbait rate |",
        "| --- | ---: | ---: |",
        f"| Original | {df['original_clickbait_penalty'].mean():.4f} | {df['original_predicted_clickbait'].mean():.2%} |",
        f"| Zero-shot | {df['zero_shot_clickbait_penalty'].mean():.4f} | {df['zero_shot_predicted_clickbait'].mean():.2%} |",
        f"| Round-1 final | {df['final_clickbait_penalty'].mean():.4f} | {df['final_predicted_clickbait'].mean():.2%} |",
        f"| Round-2 final | {df['round2_final_clickbait_penalty'].mean():.4f} | {df['round2_final_predicted_clickbait'].mean():.2%} |",
        "",
        "## Round-2 Target Rows Only",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Round-1 rewritten mean penalty | {target['rewritten_clickbait_penalty'].mean():.4f} |",
        f"| Round-2 mean penalty | {target['round2_clickbait_penalty'].mean():.4f} |",
        f"| Mean delta vs round 1 | {target['round2_vs_round1_delta'].mean():.4f} |",
        f"| Median delta vs round 1 | {target['round2_vs_round1_delta'].median():.4f} |",
        f"| Rows improved vs round 1 | {(target['round2_vs_round1_delta'] < 0).mean():.2%} |",
        f"| Rows below threshold after round 2 | {(target['round2_clickbait_penalty'] < threshold).mean():.2%} |",
        "",
        "## Round-2 Examples",
        "",
        "| Seed | Category | Round-1 penalty | Round-2 penalty | Delta | Round-1 title | Round-2 title |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for row in target.sort_values("round2_vs_round1_delta").itertuples(index=False):
        round1 = str(row.rewritten_title_clean).replace("|", "\\|")
        round2 = str(row.round2_title_clean).replace("|", "\\|")
        lines.append(
            f"| {row.seed_id} | {row.category} | {row.rewritten_clickbait_penalty:.4f} | "
            f"{row.round2_clickbait_penalty:.4f} | {row.round2_vs_round1_delta:.4f} | "
            f"{round1} | {round2} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Round 2 adds stricter lexical constraints after the clickbait critic still flags a title. This shows both the benefit and the limitation of prompt-based rewriting: some phrases remain highly scored because the penalty model has learned genre/style cues from lifestyle and listicle headlines.",
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
    df["round2_title_clean"] = df["round2_title"].map(clean_title)

    target_mask = df["rewritten_clickbait_penalty"] >= args.threshold
    missing = target_mask & df["round2_title_clean"].eq("")
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} target rows are missing round2_title.")

    print(f"Scoring {int(target_mask.sum()):,} round-2 titles on {device}...")
    round2_probs = np.full(len(df), np.nan, dtype=np.float32)
    round2_probs[target_mask.to_numpy()] = score_titles(
        df.loc[target_mask, "round2_title_clean"].tolist(),
        args.model,
        args.batch_size,
        args.max_length,
        device,
    )

    df["round2_clickbait_penalty"] = round2_probs
    df["round2_predicted_clickbait"] = np.where(
        df["round2_clickbait_penalty"].notna(),
        (df["round2_clickbait_penalty"] >= args.threshold).astype(int),
        np.nan,
    )
    df["round2_vs_round1_delta"] = df["round2_clickbait_penalty"] - df["rewritten_clickbait_penalty"]
    df["round2_final_title"] = np.where(target_mask, df["round2_title_clean"], df["final_title"])
    df["round2_final_clickbait_penalty"] = np.where(
        target_mask,
        df["round2_clickbait_penalty"],
        df["final_clickbait_penalty"],
    )
    df["round2_final_predicted_clickbait"] = (df["round2_final_clickbait_penalty"] >= args.threshold).astype(int)

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
        "round2_target_rows": int(target_mask.sum()),
        "threshold": args.threshold,
        "round2_target_round1_mean_penalty": float(target["rewritten_clickbait_penalty"].mean()),
        "round2_target_round2_mean_penalty": float(target["round2_clickbait_penalty"].mean()),
        "round2_target_mean_delta": float(target["round2_vs_round1_delta"].mean()),
        "round2_target_below_threshold_rate": float((target["round2_clickbait_penalty"] < args.threshold).mean()),
        "round2_final_mean_penalty": float(df["round2_final_clickbait_penalty"].mean()),
        "round2_final_clickbait_rate": float(df["round2_final_predicted_clickbait"].mean()),
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
