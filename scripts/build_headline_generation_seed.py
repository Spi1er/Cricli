#!/usr/bin/env python3
"""Build a small stratified seed set for headline generation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mind_headline_pool_with_clickbait_penalty.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_eval_seed_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_eval_seed_100_profile.md"


TARGET_CATEGORY_COUNTS = {
    "news": 25,
    "sports": 20,
    "finance": 10,
    "lifestyle": 10,
    "health": 10,
    "travel": 8,
    "foodanddrink": 7,
    "weather": 5,
    "autos": 5,
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_seed(df: pd.DataFrame, seed_size: int, random_state: int) -> pd.DataFrame:
    df = df.copy()
    df["summary"] = df["summary"].map(clean_text)
    df["title"] = df["title"].map(clean_text)
    df["category"] = df["category"].map(clean_text)
    df["subvert"] = df["subvert"].map(clean_text)
    df["summary_word_count"] = df["summary"].str.split().map(len)

    eligible = df[
        (df["summary_word_count"] >= 15)
        & (df["title"].str.split().map(len).between(4, 30))
        & df["category"].ne("")
    ].copy()

    parts = []
    used_indices: set[int] = set()
    for category, count in TARGET_CATEGORY_COUNTS.items():
        group = eligible[eligible["category"] == category]
        if group.empty:
            continue
        sample = group.sample(n=min(count, len(group)), random_state=random_state)
        parts.append(sample)
        used_indices.update(sample.index.tolist())

    seed_df = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()
    remaining_needed = seed_size - len(seed_df)
    if remaining_needed > 0:
        remainder = eligible.drop(index=list(used_indices), errors="ignore")
        if len(remainder) > 0:
            extra = remainder.sample(n=min(remaining_needed, len(remainder)), random_state=random_state)
            seed_df = pd.concat([seed_df, extra], ignore_index=False)

    seed_df = seed_df.sample(frac=1, random_state=random_state).head(seed_size).reset_index(drop=True)
    seed_df.insert(0, "seed_id", range(1, len(seed_df) + 1))

    columns = [
        "seed_id",
        "nid",
        "news_id",
        "summary",
        "title",
        "category",
        "subvert",
        "clickbait_penalty",
        "predicted_clickbait",
        "title_word_count",
        "abstract_word_count",
        "body_word_count",
        "url",
    ]
    return seed_df[columns]


def build_report(seed_df: pd.DataFrame, output_path: Path) -> str:
    category_counts = seed_df["category"].value_counts()
    penalty_summary = seed_df["clickbait_penalty"].describe()

    lines = [
        "# Headline Generation Eval Seed Profile",
        "",
        f"- Output: `{output_path}`",
        f"- Rows: {len(seed_df):,}",
        f"- Mean original clickbait penalty: {seed_df['clickbait_penalty'].mean():.4f}",
        f"- Median original clickbait penalty: {seed_df['clickbait_penalty'].median():.4f}",
        f"- Predicted clickbait rate: {seed_df['predicted_clickbait'].mean():.2%}",
        "",
        "## Category Counts",
        "",
        "| Category | Rows |",
        "| --- | ---: |",
    ]
    for category, count in category_counts.items():
        lines.append(f"| {category} | {count:,} |")

    lines.extend(
        [
            "",
            "## Original Penalty Distribution",
            "",
            "```json",
            json.dumps({key: float(value) for key, value in penalty_summary.items()}, indent=2),
            "```",
            "",
            "## Highest Original Clickbait Penalty Examples",
            "",
            "| Penalty | Category | Title |",
            "| ---: | --- | --- |",
        ]
    )

    for row in seed_df.sort_values("clickbait_penalty", ascending=False).head(10).itertuples(index=False):
        title = str(row.title).replace("|", "\\|")
        lines.append(f"| {row.clickbait_penalty:.4f} | {row.category} | {title} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=5293)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    seed_df = build_seed(df, args.seed_size, args.random_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(args.output, index=False)

    report = build_report(seed_df, args.output)
    args.report.write_text(report, encoding="utf-8")

    print("Wrote", args.output, len(seed_df))
    print("Wrote", args.report)
    print(seed_df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
