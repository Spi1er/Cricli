#!/usr/bin/env python3
"""Summarize audience persona voting results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes_metadata.json"

SCORE_FIELDS = ["trust", "engagement", "clarity", "audience_fit", "overall"]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    string_df = df.copy()
    for col in string_df.columns:
        if pd.api.types.is_float_dtype(string_df[col]):
            string_df[col] = string_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            string_df[col] = string_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(string_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(string_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).replace("|", "\\|") for col in string_df.columns) + " |"
        for _, row in string_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_report(votes: pd.DataFrame, input_path: Path) -> str:
    clean = votes[votes["judge_error"].fillna("").astype(str).eq("")].copy()
    completed_seed_count = clean["seed_id"].nunique()
    persona_count = clean["persona"].nunique()

    persona_winners = (
        clean[clean["is_persona_best"].eq(True)]
        .groupby(["persona", "variant"], dropna=False)
        .size()
        .reset_index(name="best_count")
        .sort_values(["persona", "best_count"], ascending=[True, False])
    )
    consensus = (
        clean[clean["is_consensus_best"].eq(True)]
        .drop_duplicates(["seed_id", "candidate_label"])
        .groupby("variant", dropna=False)
        .size()
        .reset_index(name="consensus_best_count")
        .sort_values("consensus_best_count", ascending=False)
    )
    mean_scores = (
        clean.groupby(["persona", "variant"], dropna=False)[SCORE_FIELDS]
        .mean()
        .reset_index()
        .sort_values(["persona", "overall"], ascending=[True, False])
    )
    persona_disagreement = (
        clean[clean["is_persona_best"].eq(True)]
        .groupby("seed_id")["variant"]
        .nunique()
        .reset_index(name="distinct_persona_best_variants")
    )

    examples = (
        clean[clean["is_persona_best"].eq(True)]
        .sort_values(["seed_id", "persona"])
        .head(24)[
            [
                "seed_id",
                "persona",
                "variant",
                "headline",
                "overall",
                "rationale",
            ]
        ]
    )

    lines = [
        "# Audience Persona Voting Profile",
        "",
        f"- Input: `{input_path}`",
        f"- Completed seed count: {completed_seed_count:,}",
        f"- Persona count: {persona_count:,}",
        f"- Vote rows: {len(clean):,}",
        "",
        "## Interpretation",
        "",
        "Persona voting is the proposal-aligned audience layer. It should be treated as an evaluator module that estimates how different audience goals prefer different candidate headlines.",
        "",
        "## Consensus Best Counts",
        "",
        markdown_table(consensus),
        "",
        "## Persona Best Counts",
        "",
        markdown_table(persona_winners),
        "",
        "## Mean Scores By Persona And Variant",
        "",
        markdown_table(mean_scores),
        "",
        "## Persona Disagreement",
        "",
        markdown_table(persona_disagreement["distinct_persona_best_variants"].value_counts().rename_axis("distinct_best_variants").reset_index(name="seed_count")),
        "",
        "## Example Persona Winners",
        "",
        markdown_table(examples),
        "",
        "## Next Use",
        "",
        "- Merge persona `overall`, `trust`, and `engagement` scores into the multi-agent candidate matrix.",
        "- Compare objective selectors with and without persona rewards.",
        "- Use persona disagreement as evidence that headline quality is audience-dependent.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    votes = pd.read_csv(args.input)
    clean = votes[votes["judge_error"].fillna("").astype(str).eq("")].copy()
    args.report.write_text(build_report(votes, args.input), encoding="utf-8")
    metadata = {
        "input": str(args.input),
        "report": str(args.report),
        "rows": int(len(votes)),
        "clean_rows": int(len(clean)),
        "completed_seed_count": int(clean["seed_id"].nunique()),
        "persona_count": int(clean["persona"].nunique()),
        "personas": sorted(clean["persona"].dropna().unique().tolist()),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
