#!/usr/bin/env python3
"""Analyze agentic v3 wins, losses, and reward/judge misalignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_v3_specificity_scores.csv"
DEFAULT_LOCAL = PROJECT_ROOT / "data" / "processed" / "headline_agentic_v3_specificity_vs_baselines_eval.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "headline_agentic_v3_error_analysis.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_agentic_v3_error_analysis.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_agentic_v3_error_analysis_metadata.json"

DIMENSIONS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def pivot_judge(judge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed_id, group in judge.groupby("seed_id"):
        base = group.iloc[0]
        row = {
            "seed_id": int(seed_id),
            "category": base["category"],
            "summary": base["summary"],
            "best_variant": base["best_variant"],
            "worst_variant": base["worst_variant"],
        }
        for variant in ["original", "zero_shot", "optimized", "agentic_selected"]:
            match = group[group["variant"].eq(variant)]
            if match.empty:
                continue
            item = match.iloc[0]
            row[f"{variant}_headline"] = item["headline"]
            row[f"{variant}_rationale"] = item["rationale"]
            row[f"{variant}_clickbait_penalty"] = item["clickbait_penalty"]
            for dim in DIMENSIONS:
                row[f"{variant}_{dim}"] = item[dim]
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_local(local: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variant_map = {
        "zero_shot": "zero_shot",
        "optimized": "round2_final",
        "agentic_selected": "agentic_selected",
    }
    for seed_id, group in local.groupby("seed_id"):
        row = {"seed_id": int(seed_id)}
        for out_variant, in_variant in variant_map.items():
            match = group[group["variant"].eq(in_variant)]
            if match.empty:
                continue
            item = match.iloc[0]
            row[f"{out_variant}_local_final_score"] = item["final_score"]
            row[f"{out_variant}_local_quality_reward"] = item["quality_reward"]
            row[f"{out_variant}_local_pairwise_reward"] = item["pairwise_reward"]
            row[f"{out_variant}_local_pred_faithfulness"] = item["pred_faithfulness"]
            row[f"{out_variant}_local_pred_clarity"] = item["pred_clarity"]
            row[f"{out_variant}_local_pred_specificity"] = item["pred_specificity"]
            row[f"{out_variant}_local_pred_overall"] = item["pred_overall"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_analysis(judge: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    out = pivot_judge(judge).merge(pivot_local(local), on="seed_id", how="left")
    out["agentic_vs_zero_overall_delta"] = out["agentic_selected_overall"] - out["zero_shot_overall"]
    out["agentic_vs_optimized_overall_delta"] = out["agentic_selected_overall"] - out["optimized_overall"]
    out["agentic_vs_zero_faithfulness_delta"] = out["agentic_selected_faithfulness"] - out["zero_shot_faithfulness"]
    out["agentic_vs_zero_clarity_delta"] = out["agentic_selected_clarity"] - out["zero_shot_clarity"]
    out["agentic_vs_zero_specificity_delta"] = out["agentic_selected_specificity"] - out["zero_shot_specificity"]
    out["agentic_vs_zero_attractiveness_delta"] = out["agentic_selected_attractiveness"] - out["zero_shot_attractiveness"]
    out["agentic_vs_zero_non_clickbait_delta"] = out["agentic_selected_non_clickbait"] - out["zero_shot_non_clickbait"]
    out["agentic_vs_zero_local_delta"] = out["agentic_selected_local_final_score"] - out["zero_shot_local_final_score"]
    out["agentic_vs_optimized_local_delta"] = out["agentic_selected_local_final_score"] - out["optimized_local_final_score"]

    out["case_type"] = "tie_or_mixed"
    out.loc[out["agentic_vs_zero_overall_delta"] > 0, "case_type"] = "agentic_beats_zero_shot"
    out.loc[out["agentic_vs_zero_overall_delta"] < 0, "case_type"] = "zero_shot_beats_agentic"
    out.loc[
        (out["agentic_vs_zero_local_delta"] > 0) & (out["agentic_vs_zero_overall_delta"] < 0),
        "case_type",
    ] = "local_reward_overestimates_agentic"
    out.loc[
        (out["agentic_vs_zero_local_delta"] < 0) & (out["agentic_vs_zero_overall_delta"] > 0),
        "case_type",
    ] = "local_reward_underestimates_agentic"

    loss_dims = []
    for row in out.itertuples(index=False):
        dims = []
        for dim in ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait"]:
            delta = getattr(row, f"agentic_vs_zero_{dim}_delta")
            if delta < 0:
                dims.append(dim)
        loss_dims.append(", ".join(dims))
    out["agentic_loss_dimensions_vs_zero"] = loss_dims
    return out


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


def compact_examples(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    cols = [
        "seed_id",
        "category",
        "agentic_vs_zero_overall_delta",
        "agentic_vs_zero_local_delta",
        "agentic_loss_dimensions_vs_zero",
        "zero_shot_headline",
        "agentic_selected_headline",
        "zero_shot_rationale",
        "agentic_selected_rationale",
    ]
    return df[cols].head(n).copy()


def write_report(analysis: pd.DataFrame, report: Path, output: Path) -> None:
    case_counts = analysis["case_type"].value_counts().rename_axis("case_type").reset_index(name="count")
    case_counts["rate"] = case_counts["count"] / len(analysis)

    dim_counts = (
        analysis["agentic_loss_dimensions_vs_zero"]
        .str.get_dummies(sep=", ")
        .sum()
        .sort_values(ascending=False)
        .rename_axis("dimension")
        .reset_index(name="loss_count")
    )

    wins = analysis[analysis["agentic_vs_zero_overall_delta"] > 0].sort_values(
        "agentic_vs_zero_overall_delta", ascending=False
    )
    losses = analysis[analysis["agentic_vs_zero_overall_delta"] < 0].sort_values(
        "agentic_vs_zero_overall_delta", ascending=True
    )
    overestimates = analysis[
        (analysis["agentic_vs_zero_local_delta"] > 0) & (analysis["agentic_vs_zero_overall_delta"] < 0)
    ].sort_values("agentic_vs_zero_local_delta", ascending=False)
    underestimates = analysis[
        (analysis["agentic_vs_zero_local_delta"] < 0) & (analysis["agentic_vs_zero_overall_delta"] > 0)
    ].sort_values("agentic_vs_zero_local_delta", ascending=True)

    lines = [
        "# Agentic V3 Error Analysis",
        "",
        f"- Analysis CSV: `{output}`",
        f"- Seeds analyzed: {len(analysis):,}",
        "",
        "## Case Counts",
        "",
        markdown_table(case_counts),
        "",
        "## Dimensions Where Agentic Loses To Zero-Shot",
        "",
        markdown_table(dim_counts),
        "",
        "## Strong Agentic Wins Over Zero-Shot",
        "",
        markdown_table(compact_examples(wins)),
        "",
        "## Strong Agentic Losses To Zero-Shot",
        "",
        markdown_table(compact_examples(losses)),
        "",
        "## Local Reward Overestimates Agentic",
        "",
        "These are reward-misalignment examples: local reward prefers agentic, but the LLM judge gives lower overall score than zero-shot.",
        "",
        markdown_table(compact_examples(overestimates)),
        "",
        "## Local Reward Underestimates Agentic",
        "",
        "These are cases where the judge prefers agentic, but local reward does not. They may reveal missed reward features.",
        "",
        markdown_table(compact_examples(underestimates)),
        "",
        "## Takeaways",
        "",
        "- V3 improved by generating more specific candidate headlines, but the main remaining risks are faithfulness and clarity.",
        "- Reward misalignment still exists: the local v2 reward can favor detailed agentic titles that the LLM judge sees as less faithful or less clear.",
        "- The next model-side improvement should emphasize source-grounded specificity: concrete details are useful only when they are explicitly supported by the summary.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    judge = pd.read_csv(args.judge)
    local = pd.read_csv(args.local)
    analysis = build_analysis(judge, local)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output, index=False)
    write_report(analysis, args.report, args.output)

    metadata = {
        "judge": str(args.judge),
        "local": str(args.local),
        "output": str(args.output),
        "report": str(args.report),
        "seed_count": int(len(analysis)),
        "case_counts": analysis["case_type"].value_counts().to_dict(),
        "mean_agentic_vs_zero_overall_delta": float(analysis["agentic_vs_zero_overall_delta"].mean()),
        "mean_agentic_vs_zero_local_delta": float(analysis["agentic_vs_zero_local_delta"].mean()),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
