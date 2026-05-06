#!/usr/bin/env python3
"""Analyze LLM-judge results for generic vs specificity-aware SFT headline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_sft_scores.csv"
DEFAULT_LOCAL = PROJECT_ROOT / "data" / "processed" / "headline_sft_generators_eval.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_sft_judge_error_analysis.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_sft_judge_error_analysis.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_sft_judge_error_analysis_metadata.json"

VARIANTS = ["original", "generic_sft", "specificity_sft"]
SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def wide_scores(judge: pd.DataFrame) -> pd.DataFrame:
    columns = ["seed_id", "variant", "headline", "rationale", *SCORE_FIELDS]
    frame = judge[columns].copy()
    pieces = []
    for variant in VARIANTS:
        part = frame[frame["variant"].eq(variant)].copy()
        rename = {
            "headline": f"{variant}_headline",
            "rationale": f"{variant}_rationale",
        }
        rename.update({field: f"{variant}_{field}" for field in SCORE_FIELDS})
        pieces.append(part.drop(columns=["variant"]).rename(columns=rename))
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="seed_id", how="outer")
    meta_cols = ["seed_id", "category", "summary"]
    meta = judge[meta_cols].drop_duplicates("seed_id")
    return meta.merge(out, on="seed_id", how="left")


def add_local_features(analysis: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    local_cols = [
        "seed_id",
        "variant",
        "clickbait_penalty",
        "quality_reward",
        "pairwise_reward",
        "final_score",
        "reference_token_f1",
        "summary_support_rate",
        "headline_word_count",
        "has_specific_signal",
    ]
    local = local[[col for col in local_cols if col in local.columns]].copy()
    for variant in VARIANTS:
        part = local[local["variant"].eq(variant)].copy().drop(columns=["variant"])
        part = part.rename(columns={col: f"{variant}_{col}" for col in part.columns if col != "seed_id"})
        analysis = analysis.merge(part, on="seed_id", how="left")
    return analysis


def classify_row(row: pd.Series) -> str:
    original = float(row["original_overall"])
    generic = float(row["generic_sft_overall"])
    specificity = float(row["specificity_sft_overall"])

    if specificity > generic and specificity >= original:
        return "specificity_best"
    if generic > specificity and generic >= original:
        return "generic_best"
    if original > generic and original > specificity:
        if specificity > generic:
            return "original_best_specificity_above_generic"
        if generic > specificity:
            return "original_best_generic_above_specificity"
        return "original_best_sft_tie"
    if specificity > generic:
        return "specificity_above_generic_mixed"
    if generic > specificity:
        return "generic_above_specificity_mixed"
    return "sft_tie"


def add_deltas(analysis: pd.DataFrame) -> pd.DataFrame:
    out = analysis.copy()
    for field in SCORE_FIELDS:
        out[f"specificity_minus_generic_{field}"] = out[f"specificity_sft_{field}"] - out[f"generic_sft_{field}"]
        out[f"specificity_minus_original_{field}"] = out[f"specificity_sft_{field}"] - out[f"original_{field}"]
        out[f"generic_minus_original_{field}"] = out[f"generic_sft_{field}"] - out[f"original_{field}"]
    out["failure_group"] = out.apply(classify_row, axis=1)
    return out


def mean_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison in ["specificity_minus_generic", "specificity_minus_original", "generic_minus_original"]:
        row = {"comparison": comparison}
        for field in SCORE_FIELDS:
            row[f"mean_{field}_delta"] = float(df[f"{comparison}_{field}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


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


def build_report(analysis: pd.DataFrame, output: Path) -> str:
    group_counts = (
        analysis["failure_group"]
        .value_counts()
        .rename_axis("failure_group")
        .reset_index(name="rows")
    )
    category = (
        analysis.groupby("category", dropna=False)
        .agg(
            rows=("seed_id", "size"),
            mean_specificity_minus_generic=("specificity_minus_generic_overall", "mean"),
            mean_specificity_minus_original=("specificity_minus_original_overall", "mean"),
            mean_generic_minus_original=("generic_minus_original_overall", "mean"),
        )
        .sort_values("rows", ascending=False)
        .head(12)
        .reset_index()
    )

    local_alignment = pd.DataFrame(
        [
            {
                "comparison": "specificity_sft - generic_sft",
                "mean_llm_overall_delta": analysis["specificity_minus_generic_overall"].mean(),
                "mean_local_final_delta": (
                    analysis["specificity_sft_final_score"] - analysis["generic_sft_final_score"]
                ).mean(),
            },
            {
                "comparison": "specificity_sft - original",
                "mean_llm_overall_delta": analysis["specificity_minus_original_overall"].mean(),
                "mean_local_final_delta": (
                    analysis["specificity_sft_final_score"] - analysis["original_final_score"]
                ).mean(),
            },
            {
                "comparison": "generic_sft - original",
                "mean_llm_overall_delta": analysis["generic_minus_original_overall"].mean(),
                "mean_local_final_delta": (
                    analysis["generic_sft_final_score"] - analysis["original_final_score"]
                ).mean(),
            },
        ]
    )

    examples = analysis.sort_values("specificity_minus_generic_overall", ascending=False).head(10)
    examples = examples[
        [
            "seed_id",
            "category",
            "specificity_minus_generic_overall",
            "original_headline",
            "generic_sft_headline",
            "specificity_sft_headline",
            "specificity_sft_rationale",
        ]
    ]

    worst_examples = analysis.sort_values("specificity_minus_original_overall", ascending=True).head(10)
    worst_examples = worst_examples[
        [
            "seed_id",
            "category",
            "specificity_minus_original_overall",
            "original_headline",
            "specificity_sft_headline",
            "specificity_sft_rationale",
        ]
    ]

    lines = [
        "# SFT Judge Error Analysis",
        "",
        f"- Input rows: {len(analysis):,}",
        f"- Output: `{output}`",
        "",
        "## Main Result",
        "",
        "The specificity-aware SFT model is slightly better than the generic SFT model on average, but both SFT generators are judged worse than the original human-written headlines. This means the current SFT step improves over a naive generator setup but has not yet matched the editorial target distribution.",
        "",
        "## Mean Judge Deltas",
        "",
        markdown_table(mean_delta_table(analysis)),
        "",
        "## Local Critic vs LLM Judge Alignment",
        "",
        markdown_table(local_alignment),
        "",
        "## Failure Groups",
        "",
        markdown_table(group_counts),
        "",
        "## Category Breakdown",
        "",
        markdown_table(category),
        "",
        "## Where Specificity SFT Beats Generic SFT",
        "",
        markdown_table(examples),
        "",
        "## Worst Specificity SFT vs Original Examples",
        "",
        markdown_table(worst_examples),
        "",
        "## Implications",
        "",
        "- The SFT models often produce summary-like or overlong headlines, so the next SFT data pass should control headline style and length more tightly.",
        "- Specificity-aware filtering helps specificity and non-clickbait scores, but it does not solve attractiveness or editorial sharpness.",
        "- Local critics overestimated SFT outputs versus original titles, so the reward model should be updated with these SFT judge labels before being used for agentic reranking.",
        "- The next model improvement should use SFT labels and judge feedback before adding more complex agents.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    judge = pd.read_csv(args.judge)
    local = pd.read_csv(args.local)
    analysis = wide_scores(judge)
    analysis = add_local_features(analysis, local)
    analysis = add_deltas(analysis)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output, index=False)
    args.report.write_text(build_report(analysis, args.output), encoding="utf-8")

    metadata = {
        "judge": str(args.judge),
        "local": str(args.local),
        "output": str(args.output),
        "report": str(args.report),
        "rows": int(len(analysis)),
        "failure_groups": {
            str(k): int(v) for k, v in analysis["failure_group"].value_counts().to_dict().items()
        },
        "mean_deltas": mean_delta_table(analysis).to_dict(orient="records"),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
