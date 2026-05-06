#!/usr/bin/env python3
"""Build persona-aware calibrated headline selections.

This script adds audience/persona preference signals to the existing multi-agent
candidate matrix, then reselects headlines for each objective with persona-aware
calibration weights.

It is a functional extension of the evaluation/reward workstream:
local critic reward remains the base selector, while persona votes act as an
additional audience-alignment signal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_candidate_matrix.csv"
DEFAULT_BASE_SELECTION = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_objective_selection.csv"
DEFAULT_PERSONA_VOTES = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes.csv"
DEFAULT_OUTPUT_MATRIX = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_candidate_matrix.csv"
DEFAULT_OUTPUT_SELECTION = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_objective_selection.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_objective_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_objective_metadata.json"

PERSONA_SCORE_FIELDS = ["trust", "engagement", "clarity", "audience_fit", "overall"]
OBJECTIVES = ["trust_safety", "growth", "editorial", "specificity"]

CALIBRATION_CONFIG: dict[str, dict[str, Any]] = {
    "trust_safety": {
        "base_score": "objective_trust_safety_score",
        "description": "Trust-sensitive selector: preserve the original safety objective, then boost candidates preferred by trust-sensitive readers.",
        "terms": {
            "trust_sensitive_reader_overall": 0.35,
            "trust_sensitive_reader_trust": 0.25,
            "persona_consensus_best_rate": 0.20,
            "persona_best_rate": 0.10,
        },
    },
    "growth": {
        "base_score": "objective_growth_score",
        "description": "Growth selector: preserve the original growth objective, then boost candidates that growth-oriented readers find engaging.",
        "terms": {
            "growth_reader_overall": 0.30,
            "growth_reader_engagement": 0.35,
            "growth_reader_audience_fit": 0.15,
            "persona_consensus_best_rate": 0.10,
        },
    },
    "editorial": {
        "base_score": "objective_editorial_score",
        "description": "Editorial selector: preserve balanced editorial scoring, then boost candidates preferred by editorial reviewers and busy readers.",
        "terms": {
            "editorial_reviewer_overall": 0.35,
            "editorial_reviewer_clarity": 0.20,
            "busy_news_reader_overall": 0.10,
            "persona_consensus_best_rate": 0.15,
        },
    },
    "specificity": {
        "base_score": "objective_specificity_score",
        "description": "Specificity selector: preserve source-supported detail scoring, then boost candidates that remain clear and audience-fit.",
        "terms": {
            "editorial_reviewer_clarity": 0.20,
            "editorial_reviewer_audience_fit": 0.15,
            "busy_news_reader_clarity": 0.10,
            "persona_mean_audience_fit": 0.15,
        },
    },
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalized_headline(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_clean_persona_votes(path: Path) -> pd.DataFrame:
    votes = pd.read_csv(path)
    votes = votes[votes["judge_error"].fillna("").astype(str).eq("")].copy()
    votes["headline_norm"] = votes["headline"].map(normalized_headline)
    votes["candidate_id"] = votes["candidate_id"].map(clean_text)
    votes["variant"] = votes["variant"].map(clean_text)
    return votes


def build_persona_features(votes: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["seed_id", "variant", "candidate_id", "headline_norm"]

    mean_scores = (
        votes.groupby(group_cols, dropna=False)[PERSONA_SCORE_FIELDS]
        .mean()
        .reset_index()
        .rename(columns={field: f"persona_mean_{field}" for field in PERSONA_SCORE_FIELDS})
    )

    best_rates = (
        votes.groupby(group_cols, dropna=False)
        .agg(
            persona_vote_count=("persona", "count"),
            persona_count=("persona", "nunique"),
            persona_best_rate=("is_persona_best", "mean"),
            persona_consensus_best_rate=("is_consensus_best", "mean"),
        )
        .reset_index()
    )

    persona_pivots = []
    for field in PERSONA_SCORE_FIELDS:
        pivot = votes.pivot_table(
            index=group_cols,
            columns="persona",
            values=field,
            aggfunc="mean",
        )
        pivot.columns = [f"{str(persona)}_{field}" for persona in pivot.columns]
        persona_pivots.append(pivot.reset_index())

    out = mean_scores.merge(best_rates, on=group_cols, how="left")
    for pivot in persona_pivots:
        out = out.merge(pivot, on=group_cols, how="left")
    return out


def add_persona_features(matrix: pd.DataFrame, persona_features: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    out["headline_norm"] = out["headline"].map(normalized_headline)
    out["candidate_id"] = out["candidate_id"].map(clean_text)
    out["variant"] = out["variant"].map(clean_text)
    out = out.merge(
        persona_features,
        on=["seed_id", "variant", "candidate_id", "headline_norm"],
        how="left",
    )
    out["persona_signal_available"] = out["persona_vote_count"].fillna(0).gt(0)
    out["persona_vote_count"] = out["persona_vote_count"].fillna(0).astype(int)
    out["persona_count"] = out["persona_count"].fillna(0).astype(int)
    out["persona_best_rate"] = out["persona_best_rate"].fillna(0.0)
    out["persona_consensus_best_rate"] = out["persona_consensus_best_rate"].fillna(0.0)

    if "pred_overall" in out.columns:
        out["persona_overall_gap"] = out["pred_overall"] - out["persona_mean_overall"]
        out["persona_overestimate_flag"] = out["persona_overall_gap"].ge(0.75)
    else:
        out["persona_overall_gap"] = pd.NA
        out["persona_overestimate_flag"] = False

    if "llm_overall" in out.columns:
        out["llm_overall_gap"] = out["pred_overall"] - out["llm_overall"]
        out["llm_overestimate_flag"] = out["llm_overall_gap"].ge(0.75)
    else:
        out["llm_overall_gap"] = pd.NA
        out["llm_overestimate_flag"] = False
    return out


def centered_persona_value(row: pd.Series, field: str, neutral: float = 3.0) -> float:
    value = row.get(field)
    if pd.isna(value):
        return 0.0
    return float(value) - neutral


def add_calibrated_scores(matrix: pd.DataFrame, calibration_strength: float) -> pd.DataFrame:
    out = matrix.copy()
    for objective, config in CALIBRATION_CONFIG.items():
        base_col = config["base_score"]
        score_col = f"persona_calibrated_{objective}_score"
        raw_adjustment_col = f"persona_{objective}_raw_adjustment"
        adjustment_col = f"persona_{objective}_adjustment"
        raw_adjustments = []
        scaled_adjustments = []
        scores = []
        for _, row in out.iterrows():
            raw_adjustment = 0.0
            for field, weight in config["terms"].items():
                if field.endswith("_rate"):
                    value = row.get(field)
                    if pd.isna(value):
                        value = 0.0
                    raw_adjustment += weight * float(value)
                else:
                    raw_adjustment += weight * centered_persona_value(row, field)
            base = row.get(base_col)
            if pd.isna(base):
                base = 0.0
            scaled_adjustment = calibration_strength * float(raw_adjustment)
            raw_adjustments.append(float(raw_adjustment))
            scaled_adjustments.append(float(scaled_adjustment))
            scores.append(float(base) + float(scaled_adjustment))
        out[raw_adjustment_col] = raw_adjustments
        out[adjustment_col] = scaled_adjustments
        out[score_col] = scores
    return out


def target_persona_columns(objective: str) -> tuple[str, str]:
    if objective == "trust_safety":
        return "trust_sensitive_reader_overall", "trust_sensitive_reader_trust"
    if objective == "growth":
        return "growth_reader_overall", "growth_reader_engagement"
    if objective == "editorial":
        return "editorial_reviewer_overall", "editorial_reviewer_clarity"
    return "editorial_reviewer_overall", "persona_mean_audience_fit"


def select_by_calibrated_objective(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective, config in CALIBRATION_CONFIG.items():
        score_col = f"persona_calibrated_{objective}_score"
        base_score_col = config["base_score"]
        adjustment_col = f"persona_{objective}_adjustment"
        target_overall_col, target_secondary_col = target_persona_columns(objective)
        for seed_id, group in matrix.groupby("seed_id", sort=True):
            group = group[group["headline"].fillna("").astype(str).str.len().gt(0)].copy()
            if group.empty:
                continue
            ranked = group.sort_values(score_col, ascending=False)
            selected = ranked.iloc[0]
            runner_up = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
            rows.append(
                {
                    "objective": objective,
                    "objective_description": config["description"],
                    "seed_id": int(seed_id),
                    "category": selected.get("category"),
                    "selected_variant": selected.get("variant"),
                    "selected_candidate_id": selected.get("candidate_id"),
                    "selected_source": selected.get("candidate_source"),
                    "selected_headline": selected.get("headline"),
                    "selected_score": selected.get(score_col),
                    "selected_base_score": selected.get(base_score_col),
                    "selected_persona_adjustment": selected.get(adjustment_col),
                    "runner_up_variant": runner_up.get("variant"),
                    "runner_up_candidate_id": runner_up.get("candidate_id"),
                    "runner_up_headline": runner_up.get("headline"),
                    "runner_up_score": runner_up.get(score_col),
                    "score_margin": selected.get(score_col) - runner_up.get(score_col),
                    "clickbait_penalty": selected.get("clickbait_penalty"),
                    "pred_faithfulness": selected.get("pred_faithfulness"),
                    "pred_clarity": selected.get("pred_clarity"),
                    "pred_specificity": selected.get("pred_specificity"),
                    "pred_attractiveness": selected.get("pred_attractiveness"),
                    "pred_non_clickbait": selected.get("pred_non_clickbait"),
                    "pred_overall": selected.get("pred_overall"),
                    "pairwise_reward": selected.get("pairwise_reward"),
                    "summary_support_rate": selected.get("summary_support_rate"),
                    "llm_overall": selected.get("llm_overall"),
                    "persona_signal_available": bool(selected.get("persona_signal_available", False)),
                    "persona_mean_overall": selected.get("persona_mean_overall"),
                    "persona_best_rate": selected.get("persona_best_rate"),
                    "persona_consensus_best_rate": selected.get("persona_consensus_best_rate"),
                    "target_persona_overall": selected.get(target_overall_col),
                    "target_persona_secondary": selected.get(target_secondary_col),
                }
            )
    return pd.DataFrame(rows)


def compare_to_base(calibrated_selection: pd.DataFrame, base_selection: pd.DataFrame) -> pd.DataFrame:
    base = base_selection[
        [
            "objective",
            "seed_id",
            "selected_variant",
            "selected_candidate_id",
            "selected_headline",
            "selected_score",
            "llm_overall",
        ]
    ].rename(
        columns={
            "selected_variant": "base_selected_variant",
            "selected_candidate_id": "base_selected_candidate_id",
            "selected_headline": "base_selected_headline",
            "selected_score": "base_selected_score",
            "llm_overall": "base_llm_overall",
        }
    )
    out = calibrated_selection.merge(base, on=["objective", "seed_id"], how="left")
    out["selection_changed"] = (
        out["selected_variant"].astype(str).ne(out["base_selected_variant"].astype(str))
        | out["selected_candidate_id"].astype(str).ne(out["base_selected_candidate_id"].astype(str))
    )
    return out


def build_report(matrix: pd.DataFrame, selection: pd.DataFrame, compared: pd.DataFrame, args: argparse.Namespace) -> str:
    coverage = {
        "candidate_rows": int(len(matrix)),
        "seed_count": int(matrix["seed_id"].nunique()),
        "rows_with_persona_signal": int(matrix["persona_signal_available"].sum()),
        "seeds_with_persona_signal": int(matrix.loc[matrix["persona_signal_available"], "seed_id"].nunique()),
    }

    coverage_table = pd.DataFrame([coverage])

    change_summary = (
        compared.groupby("objective", dropna=False)
        .agg(
            seeds=("seed_id", "nunique"),
            changed_count=("selection_changed", "sum"),
            mean_selected_score=("selected_score", "mean"),
            mean_persona_adjustment=("selected_persona_adjustment", "mean"),
            mean_llm_overall=("llm_overall", "mean"),
            mean_target_persona_overall=("target_persona_overall", "mean"),
            mean_clickbait_penalty=("clickbait_penalty", "mean"),
        )
        .reset_index()
    )
    change_summary["changed_rate"] = change_summary["changed_count"] / change_summary["seeds"]

    variant_counts = (
        selection.groupby(["objective", "selected_variant"], dropna=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["objective", "selected_count"], ascending=[True, False])
    )

    persona_coverage_by_variant = (
        matrix.groupby("variant", dropna=False)
        .agg(
            rows=("headline", "size"),
            rows_with_persona_signal=("persona_signal_available", "sum"),
            mean_persona_overall=("persona_mean_overall", "mean"),
            mean_persona_best_rate=("persona_best_rate", "mean"),
            mean_persona_overall_gap=("persona_overall_gap", "mean"),
        )
        .reset_index()
    )
    persona_coverage_by_variant["persona_coverage_rate"] = persona_coverage_by_variant["rows_with_persona_signal"] / persona_coverage_by_variant["rows"]

    changed_examples = compared[compared["selection_changed"]].copy()
    changed_examples = changed_examples.sort_values(
        ["objective", "selected_persona_adjustment", "score_margin"],
        ascending=[True, False, False],
    ).groupby("objective").head(5)
    example_cols = [
        "objective",
        "seed_id",
        "category",
        "base_selected_variant",
        "base_selected_headline",
        "selected_variant",
        "selected_headline",
        "selected_persona_adjustment",
        "target_persona_overall",
        "score_margin",
    ]

    overestimate_examples = matrix[
        matrix["persona_overestimate_flag"].fillna(False) & matrix["persona_signal_available"]
    ].copy()
    overestimate_examples = overestimate_examples.sort_values("persona_overall_gap", ascending=False).head(12)
    overestimate_cols = [
        "seed_id",
        "category",
        "variant",
        "headline",
        "pred_overall",
        "persona_mean_overall",
        "persona_overall_gap",
        "llm_overall",
    ]

    lines = [
        "# Persona-Calibrated Objective Selection",
        "",
        "This report adds audience/persona preference signals to the existing multi-agent objective selector.",
        "",
        "## Functional Change",
        "",
        "The previous selector used local critic scores, clickbait penalty, pairwise reward, style heuristics, and support heuristics. This extension keeps those base objective scores, then adds persona-specific calibration terms. The default calibration strength is intentionally moderate so persona signals adjust rather than replace local critic scores.",
        "",
        "```text",
        "base objective score",
        "+ persona target preference adjustment",
        "+ consensus / persona-best bonus",
        "= persona-calibrated objective score",
        "```",
        "",
        "## Files",
        "",
        f"- Input matrix: `{display_path(args.matrix)}`",
        f"- Persona votes: `{display_path(args.persona_votes)}`",
        f"- Output matrix: `{display_path(args.output_matrix)}`",
        f"- Output selection: `{display_path(args.output_selection)}`",
        f"- Calibration strength: `{args.calibration_strength:.2f}`",
        "",
        "## Persona Signal Coverage",
        "",
        markdown_table(coverage_table),
        "",
        "## Calibration Terms",
        "",
    ]

    for objective, config in CALIBRATION_CONFIG.items():
        terms = pd.DataFrame([{"signal": key, "weight": value} for key, value in config["terms"].items()])
        lines.extend(
            [
                f"### {objective}",
                "",
                config["description"],
                "",
                markdown_table(terms),
                "",
            ]
        )

    lines.extend(
        [
            "## Selection Change Summary",
            "",
            markdown_table(change_summary),
            "",
            "## Calibrated Selected Variant Counts",
            "",
            markdown_table(variant_counts),
            "",
            "## Persona Coverage By Variant",
            "",
            markdown_table(persona_coverage_by_variant),
            "",
            "## Examples Where Persona Calibration Changed Selection",
            "",
            markdown_table(changed_examples[example_cols]),
            "",
            "## Potential Local Reward Overestimation Examples",
            "",
            "These examples have local `pred_overall` at least 0.75 points higher than the mean persona `overall` score.",
            "",
            markdown_table(overestimate_examples[overestimate_cols]),
            "",
            "## Interpretation",
            "",
            "Persona calibration turns audience votes into an operational selection signal. It does not replace local critics; it adjusts them when a candidate appears better aligned with the target audience for a specific objective.",
            "",
            "This makes the system closer to the intended product: a headline review console where a user can switch between trust, growth, editorial, and specificity goals and see different recommended headlines.",
            "",
            "## Caveats",
            "",
            "- Persona signals are available for only the completed persona-vote subset.",
            "- Persona votes are simulated with an LLM, not collected from real users.",
            "- The calibrated selector should be treated as a demo/control-layer feature, not a production ranking model.",
            "- Missing persona signals are treated as neutral, so unvoted candidates are not directly penalized but receive no persona boost.",
            "",
        ]
    )
    return "\n".join(lines)


def write_metadata(args: argparse.Namespace, matrix: pd.DataFrame, selection: pd.DataFrame, compared: pd.DataFrame) -> None:
    change_summary = (
        compared.groupby("objective", dropna=False)["selection_changed"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "seeds", "sum": "changed_count", "mean": "changed_rate"})
    )
    metadata = {
        "matrix": display_path(args.matrix),
        "base_selection": display_path(args.base_selection),
        "persona_votes": display_path(args.persona_votes),
        "output_matrix": display_path(args.output_matrix),
        "output_selection": display_path(args.output_selection),
        "report": display_path(args.report),
        "calibration_strength": float(args.calibration_strength),
        "rows": int(len(matrix)),
        "seed_count": int(matrix["seed_id"].nunique()),
        "selection_rows": int(len(selection)),
        "rows_with_persona_signal": int(matrix["persona_signal_available"].sum()),
        "seeds_with_persona_signal": int(matrix.loc[matrix["persona_signal_available"], "seed_id"].nunique()),
        "calibration_config": CALIBRATION_CONFIG,
        "change_summary": change_summary.to_dict(orient="records"),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--base-selection", type=Path, default=DEFAULT_BASE_SELECTION)
    parser.add_argument("--persona-votes", type=Path, default=DEFAULT_PERSONA_VOTES)
    parser.add_argument("--output-matrix", type=Path, default=DEFAULT_OUTPUT_MATRIX)
    parser.add_argument("--output-selection", type=Path, default=DEFAULT_OUTPUT_SELECTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--calibration-strength",
        type=float,
        default=0.5,
        help="Scale applied to persona calibration adjustments. Use 1.0 for stronger persona influence.",
    )
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix)
    base_selection = pd.read_csv(args.base_selection)
    votes = load_clean_persona_votes(args.persona_votes)
    persona_features = build_persona_features(votes)

    calibrated_matrix = add_persona_features(matrix, persona_features)
    calibrated_matrix = add_calibrated_scores(calibrated_matrix, args.calibration_strength)
    calibrated_selection = select_by_calibrated_objective(calibrated_matrix)
    compared_selection = compare_to_base(calibrated_selection, base_selection)

    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    calibrated_matrix.to_csv(args.output_matrix, index=False)
    compared_selection.to_csv(args.output_selection, index=False)
    args.report.write_text(build_report(calibrated_matrix, calibrated_selection, compared_selection, args), encoding="utf-8")
    write_metadata(args, calibrated_matrix, calibrated_selection, compared_selection)

    print("Wrote", args.output_matrix)
    print("Wrote", args.output_selection)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(
        json.dumps(
            {
                "rows": int(len(calibrated_matrix)),
                "seed_count": int(calibrated_matrix["seed_id"].nunique()),
                "selection_rows": int(len(calibrated_selection)),
                "rows_with_persona_signal": int(calibrated_matrix["persona_signal_available"].sum()),
                "seeds_with_persona_signal": int(calibrated_matrix.loc[calibrated_matrix["persona_signal_available"], "seed_id"].nunique()),
                "changed_by_objective": compared_selection.groupby("objective")["selection_changed"].sum().astype(int).to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
