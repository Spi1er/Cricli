#!/usr/bin/env python3
"""Build simplified headline review demo cases.

This script converts the full research candidate matrix into a product-facing
review dataset. The product demo exposes only a few meaningful options per
article/objective while keeping the larger research pool hidden:

- Human baseline
- GenAI baseline
- Low-risk alternative
- Recommended

It also folds the earlier overlapping critics into a unified scorecard and adds
short "why selected / why not selected" explanations for the demo UI.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_candidate_matrix.csv"
DEFAULT_SELECTION = PROJECT_ROOT / "data" / "processed" / "headline_persona_calibrated_objective_selection.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_cases.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_metadata.json"

OBJECTIVES = ["trust_safety", "growth", "editorial", "specificity"]
DISPLAY_LABEL_ORDER = {
    "Human baseline": 1,
    "GenAI baseline": 2,
    "Low-risk alternative": 3,
    "Recommended": 4,
}
OBJECTIVE_NAMES = {
    "trust_safety": "Trust / Safety",
    "growth": "Growth",
    "editorial": "Editorial",
    "specificity": "Specificity",
}
OBJECTIVE_EXPLANATIONS = {
    "trust_safety": "Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces.",
    "growth": "Prioritizes engaging headlines while still controlling clickbait and trust risk.",
    "editorial": "Prioritizes balanced, compact, publication-ready news headlines.",
    "specificity": "Prioritizes concrete, source-supported details without losing clarity.",
}

# Product-facing display weights. The underlying research selector still comes
# from the persona-calibrated matrix; these weights make the demo explanation
# legible to non-technical users.
DEMO_DECISION_WEIGHTS = {
    "trust_safety": {"quality": 0.30, "risk": 0.35, "audience": 0.15, "objective": 0.20},
    "growth": {"quality": 0.25, "risk": 0.20, "audience": 0.25, "objective": 0.30},
    "editorial": {"quality": 0.35, "risk": 0.25, "audience": 0.15, "objective": 0.25},
    "specificity": {"quality": 0.30, "risk": 0.15, "audience": 0.10, "objective": 0.45},
}

CALIBRATED_BASE_SCORE_COLS = {
    "trust_safety": "objective_trust_safety_score",
    "growth": "objective_growth_score",
    "editorial": "objective_editorial_score",
    "specificity": "objective_specificity_score",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalized_headline(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def first_row(group: pd.DataFrame, variant: str) -> pd.Series | None:
    match = group[group["variant"].eq(variant)]
    if match.empty:
        return None
    return match.iloc[0]


def selected_row_from_selection(group: pd.DataFrame, selection_row: pd.Series) -> pd.Series | None:
    variant = clean_text(selection_row.get("selected_variant"))
    candidate_id = clean_text(selection_row.get("selected_candidate_id"))
    headline_norm = normalized_headline(selection_row.get("selected_headline"))

    match = group[
        group["variant"].astype(str).eq(variant)
        & group["candidate_id"].astype(str).eq(candidate_id)
    ]
    if match.empty:
        match = group[group["headline_norm"].eq(headline_norm)]
    if match.empty:
        return None
    return match.iloc[0]


def low_risk_candidate(group: pd.DataFrame) -> pd.Series | None:
    eligible = group[group["variant"].ne("original") & group["variant"].ne("zero_shot")].copy()
    if eligible.empty:
        return None
    eligible = eligible[eligible["headline"].fillna("").astype(str).str.len().gt(0)]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["clickbait_penalty", "pred_overall", "summary_support_rate"],
        ascending=[True, False, False],
    ).iloc[0]


def safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_0_to_1(value: object, denominator: float = 5.0) -> float:
    return max(0.0, min(1.0, safe_float(value) / denominator))


def blended_quality_score(row: pd.Series) -> float:
    predicted_quality = score_0_to_1(row.get("pred_overall"))
    support = max(0.0, min(1.0, safe_float(row.get("summary_support_rate"))))
    return round((0.70 * predicted_quality) + (0.30 * support), 6)


def audience_fit_score(row: pd.Series) -> float:
    if pd.isna(row.get("persona_mean_overall")):
        return 0.50
    return score_0_to_1(row.get("persona_mean_overall"))


def objective_fit_score(row: pd.Series, objective: str) -> float:
    return score_0_to_1(row.get(f"persona_calibrated_{objective}_score"), denominator=6.0)


def unified_decision_score(record: dict[str, Any], objective: str) -> float:
    weights = DEMO_DECISION_WEIGHTS.get(objective, DEMO_DECISION_WEIGHTS["editorial"])
    score = (
        weights["quality"] * safe_float(record.get("quality_score"), 0.5)
        + weights["risk"] * safe_float(record.get("risk_score"), 0.5)
        + weights["audience"] * safe_float(record.get("audience_score"), 0.5)
        + weights["objective"] * safe_float(record.get("objective_fit_score"), 0.5)
    )
    return round(score, 6)


def role_reason(role: str, objective: str) -> str:
    if role == "Human baseline":
        return "Original human-written headline used as the editorial reference point."
    if role == "GenAI baseline":
        return "Direct GenAI headline used as the strongest simple generation baseline."
    if role == "Low-risk alternative":
        return "Selected from the hidden candidate pool for low clickbait risk while keeping predicted quality."
    if role == "Recommended":
        return f"Selected by the persona-calibrated {OBJECTIVE_NAMES.get(objective, objective)} objective."
    return "Candidate headline."


def recommendation_summary(row: pd.Series, objective: str) -> str:
    parts = [OBJECTIVE_EXPLANATIONS.get(objective, "Selected by the current objective.")]
    if safe_float(row.get("clickbait_penalty")) <= 0.05:
        parts.append("Risk is low after folding clickbait into the safety score.")
    if not pd.isna(row.get("persona_mean_overall")) and safe_float(row.get("persona_mean_overall"), default=-1) >= 4.0:
        parts.append("Audience/persona scoring favors it.")
    if safe_float(row.get("pred_overall")) >= 4.5:
        parts.append("The local quality critic rates it highly.")
    if safe_float(row.get("summary_support_rate")) >= 0.5:
        parts.append("Its main terms are supported by the article summary.")
    return " ".join(parts)


def gap_text(delta: float) -> str:
    return f"{abs(delta):.2f}"


def rejection_summary(record: dict[str, Any], recommended: dict[str, Any], objective: str) -> str:
    if record.get("is_recommended"):
        return "Selected as the best fit for the current business objective."

    reasons: list[str] = []
    quality_gap = safe_float(recommended.get("quality_score"), 0.5) - safe_float(record.get("quality_score"), 0.5)
    risk_gap = safe_float(recommended.get("risk_score"), 0.5) - safe_float(record.get("risk_score"), 0.5)
    audience_gap = safe_float(recommended.get("audience_score"), 0.5) - safe_float(record.get("audience_score"), 0.5)
    objective_gap = safe_float(recommended.get("objective_fit_score"), 0.5) - safe_float(record.get("objective_fit_score"), 0.5)
    support_gap = safe_float(recommended.get("support_score"), 0.5) - safe_float(record.get("support_score"), 0.5)

    if risk_gap >= 0.10:
        reasons.append(f"higher risk by {gap_text(risk_gap)}")
    if quality_gap >= 0.05:
        reasons.append(f"lower quality by {gap_text(quality_gap)}")
    if objective_gap >= 0.05:
        reasons.append(f"weaker {OBJECTIVE_NAMES.get(objective, objective)} objective fit by {gap_text(objective_gap)}")
    if audience_gap >= 0.08:
        reasons.append(f"weaker audience fit by {gap_text(audience_gap)}")
    if support_gap >= 0.12:
        reasons.append(f"less summary support by {gap_text(support_gap)}")

    if reasons:
        return "Not selected because it has " + "; ".join(reasons[:3]) + "."
    if record.get("display_label") == "Human baseline":
        return "Kept as the editorial reference; the recommendation has a stronger combined decision score."
    if record.get("display_label") == "GenAI baseline":
        return "Strong baseline candidate, but the selector found a better match for this objective."
    if record.get("display_label") == "Low-risk alternative":
        return "Safe alternative, but the recommendation gives a better overall tradeoff."
    return "Close alternative, but not the best combined tradeoff for this objective."


def make_display_record(
    row: pd.Series,
    objective: str,
    role: str,
    is_recommended: bool,
    hidden_pool_size: int,
    hidden_source_count: int,
) -> dict[str, Any]:
    objective_score_col = f"persona_calibrated_{objective}_score"
    adjustment_col = f"persona_{objective}_adjustment"
    support = max(0.0, min(1.0, safe_float(row.get("summary_support_rate"))))
    persona_missing = pd.isna(row.get("persona_mean_overall"))
    return {
        "seed_id": int(row["seed_id"]),
        "objective": objective,
        "objective_name": OBJECTIVE_NAMES.get(objective, objective),
        "category": row.get("category"),
        "summary": row.get("summary"),
        "display_label": role,
        "display_order": DISPLAY_LABEL_ORDER[role],
        "variant": row.get("variant"),
        "candidate_id": row.get("candidate_id"),
        "candidate_source": row.get("candidate_source"),
        "headline": row.get("headline"),
        "is_recommended": bool(is_recommended),
        "recommendation_summary": recommendation_summary(row, objective) if is_recommended else role_reason(role, objective),
        "quality_score": blended_quality_score(row),
        "risk_score": 1.0 - max(0.0, min(1.0, safe_float(row.get("clickbait_penalty")))),
        "audience_score": audience_fit_score(row),
        "audience_signal": "neutral fallback" if persona_missing else "persona voted",
        "objective_fit_score": objective_fit_score(row, objective),
        "support_score": support,
        "evidence_support_score": support,
        "recommendation_score": row.get(objective_score_col),
        "base_objective_score": row.get(CALIBRATED_BASE_SCORE_COLS[objective]),
        "persona_adjustment": row.get(adjustment_col),
        "clickbait_penalty": row.get("clickbait_penalty"),
        "pred_overall": row.get("pred_overall"),
        "persona_mean_overall": row.get("persona_mean_overall"),
        "summary_support_rate": row.get("summary_support_rate"),
        "llm_overall": row.get("llm_overall"),
        "hidden_candidate_pool_size": hidden_pool_size,
        "hidden_candidate_source_count": hidden_source_count,
    }


def annotate_decision_explanations(records: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    if not records:
        return records
    for record in records:
        record["unified_decision_score"] = unified_decision_score(record, objective)
    recommended = next((record for record in records if record.get("is_recommended")), records[0])
    for record in records:
        record["decision_explanation"] = (
            record.get("recommendation_summary", "")
            if record.get("is_recommended")
            else rejection_summary(record, recommended, objective)
        )
    return records


def build_demo_cases(matrix: pd.DataFrame, selection: pd.DataFrame, limit_seeds: int | None = None) -> pd.DataFrame:
    matrix = matrix.copy()
    matrix["headline"] = matrix["headline"].map(clean_text)
    matrix["headline_norm"] = matrix["headline"].map(normalized_headline)
    matrix["variant"] = matrix["variant"].map(clean_text)
    matrix["candidate_id"] = matrix["candidate_id"].map(clean_text)

    seed_ids = sorted(matrix["seed_id"].dropna().astype(int).unique().tolist())
    if limit_seeds:
        seed_ids = seed_ids[:limit_seeds]

    selection_lookup = {
        (str(row["objective"]), int(row["seed_id"])): row
        for _, row in selection.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for seed_id in seed_ids:
        group = matrix[matrix["seed_id"].astype(int).eq(seed_id)].copy()
        hidden_pool_size = int(len(group))
        hidden_source_count = int(group["candidate_source"].nunique())
        for objective in OBJECTIVES:
            selected_meta = selection_lookup.get((objective, seed_id))
            chosen_roles: list[tuple[str, pd.Series | None]] = [
                ("Human baseline", first_row(group, "original")),
                ("GenAI baseline", first_row(group, "zero_shot")),
                ("Low-risk alternative", low_risk_candidate(group)),
            ]
            selected = selected_row_from_selection(group, selected_meta) if selected_meta is not None else None
            chosen_roles.append(("Recommended", selected))

            by_headline: dict[str, dict[str, Any]] = {}
            for role, candidate in chosen_roles:
                if candidate is None:
                    continue
                norm = normalized_headline(candidate.get("headline"))
                if not norm:
                    continue
                is_recommended = role == "Recommended"
                if norm not in by_headline:
                    by_headline[norm] = make_display_record(
                        candidate,
                        objective,
                        role,
                        is_recommended,
                        hidden_pool_size,
                        hidden_source_count,
                    )
                    by_headline[norm]["roles"] = role
                else:
                    existing = by_headline[norm]
                    if role not in existing["roles"].split("; "):
                        existing["roles"] = existing["roles"] + "; " + role
                    existing["is_recommended"] = bool(existing["is_recommended"] or is_recommended)
                    if is_recommended:
                        updated = make_display_record(
                            candidate,
                            objective,
                            "Recommended",
                            True,
                            hidden_pool_size,
                            hidden_source_count,
                        )
                        roles = existing["roles"]
                        existing.update(updated)
                        existing["roles"] = roles if "Recommended" in roles.split("; ") else roles + "; Recommended"
                    elif not existing["is_recommended"]:
                        existing["display_order"] = min(existing["display_order"], DISPLAY_LABEL_ORDER[role])
            rows.extend(annotate_decision_explanations(list(by_headline.values()), objective))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["seed_id", "objective", "display_order", "headline"]).reset_index(drop=True)
    return out


def build_report(demo: pd.DataFrame, args: argparse.Namespace) -> str:
    coverage = pd.DataFrame(
        [
            {
                "demo_rows": len(demo),
                "seed_count": demo["seed_id"].nunique(),
                "objective_count": demo["objective"].nunique(),
                "mean_visible_options": len(demo) / max(1, demo[["seed_id", "objective"]].drop_duplicates().shape[0]),
                "mean_hidden_pool_size": demo.drop_duplicates(["seed_id", "objective"])["hidden_candidate_pool_size"].mean(),
            }
        ]
    )

    visible_counts = (
        demo.groupby(["objective", "display_label"], dropna=False)
        .size()
        .reset_index(name="visible_count")
        .sort_values(["objective", "visible_count"], ascending=[True, False])
    )

    recommended_counts = (
        demo[demo["is_recommended"]]
        .groupby(["objective", "variant"], dropna=False)
        .size()
        .reset_index(name="recommended_count")
        .sort_values(["objective", "recommended_count"], ascending=[True, False])
    )

    score_means = (
        demo[demo["is_recommended"]]
        .groupby("objective", dropna=False)
        .agg(
            mean_quality_score=("quality_score", "mean"),
            mean_risk_score=("risk_score", "mean"),
            mean_audience_score=("audience_score", "mean"),
            mean_objective_fit_score=("objective_fit_score", "mean"),
            mean_support_score=("support_score", "mean"),
            mean_unified_decision_score=("unified_decision_score", "mean"),
        )
        .reset_index()
    )

    examples = demo[demo["is_recommended"]].sort_values(["objective", "seed_id"]).groupby("objective").head(5)
    example_cols = [
        "objective",
        "seed_id",
        "category",
        "variant",
        "headline",
        "quality_score",
        "risk_score",
        "audience_score",
        "objective_fit_score",
        "support_score",
        "unified_decision_score",
        "decision_explanation",
    ]

    lines = [
        "# Headline Review Demo Cases",
        "",
        "This report describes the simplified product/demo dataset. It hides the full research candidate pool and exposes only a few meaningful options per article and objective.",
        "",
        "## Simplified Product Flow",
        "",
        "```text",
        "article summary",
        "-> hidden candidate pool",
        "-> unified evaluator: quality + risk/safety + audience fit + objective fit",
        "-> objective/persona-calibrated recommendation",
        "-> show visible options with selected/not-selected explanations",
        "```",
        "",
        "## Files",
        "",
        f"- Input matrix: `{display_path(args.matrix)}`",
        f"- Input selection: `{display_path(args.selection)}`",
        f"- Output demo cases: `{display_path(args.output)}`",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Visible Option Counts",
        "",
        markdown_table(visible_counts),
        "",
        "## Recommended Variant Counts",
        "",
        markdown_table(recommended_counts),
        "",
        "## Mean Recommended Unified Scores",
        "",
        markdown_table(score_means),
        "",
        "## Recommended Examples",
        "",
        markdown_table(examples[example_cols]),
        "",
        "## Interpretation",
        "",
        "The demo should use this file instead of the full 1,200-row research matrix. The UI can still mention that candidates come from a larger hidden pool, but it should show only the human baseline, GenAI baseline, low-risk alternative, and final recommendation.",
        "",
        "The standalone clickbait critic is folded into the Risk/Safety score. The product-facing decision is shown through a unified scorecard: Quality, Risk/Safety, Audience Fit, and Objective Fit. Each non-selected option includes a short reason explaining why it lost to the recommendation.",
        "",
    ]
    return "\n".join(lines)


def write_metadata(args: argparse.Namespace, demo: pd.DataFrame) -> None:
    metadata = {
        "matrix": display_path(args.matrix),
        "selection": display_path(args.selection),
        "output": display_path(args.output),
        "report": display_path(args.report),
        "rows": int(len(demo)),
        "seed_count": int(demo["seed_id"].nunique()) if not demo.empty else 0,
        "objective_count": int(demo["objective"].nunique()) if not demo.empty else 0,
        "recommended_rows": int(demo["is_recommended"].sum()) if not demo.empty else 0,
        "visible_labels": sorted(demo["display_label"].dropna().unique().tolist()) if not demo.empty else [],
        "unified_scorecard": ["quality_score", "risk_score", "audience_score", "objective_fit_score"],
        "decision_explanation_column": "decision_explanation",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit-seeds", type=int, help="Optional seed cap for quick local checks.")
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix)
    selection = pd.read_csv(args.selection)
    demo = build_demo_cases(matrix, selection, limit_seeds=args.limit_seeds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(args.output, index=False)
    args.report.write_text(build_report(demo, args), encoding="utf-8")
    write_metadata(args, demo)

    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(
        json.dumps(
            {
                "rows": int(len(demo)),
                "seed_count": int(demo["seed_id"].nunique()) if not demo.empty else 0,
                "objective_count": int(demo["objective"].nunique()) if not demo.empty else 0,
                "recommended_rows": int(demo["is_recommended"].sum()) if not demo.empty else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
