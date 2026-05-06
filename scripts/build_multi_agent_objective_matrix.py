#!/usr/bin/env python3
"""Build a multi-agent, multi-objective headline candidate matrix.

This reframes the project as an offline agentic-selection problem:

state   = article summary and category
action  = choose one candidate headline from multiple generator agents
reward  = vector of critic scores
policy  = objective-specific selector that scalarizes the reward vector
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENTIC_LOCAL = PROJECT_ROOT / "data" / "processed" / "headline_agentic_v3_specificity_vs_baselines_eval.csv"
DEFAULT_SFT_LOCAL = PROJECT_ROOT / "data" / "processed" / "headline_sft_generators_eval.csv"
DEFAULT_AGENTIC_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_candidates_v3_specificity_100.csv"
DEFAULT_AGENTIC_JUDGE = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_v3_specificity_scores.csv"
DEFAULT_SFT_JUDGE = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_sft_scores.csv"
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_candidate_matrix.csv"
DEFAULT_SELECTION = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_objective_selection.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_objective_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_objective_metadata.json"

LOCAL_SCORE_FIELDS = [
    "pred_faithfulness",
    "pred_clarity",
    "pred_specificity",
    "pred_attractiveness",
    "pred_non_clickbait",
    "pred_overall",
]
JUDGE_SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]
TOP_LEVEL_VARIANTS = ["original", "zero_shot", "round1_final", "round2_final", "agentic_selected", "generic_sft", "specificity_sft"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
}

OBJECTIVES = {
    "trust_safety": {
        "description": "Prefer faithful, clear, non-clickbait titles for trust-sensitive surfaces.",
        "weights": {
            "pred_faithfulness": 0.40,
            "pred_clarity": 0.20,
            "pred_specificity": 0.10,
            "pred_attractiveness": 0.00,
            "pred_non_clickbait": 0.30,
            "pairwise_reward": 0.10,
            "clickbait_penalty": -1.00,
            "length_style_score": 0.20,
        },
    },
    "growth": {
        "description": "Prefer attractive, clear, specific titles while keeping clickbait risk bounded.",
        "weights": {
            "pred_faithfulness": 0.15,
            "pred_clarity": 0.20,
            "pred_specificity": 0.20,
            "pred_attractiveness": 0.35,
            "pred_non_clickbait": 0.10,
            "pairwise_reward": 0.20,
            "clickbait_penalty": -0.35,
            "length_style_score": 0.15,
        },
    },
    "editorial": {
        "description": "Prefer balanced titles that look like compact human-edited news headlines.",
        "weights": {
            "pred_faithfulness": 0.25,
            "pred_clarity": 0.20,
            "pred_specificity": 0.20,
            "pred_attractiveness": 0.20,
            "pred_non_clickbait": 0.15,
            "pairwise_reward": 0.15,
            "clickbait_penalty": -0.50,
            "length_style_score": 0.35,
        },
    },
    "specificity": {
        "description": "Prefer concrete, supported details without sacrificing faithfulness.",
        "weights": {
            "pred_faithfulness": 0.30,
            "pred_clarity": 0.10,
            "pred_specificity": 0.40,
            "pred_attractiveness": 0.05,
            "pred_non_clickbait": 0.15,
            "pairwise_reward": 0.10,
            "clickbait_penalty": -0.45,
            "length_style_score": 0.10,
            "summary_support_rate": 0.20,
        },
    },
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", clean_text(text).lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in word_tokens(text) if token not in STOPWORDS and len(token) > 2}


def summary_support_rate(headline: str, summary: str) -> float:
    headline_tokens = content_tokens(headline)
    summary_tokens = content_tokens(summary)
    if not headline_tokens:
        return 0.0
    return len(headline_tokens & summary_tokens) / len(headline_tokens)


def contains_specific_signal(title: str) -> bool:
    if re.search(r"\d", title):
        return True
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]{2,}\b", clean_text(title))
    return len(tokens) >= 2


def length_style_score(word_count: float) -> float:
    if pd.isna(word_count):
        return 0.0
    if 6 <= word_count <= 14:
        return 1.0
    if word_count < 6:
        return max(0.0, 1.0 - (6 - word_count) / 6.0)
    return max(0.0, 1.0 - (word_count - 14) / 12.0)


def normalize_top_level_local(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["candidate_source"] = out["variant"].map(
        {
            "original": "human_editor",
            "zero_shot": "api_zero_shot_generator",
            "round1_final": "critic_guided_rewriter_round1",
            "round2_final": "critic_guided_rewriter_round2",
            "agentic_selected": "agentic_selector_v3",
        }
    ).fillna(out["variant"])
    out["candidate_id"] = out["variant"]
    out["source_policy"] = out["variant"]
    out["local_final_score"] = out["final_score"]
    keep = [
        "seed_id",
        "nid",
        "news_id",
        "category",
        "subvert",
        "summary",
        "variant",
        "candidate_id",
        "candidate_source",
        "source_policy",
        "headline",
        "headline_word_count",
        "clickbait_penalty",
        "predicted_clickbait",
        *LOCAL_SCORE_FIELDS,
        "quality_reward",
        "pairwise_reward",
        "local_final_score",
    ]
    return out[keep]


def normalize_sft_local(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["variant"].isin(["generic_sft", "specificity_sft"])].copy()
    df["candidate_source"] = df["variant"].map(
        {
            "generic_sft": "generic_sft_generator",
            "specificity_sft": "specificity_sft_generator",
        }
    )
    df["candidate_id"] = df["variant"]
    df["source_policy"] = df["variant"]
    df["local_final_score"] = df["final_score"]
    keep = [
        "seed_id",
        "nid",
        "news_id",
        "category",
        "subvert",
        "summary",
        "variant",
        "candidate_id",
        "candidate_source",
        "source_policy",
        "headline",
        "headline_word_count",
        "clickbait_penalty",
        "predicted_clickbait",
        *LOCAL_SCORE_FIELDS,
        "quality_reward",
        "pairwise_reward",
        "local_final_score",
    ]
    return df[keep]


def normalize_agentic_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "seed_id": df["seed_id"],
            "nid": df["nid"],
            "news_id": df["news_id"],
            "category": df["category"],
            "subvert": df["subvert"],
            "summary": df["summary"],
            "variant": "agentic_candidate",
            "candidate_id": df["candidate_id"],
            "candidate_source": "agentic_generator_v3_candidate",
            "source_policy": "agentic_candidate_rank_" + df["candidate_rank"].astype(str),
            "headline": df["candidate_title"],
            "headline_word_count": df["candidate_title"].fillna("").astype(str).str.split().map(len),
            "clickbait_penalty": df["candidate_clickbait_penalty"],
            "predicted_clickbait": df["candidate_predicted_clickbait"],
            "pred_faithfulness": df["candidate_pred_faithfulness"],
            "pred_clarity": df["candidate_pred_clarity"],
            "pred_specificity": df["candidate_pred_specificity"],
            "pred_attractiveness": df["candidate_pred_attractiveness"],
            "pred_non_clickbait": df["candidate_pred_non_clickbait"],
            "pred_overall": df["candidate_pred_overall"],
            "quality_reward": df["candidate_quality_reward"],
            "pairwise_reward": df["candidate_pairwise_reward"],
            "local_final_score": df["agentic_final_score"],
        }
    )
    return out


def normalize_judge(path: Path, variant_map: dict[str, str], source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["overall"].notna()].copy()
    df["variant"] = df["variant"].map(lambda value: variant_map.get(str(value), str(value)))
    columns = ["seed_id", "variant", "headline", "rationale", "best_variant", "worst_variant", *JUDGE_SCORE_FIELDS]
    out = df[columns].copy()
    out = out.rename(
        columns={
            "headline": f"{source_name}_judge_headline",
            "rationale": f"{source_name}_judge_rationale",
            "best_variant": f"{source_name}_best_variant",
            "worst_variant": f"{source_name}_worst_variant",
            **{field: f"{source_name}_{field}" for field in JUDGE_SCORE_FIELDS},
        }
    )
    return out


def enrich_features(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    out["headline"] = out["headline"].map(clean_text)
    out["summary"] = out["summary"].map(clean_text)
    out["headline_word_count"] = out["headline_word_count"].fillna(out["headline"].str.split().map(len))
    out["has_specific_signal"] = out["headline"].map(contains_specific_signal)
    out["summary_support_rate"] = [
        summary_support_rate(headline, summary)
        for headline, summary in zip(out["headline"], out["summary"])
    ]
    out["length_style_score"] = out["headline_word_count"].map(length_style_score)
    out["candidate_key"] = (
        out["seed_id"].astype(str)
        + "::"
        + out["variant"].astype(str)
        + "::"
        + out["candidate_id"].astype(str)
        + "::"
        + out["headline"].astype(str).str.lower().str.slice(0, 80)
    )
    return out


def add_judge_scores(matrix: pd.DataFrame, agentic_judge: Path, sft_judge: Path) -> pd.DataFrame:
    out = matrix.copy()
    agentic = normalize_judge(
        agentic_judge,
        {"optimized": "round2_final"},
        "agentic_llm",
    )
    sft = normalize_judge(
        sft_judge,
        {},
        "sft_llm",
    )
    out = out.merge(agentic.drop(columns=["agentic_llm_judge_headline"]), on=["seed_id", "variant"], how="left")
    out = out.merge(sft.drop(columns=["sft_llm_judge_headline"]), on=["seed_id", "variant"], how="left")
    out["llm_overall_available"] = out["agentic_llm_overall"].notna() | out["sft_llm_overall"].notna()
    out["llm_overall"] = out["agentic_llm_overall"].combine_first(out["sft_llm_overall"])
    return out


def objective_score(row: pd.Series, weights: dict[str, float]) -> float:
    score = 0.0
    for field, weight in weights.items():
        value = row.get(field, 0.0)
        if pd.isna(value):
            value = 0.0
        score += weight * float(value)
    return float(score)


def add_objective_scores(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for objective, config in OBJECTIVES.items():
        out[f"objective_{objective}_score"] = out.apply(
            lambda row, weights=config["weights"]: objective_score(row, weights),
            axis=1,
        )
    return out


def select_by_objective(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective in OBJECTIVES:
        score_col = f"objective_{objective}_score"
        for seed_id, group in matrix.groupby("seed_id", sort=True):
            group = group[group["headline"].ne("")].copy()
            if group.empty:
                continue
            selected = group.sort_values(score_col, ascending=False).iloc[0]
            runner_up = group.sort_values(score_col, ascending=False).iloc[1] if len(group) > 1 else selected
            rows.append(
                {
                    "objective": objective,
                    "objective_description": OBJECTIVES[objective]["description"],
                    "seed_id": int(seed_id),
                    "category": selected["category"],
                    "selected_variant": selected["variant"],
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_source": selected["candidate_source"],
                    "selected_headline": selected["headline"],
                    "selected_score": selected[score_col],
                    "runner_up_variant": runner_up["variant"],
                    "runner_up_candidate_id": runner_up["candidate_id"],
                    "runner_up_headline": runner_up["headline"],
                    "runner_up_score": runner_up[score_col],
                    "score_margin": selected[score_col] - runner_up[score_col],
                    "clickbait_penalty": selected["clickbait_penalty"],
                    "pred_faithfulness": selected["pred_faithfulness"],
                    "pred_clarity": selected["pred_clarity"],
                    "pred_specificity": selected["pred_specificity"],
                    "pred_attractiveness": selected["pred_attractiveness"],
                    "pred_non_clickbait": selected["pred_non_clickbait"],
                    "pairwise_reward": selected["pairwise_reward"],
                    "length_style_score": selected["length_style_score"],
                    "summary_support_rate": selected["summary_support_rate"],
                    "llm_overall": selected["llm_overall"],
                }
            )
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


def build_report(matrix: pd.DataFrame, selection: pd.DataFrame, args: argparse.Namespace) -> str:
    source_summary = (
        matrix.groupby(["variant", "candidate_source"], dropna=False)
        .agg(
            rows=("headline", "size"),
            mean_clickbait_penalty=("clickbait_penalty", "mean"),
            mean_faithfulness=("pred_faithfulness", "mean"),
            mean_specificity=("pred_specificity", "mean"),
            mean_attractiveness=("pred_attractiveness", "mean"),
            mean_local_final=("local_final_score", "mean"),
            mean_llm_overall=("llm_overall", "mean"),
        )
        .reset_index()
        .sort_values(["variant", "candidate_source"])
    )

    objective_counts = (
        selection.groupby(["objective", "selected_variant"], dropna=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["objective", "selected_count"], ascending=[True, False])
    )

    objective_means = (
        selection.groupby("objective", dropna=False)
        .agg(
            mean_clickbait_penalty=("clickbait_penalty", "mean"),
            mean_faithfulness=("pred_faithfulness", "mean"),
            mean_specificity=("pred_specificity", "mean"),
            mean_attractiveness=("pred_attractiveness", "mean"),
            mean_summary_support=("summary_support_rate", "mean"),
            mean_llm_overall=("llm_overall", "mean"),
        )
        .reset_index()
    )

    example_cols = [
        "objective",
        "seed_id",
        "category",
        "selected_variant",
        "selected_source",
        "selected_headline",
        "runner_up_variant",
        "score_margin",
    ]
    examples = selection.sort_values(["objective", "score_margin"], ascending=[True, False]).groupby("objective").head(5)

    lines = [
        "# Multi-Agent Objective Headline Matrix",
        "",
        "This report reframes the project as an offline agentic RL-style selection system.",
        "",
        "## Agentic Framing",
        "",
        "- State: article summary, category, and context.",
        "- Action: choose a candidate headline from multiple generator agents.",
        "- Reward vector: local critic scores, clickbait penalty, pairwise reward, style heuristics, and retrospective LLM judge labels when available.",
        "- Policy: objective-specific selector that scalarizes the reward vector differently for trust, growth, editorial, or specificity goals.",
        "",
        "## Files",
        "",
        f"- Candidate matrix: `{args.matrix_output}`",
        f"- Objective selections: `{args.selection_output}`",
        "",
        "## Candidate Sources",
        "",
        markdown_table(source_summary),
        "",
        "## Objective Selection Counts",
        "",
        markdown_table(objective_counts),
        "",
        "## Objective Mean Selected Scores",
        "",
        markdown_table(objective_means),
        "",
        "## Objective Presets",
        "",
    ]
    for objective, config in OBJECTIVES.items():
        lines.extend(
            [
                f"### {objective}",
                "",
                config["description"],
                "",
                "Weights:",
                "",
                markdown_table(pd.DataFrame([{"signal": k, "weight": v} for k, v in config["weights"].items()])),
                "",
            ]
        )

    lines.extend(
        [
            "## Selection Examples",
            "",
            markdown_table(examples[example_cols]),
            "",
            "## Next Step",
            "",
            "Use this matrix as the control plane for multi-agent work: add audience persona agents as new reward columns, then compare how their votes change objective-specific selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentic-local", type=Path, default=DEFAULT_AGENTIC_LOCAL)
    parser.add_argument("--sft-local", type=Path, default=DEFAULT_SFT_LOCAL)
    parser.add_argument("--agentic-candidates", type=Path, default=DEFAULT_AGENTIC_CANDIDATES)
    parser.add_argument("--agentic-judge", type=Path, default=DEFAULT_AGENTIC_JUDGE)
    parser.add_argument("--sft-judge", type=Path, default=DEFAULT_SFT_JUDGE)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    frames = [
        normalize_top_level_local(args.agentic_local),
        normalize_sft_local(args.sft_local),
        normalize_agentic_candidates(args.agentic_candidates),
    ]
    matrix = pd.concat(frames, ignore_index=True)
    matrix = matrix.drop_duplicates(["seed_id", "variant", "candidate_id", "headline"]).reset_index(drop=True)
    matrix = enrich_features(matrix)
    matrix = add_judge_scores(matrix, args.agentic_judge, args.sft_judge)
    matrix = add_objective_scores(matrix)
    selection = select_by_objective(matrix)

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.matrix_output, index=False)
    selection.to_csv(args.selection_output, index=False)
    args.report.write_text(build_report(matrix, selection, args), encoding="utf-8")

    metadata = {
        "agentic_local": str(args.agentic_local),
        "sft_local": str(args.sft_local),
        "agentic_candidates": str(args.agentic_candidates),
        "agentic_judge": str(args.agentic_judge),
        "sft_judge": str(args.sft_judge),
        "matrix_output": str(args.matrix_output),
        "selection_output": str(args.selection_output),
        "report": str(args.report),
        "rows": int(len(matrix)),
        "seed_count": int(matrix["seed_id"].nunique()),
        "selection_rows": int(len(selection)),
        "objectives": OBJECTIVES,
        "candidate_sources": {
            str(k): int(v) for k, v in matrix["candidate_source"].value_counts().to_dict().items()
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.matrix_output)
    print("Wrote", args.selection_output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps({k: metadata[k] for k in ["rows", "seed_count", "selection_rows", "candidate_sources"]}, indent=2))


if __name__ == "__main__":
    main()
