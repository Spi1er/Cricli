#!/usr/bin/env python3
"""Generate and evaluate generic vs specificity-aware SFT headline models."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_agentic_vs_baselines import (  # noqa: E402
    SCORE_FIELDS,
    batched_clickbait_scores,
    batched_pairwise_scores,
    batched_quality_scores,
    validate_device,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_eval_seed_100.csv"
DEFAULT_GENERIC_MODEL = PROJECT_ROOT / "models" / "headline_generator_flan_t5_small_generic_sft"
DEFAULT_SPECIFICITY_MODEL = PROJECT_ROOT / "models" / "headline_generator_flan_t5_small_specificity_sft"
DEFAULT_CLICKBAIT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models" / "headline_quality_reward_distilbert_v2"
DEFAULT_PAIRWISE_MODEL = PROJECT_ROOT / "models" / "headline_pairwise_reward_distilbert_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_sft_generators_eval.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_sft_generators_eval_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_sft_generators_eval_metadata.json"

VARIANT_ORDER = ["original", "generic_sft", "specificity_sft"]
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


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", clean_text(text).lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in word_tokens(text) if token not in STOPWORDS and len(token) > 2}


def token_f1(prediction: str, reference: str) -> float:
    pred = content_tokens(prediction)
    ref = content_tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


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


def generic_prompt(row: pd.Series) -> str:
    return "\n".join(
        [
            "Generate a news headline.",
            f"Category: {clean_text(row['category'])}",
            f"Article summary: {clean_text(row['summary'])}",
        ]
    )


def specificity_prompt(row: pd.Series) -> str:
    return "\n".join(
        [
            "Generate a faithful, specific, non-clickbait news headline.",
            "Use concrete names, numbers, places, or events only when supported by the article summary.",
            "Avoid curiosity-gap wording, exaggeration, and unsupported details.",
            f"Category: {clean_text(row['category'])}",
            f"Article summary: {clean_text(row['summary'])}",
        ]
    )


def generate_titles(
    df: pd.DataFrame,
    model_path: Path,
    prompt_fn: Callable[[pd.Series], str],
    device: torch.device,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_beams: int,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    prompts = df.apply(prompt_fn, axis=1).tolist()
    titles: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_length=max_target_length,
                num_beams=num_beams,
            )
        titles.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return [clean_text(title) for title in titles]


def reward_text(row: pd.Series) -> str:
    return (
        f"Category: {clean_text(row['category'])}\n"
        f"Summary: {clean_text(row['summary'])}\n"
        f"Headline: {clean_text(row['headline'])}"
    )


def build_long_dataframe(seed_df: pd.DataFrame, generic_titles: list[str], specificity_titles: list[str]) -> pd.DataFrame:
    rows = []
    for i, row in seed_df.reset_index(drop=True).iterrows():
        base = {
            "seed_id": int(row["seed_id"]),
            "nid": clean_text(row.get("nid", "")),
            "news_id": clean_text(row.get("news_id", "")),
            "category": clean_text(row["category"]),
            "subvert": clean_text(row.get("subvert", "")),
            "summary": clean_text(row["summary"]),
            "reference_title": clean_text(row["title"]),
        }
        variants = {
            "original": clean_text(row["title"]),
            "generic_sft": generic_titles[i],
            "specificity_sft": specificity_titles[i],
        }
        for variant, headline in variants.items():
            out = dict(base)
            out["variant"] = variant
            out["headline"] = clean_text(headline)
            out["headline_word_count"] = len(word_tokens(out["headline"]))
            out["has_specific_signal"] = contains_specific_signal(out["headline"])
            out["reference_token_f1"] = token_f1(out["headline"], out["reference_title"])
            out["summary_support_rate"] = summary_support_rate(out["headline"], out["summary"])
            rows.append(out)
    return pd.DataFrame(rows)


def score_variants(df: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    out = df.copy()
    titles = out["headline"].fillna("").astype(str).tolist()
    reward_texts = out.apply(reward_text, axis=1).tolist()

    out["clickbait_penalty"] = batched_clickbait_scores(
        titles,
        args.clickbait_model,
        args.batch_size,
        args.clickbait_max_length,
        device,
    )
    quality_dims, quality_reward = batched_quality_scores(
        reward_texts,
        args.quality_model,
        args.batch_size,
        args.reward_max_length,
        device,
    )
    pairwise_reward = batched_pairwise_scores(
        reward_texts,
        args.pairwise_model,
        args.batch_size,
        args.reward_max_length,
        device,
    )
    for i, field in enumerate(SCORE_FIELDS):
        out[f"pred_{field}"] = quality_dims[:, i]
    out["quality_reward"] = quality_reward
    out["pairwise_reward"] = pairwise_reward
    out["final_score"] = (
        args.quality_weight * out["quality_reward"]
        + args.pairwise_weight * out["pairwise_reward"]
        - args.clickbait_weight * out["clickbait_penalty"]
    )
    out["predicted_clickbait"] = (out["clickbait_penalty"] >= args.clickbait_threshold).astype(int)
    return out


def paired_delta(scored: pd.DataFrame, left: str, right: str, field: str) -> dict[str, float]:
    wide = scored.pivot(index="seed_id", columns="variant", values=field)
    delta = wide[left] - wide[right]
    return {
        "comparison": f"{left} - {right}",
        f"mean_delta_{field}": float(delta.mean()),
        f"median_delta_{field}": float(delta.median()),
        "left_win_rate": float((delta > 0).mean()),
    }


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(scored: pd.DataFrame, args: argparse.Namespace, metadata: dict) -> str:
    summary = (
        scored.groupby("variant")
        .agg(
            rows=("headline", "size"),
            mean_clickbait_penalty=("clickbait_penalty", "mean"),
            clickbait_rate=("predicted_clickbait", "mean"),
            mean_quality_reward=("quality_reward", "mean"),
            mean_pairwise_reward=("pairwise_reward", "mean"),
            mean_final_score=("final_score", "mean"),
            mean_pred_overall=("pred_overall", "mean"),
            mean_reference_token_f1=("reference_token_f1", "mean"),
            mean_summary_support_rate=("summary_support_rate", "mean"),
            mean_headline_words=("headline_word_count", "mean"),
            specificity_signal_rate=("has_specific_signal", "mean"),
        )
        .reindex(VARIANT_ORDER)
        .reset_index()
    )

    best = (
        scored.loc[scored.groupby("seed_id")["final_score"].idxmax(), "variant"]
        .value_counts()
        .reindex(VARIANT_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    best.columns = ["variant", "best_count"]
    best["best_rate"] = best["best_count"] / scored["seed_id"].nunique()

    deltas = pd.DataFrame(
        [
            paired_delta(scored, "specificity_sft", "generic_sft", "final_score"),
            paired_delta(scored, "specificity_sft", "original", "final_score"),
            paired_delta(scored, "generic_sft", "original", "final_score"),
        ]
    )

    examples = scored[scored["variant"].isin(["generic_sft", "specificity_sft"])].copy()
    examples = examples.sort_values("final_score", ascending=False).head(12)
    examples = examples[
        [
            "seed_id",
            "variant",
            "category",
            "headline",
            "clickbait_penalty",
            "quality_reward",
            "pairwise_reward",
            "final_score",
        ]
    ]

    lines = [
        "# SFT Headline Generator Evaluation",
        "",
        "This report compares the generic SFT model and the specificity-aware SFT model on the same fixed 100-example headline generation seed set.",
        "",
        "## Configuration",
        "",
        f"- Input: `{args.input}`",
        f"- Generic model: `{args.generic_model}`",
        f"- Specificity model: `{args.specificity_model}`",
        f"- Device: `{metadata['device']}`",
        f"- Output: `{args.output}`",
        f"- Clickbait weight: {args.clickbait_weight}",
        f"- Quality weight: {args.quality_weight}",
        f"- Pairwise weight: {args.pairwise_weight}",
        "",
        "## Variant Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Paired Final-Score Deltas",
        "",
        dataframe_to_markdown(deltas),
        "",
        "## Best Variant By Local Final Score",
        "",
        dataframe_to_markdown(best),
        "",
        "## Top SFT Examples",
        "",
        dataframe_to_markdown(examples),
        "",
        "## Interpretation",
        "",
        "Use this as a local critic evaluation, not the final human-quality verdict. The most important next check is an LLM judge comparison between `generic_sft` and `specificity_sft`, because local critics can reward extractive or overly literal titles.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--generic-model", type=Path, default=DEFAULT_GENERIC_MODEL)
    parser.add_argument("--specificity-model", type=Path, default=DEFAULT_SPECIFICITY_MODEL)
    parser.add_argument("--clickbait-model", type=Path, default=DEFAULT_CLICKBAIT_MODEL)
    parser.add_argument("--quality-model", type=Path, default=DEFAULT_QUALITY_MODEL)
    parser.add_argument("--pairwise-model", type=Path, default=DEFAULT_PAIRWISE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--clickbait-max-length", type=int, default=96)
    parser.add_argument("--reward-max-length", type=int, default=256)
    parser.add_argument("--clickbait-threshold", type=float, default=0.5)
    parser.add_argument("--clickbait-weight", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    args = parser.parse_args()

    device = validate_device(args.device)
    seed_df = pd.read_csv(args.input)

    print("Generating generic SFT titles...")
    generic_titles = generate_titles(
        seed_df,
        args.generic_model,
        generic_prompt,
        device,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
        args.num_beams,
    )
    print("Generating specificity-aware SFT titles...")
    specificity_titles = generate_titles(
        seed_df,
        args.specificity_model,
        specificity_prompt,
        device,
        args.batch_size,
        args.max_source_length,
        args.max_target_length,
        args.num_beams,
    )

    variants = build_long_dataframe(seed_df, generic_titles, specificity_titles)
    print(f"Scoring {len(variants):,} variant rows with local critics...")
    scored = score_variants(variants, args, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    metadata = {
        "input": str(args.input),
        "generic_model": str(args.generic_model),
        "specificity_model": str(args.specificity_model),
        "clickbait_model": str(args.clickbait_model),
        "quality_model": str(args.quality_model),
        "pairwise_model": str(args.pairwise_model),
        "output": str(args.output),
        "report": str(args.report),
        "device": str(device),
        "rows": int(len(scored)),
        "seed_count": int(seed_df["seed_id"].nunique()),
        "variants": VARIANT_ORDER,
        "weights": {
            "clickbait": args.clickbait_weight,
            "quality": args.quality_weight,
            "pairwise": args.pairwise_weight,
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    args.report.write_text(build_report(scored, args, metadata), encoding="utf-8")

    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
