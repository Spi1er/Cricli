#!/usr/bin/env python3
"""Use an API judge to compare agentic selected headlines against baselines.

This is the final quality evaluation step for the agentic loop. It judges four
variants together:
  original, zero_shot, optimized (round-2 final), agentic_selected

The script is resumable and exports both pointwise scores and pairwise
preferences for later reward-model training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_agentic_vs_baselines_eval.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_scores.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_profile.md"
DEFAULT_PAIRWISE = PROJECT_ROOT / "data" / "processed" / "headline_quality_agentic_pairwise_preferences.jsonl"
DEFAULT_REWARD = PROJECT_ROOT / "data" / "processed" / "headline_quality_agentic_reward_model_examples.jsonl"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_metadata.json"

VARIANTS = ["original", "zero_shot", "optimized", "agentic_selected"]
INPUT_VARIANT_MAP = {
    "original": "original",
    "zero_shot": "zero_shot",
    "optimized": "round2_final",
    "agentic_selected": "agentic_selected",
}
SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


INSTRUCTIONS = """You are a strict but fair news headline quality judge.

Evaluate candidate headlines against the provided news summary.

Return only valid JSON with this exact schema:
{
  "scores": {
    "original": {
      "faithfulness": 1,
      "clarity": 1,
      "specificity": 1,
      "attractiveness": 1,
      "non_clickbait": 1,
      "overall": 1,
      "rationale": "short reason"
    },
    "zero_shot": {...},
    "optimized": {...},
    "agentic_selected": {...}
  },
  "ranking": ["best_variant", "second_variant", "third_variant", "fourth_variant"],
  "best_variant": "variant_name",
  "worst_variant": "variant_name"
}

Scoring rubric:
- Scores are integers from 1 to 5.
- faithfulness: preserves the main claim and does not add unsupported facts.
- clarity: easy to understand as a news headline.
- specificity: includes concrete people, places, events, or claims when supported.
- attractiveness: likely to interest a reader without being manipulative.
- non_clickbait: avoids vague teasers, exaggerated adjectives, listicle bait, and curiosity gaps.
- overall: balanced judgment across all dimensions.

Important:
- Do not reward a headline for inventing details absent from the summary.
- If two variants are identical, give them identical scores unless surrounding variants reveal a meaningful issue.
- Variant names must be exactly: original, zero_shot, optimized, agentic_selected."""


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def extract_json_object(text: str) -> dict:
    text = clean_text(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_output_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def call_openai_judge(
    *,
    api_key: str,
    base_url: str,
    model: str,
    row: pd.Series,
    timeout: int,
    retries: int,
    max_output_tokens: int,
) -> dict:
    url = base_url.rstrip("/") + "/responses"
    user_input = (
        f"Category: {clean_text(row['category']) or 'unknown'}\n\n"
        f"Summary:\n{clean_text(row['summary'])}\n\n"
        f"Candidate headlines:\n"
        f"- original: {clean_text(row['original'])}\n"
        f"- zero_shot: {clean_text(row['zero_shot'])}\n"
        f"- optimized: {clean_text(row['optimized'])}\n"
        f"- agentic_selected: {clean_text(row['agentic_selected'])}\n"
    )
    payload = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(30, 2**attempt))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} {response.reason}: {response.text[:2000]}")
            return extract_json_object(extract_output_text(response.json()))
        except Exception as exc:  # noqa: BLE001 - keep partial judge runs resumable.
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(f"OpenAI judge failed after {retries + 1} attempt(s): {last_error}") from exc
    raise RuntimeError(f"OpenAI judge failed: {last_error}")


def normalize_judgment(judgment: dict) -> dict:
    scores = judgment.get("scores", {})
    out = {"scores": {}, "ranking": [], "best_variant": "", "worst_variant": ""}

    for variant in VARIANTS:
        variant_scores = scores.get(variant, {})
        out["scores"][variant] = {}
        for field in SCORE_FIELDS:
            value = variant_scores.get(field)
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                value = max(1, min(5, value))
            out["scores"][variant][field] = value
        out["scores"][variant]["rationale"] = clean_text(variant_scores.get("rationale", ""))

    ranking = [item for item in judgment.get("ranking", []) if item in VARIANTS]
    if len(ranking) != len(VARIANTS):
        ranking = sorted(
            VARIANTS,
            key=lambda v: (
                out["scores"][v].get("overall") or 0,
                out["scores"][v].get("faithfulness") or 0,
                out["scores"][v].get("non_clickbait") or 0,
            ),
            reverse=True,
        )
    out["ranking"] = ranking
    out["best_variant"] = judgment.get("best_variant") if judgment.get("best_variant") in VARIANTS else ranking[0]
    out["worst_variant"] = judgment.get("worst_variant") if judgment.get("worst_variant") in VARIANTS else ranking[-1]
    return out


def build_seed_table(input_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed_id, group in input_df.groupby("seed_id", sort=True):
        group = group.copy()
        base = group.iloc[0]
        record = {
            "seed_id": int(seed_id),
            "nid": base["nid"],
            "news_id": base["news_id"],
            "category": base["category"],
            "summary": base["summary"],
        }
        for judge_variant, input_variant in INPUT_VARIANT_MAP.items():
            match = group[group["variant"].eq(input_variant)]
            if match.empty:
                record[judge_variant] = ""
                record[f"{judge_variant}_clickbait_penalty"] = None
            else:
                row = match.iloc[0]
                record[judge_variant] = clean_text(row["headline"])
                record[f"{judge_variant}_clickbait_penalty"] = float(row["clickbait_penalty"])
        rows.append(record)
    return pd.DataFrame(rows)


def prepare_output(seed_table: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for seed_row in seed_table.itertuples(index=False):
        for variant in VARIANTS:
            rows.append(
                {
                    "seed_id": seed_row.seed_id,
                    "nid": seed_row.nid,
                    "news_id": seed_row.news_id,
                    "category": seed_row.category,
                    "summary": seed_row.summary,
                    "variant": variant,
                    "headline": getattr(seed_row, variant),
                    "clickbait_penalty": getattr(seed_row, f"{variant}_clickbait_penalty"),
                    "faithfulness": None,
                    "clarity": None,
                    "specificity": None,
                    "attractiveness": None,
                    "non_clickbait": None,
                    "overall": None,
                    "rationale": "",
                    "ranking": "",
                    "best_variant": "",
                    "worst_variant": "",
                    "judge_model": "",
                    "judge_error": "",
                }
            )
    out = pd.DataFrame(rows)
    if not output.exists():
        return out

    existing = pd.read_csv(output)
    key_cols = ["seed_id", "variant"]
    out = out.merge(
        existing[key_cols + [col for col in existing.columns if col not in key_cols]],
        on=key_cols,
        how="left",
        suffixes=("", "_existing"),
    )
    for col in [
        "faithfulness",
        "clarity",
        "specificity",
        "attractiveness",
        "non_clickbait",
        "overall",
        "rationale",
        "ranking",
        "best_variant",
        "worst_variant",
        "judge_model",
        "judge_error",
    ]:
        existing_col = f"{col}_existing"
        if existing_col in out.columns:
            out[col] = out[existing_col].combine_first(out[col])
            out = out.drop(columns=[existing_col])
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_training_exports(scores_df: pd.DataFrame, pairwise_path: Path, reward_path: Path) -> tuple[int, int]:
    pairwise_records = []
    reward_records = []

    for seed_id, group in scores_df.groupby("seed_id"):
        group = group.copy()
        summary = clean_text(group["summary"].iloc[0])
        category = clean_text(group["category"].iloc[0])

        for row in group.itertuples(index=False):
            if pd.isna(row.overall):
                continue
            reward_records.append(
                {
                    "seed_id": int(seed_id),
                    "variant": row.variant,
                    "summary": summary,
                    "headline": row.headline,
                    "category": category,
                    "reward_dimensions": {
                        field: int(getattr(row, field))
                        for field in SCORE_FIELDS
                        if not pd.isna(getattr(row, field))
                    },
                    "clickbait_penalty": float(row.clickbait_penalty),
                    "judge_model": row.judge_model,
                    "rationale": clean_text(row.rationale),
                    "comparison_set": "agentic_4way",
                }
            )

        scored = group.dropna(subset=["overall"])
        for left, right in itertools.combinations(scored.itertuples(index=False), 2):
            left_score = (left.overall, left.faithfulness, left.non_clickbait)
            right_score = (right.overall, right.faithfulness, right.non_clickbait)
            if left_score == right_score:
                continue
            chosen, rejected = (left, right) if left_score > right_score else (right, left)
            pairwise_records.append(
                {
                    "seed_id": int(seed_id),
                    "summary": summary,
                    "category": category,
                    "chosen_variant": chosen.variant,
                    "rejected_variant": rejected.variant,
                    "chosen_title": chosen.headline,
                    "rejected_title": rejected.headline,
                    "chosen_scores": {field: int(getattr(chosen, field)) for field in SCORE_FIELDS},
                    "rejected_scores": {field: int(getattr(rejected, field)) for field in SCORE_FIELDS},
                    "preference_source": "llm_judge_agentic_4way",
                    "judge_model": chosen.judge_model,
                }
            )

    write_jsonl(pairwise_records, pairwise_path)
    write_jsonl(reward_records, reward_path)
    return len(pairwise_records), len(reward_records)


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


def build_report(scores_df: pd.DataFrame, pairwise_count: int, reward_count: int, output: Path) -> str:
    metric_table = (
        scores_df.groupby("variant", observed=True)[SCORE_FIELDS + ["clickbait_penalty"]]
        .mean()
        .reindex(VARIANTS)
        .reset_index()
    )
    best_counts = scores_df.drop_duplicates("seed_id")["best_variant"].value_counts().reindex(VARIANTS, fill_value=0)
    worst_counts = scores_df.drop_duplicates("seed_id")["worst_variant"].value_counts().reindex(VARIANTS, fill_value=0)
    winner_df = pd.DataFrame(
        {
            "variant": VARIANTS,
            "best_count": [int(best_counts[v]) for v in VARIANTS],
            "worst_count": [int(worst_counts[v]) for v in VARIANTS],
        }
    )

    pivot = scores_df.pivot(index="seed_id", columns="variant", values="overall")
    deltas = []
    for baseline in ["original", "zero_shot", "optimized"]:
        if {"agentic_selected", baseline}.issubset(pivot.columns):
            delta = pivot["agentic_selected"] - pivot[baseline]
            deltas.append(
                {
                    "comparison": f"agentic_selected - {baseline}",
                    "mean_overall_delta": float(delta.mean()),
                    "median_overall_delta": float(delta.median()),
                    "agentic_win_rate": float((delta > 0).mean()),
                }
            )
    delta_df = pd.DataFrame(deltas)

    lines = [
        "# LLM Judge Agentic Comparison",
        "",
        f"- Scores: `{output}`",
        f"- Scored rows: {scores_df[SCORE_FIELDS[0]].notna().sum():,}",
        f"- Pairwise preference examples: {pairwise_count:,}",
        f"- Pointwise reward examples: {reward_count:,}",
        "",
        "## Mean Scores By Variant",
        "",
        markdown_table(metric_table),
        "",
        "## Judge Winners",
        "",
        markdown_table(winner_df),
        "",
        "## Agentic Overall Deltas",
        "",
        markdown_table(delta_df),
        "",
        "## Interpretation",
        "",
        "This is the final LLM-judge check for whether the local critic selected headlines are actually preferred when compared directly against the original, zero-shot, and round-2 optimized baselines.",
        "",
        "## Training Use",
        "",
        "- `headline_quality_agentic_pairwise_preferences.jsonl` can extend the pairwise reward dataset with agentic-vs-baseline preferences.",
        "- `headline_quality_agentic_reward_model_examples.jsonl` can extend the pointwise reward critic training set with a fourth policy-output variant.",
        "- These labels are suitable for later best-of-N reranking, reward-model retraining, or policy optimization experiments.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pairwise-output", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--reward-output", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--limit", type=int, help="Optional number of seed examples to judge.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    source = pd.read_csv(args.input)
    seed_table = build_seed_table(source)
    if args.limit:
        seed_table = seed_table.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    args.reward_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    out = prepare_output(seed_table, args.output)
    judged_seed_ids = set(
        out[
            out["overall"].notna()
            & out["variant"].eq("agentic_selected")
            & out["judge_error"].fillna("").astype(str).eq("")
        ]["seed_id"].astype(int)
    )

    generated = 0
    errors = 0
    for _, source_row in seed_table.iterrows():
        seed_id = int(source_row["seed_id"])
        if seed_id in judged_seed_ids and not args.overwrite_existing:
            continue

        try:
            judgment = normalize_judgment(
                call_openai_judge(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    row=source_row,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_output_tokens=args.max_output_tokens,
                )
            )

            for variant in VARIANTS:
                mask = out["seed_id"].eq(seed_id) & out["variant"].eq(variant)
                for field in SCORE_FIELDS:
                    out.loc[mask, field] = judgment["scores"][variant][field]
                out.loc[mask, "rationale"] = judgment["scores"][variant]["rationale"]
                out.loc[mask, "ranking"] = json.dumps(judgment["ranking"])
                out.loc[mask, "best_variant"] = judgment["best_variant"]
                out.loc[mask, "worst_variant"] = judgment["worst_variant"]
                out.loc[mask, "judge_model"] = args.model
                out.loc[mask, "judge_error"] = ""

            generated += 1
            print(f"[{generated}] seed_id={seed_id} best={judgment['best_variant']} ranking={judgment['ranking']}")
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            mask = out["seed_id"].eq(seed_id)
            out.loc[mask, "judge_error"] = str(exc)
            print(f"ERROR seed_id={seed_id}: {exc}")

        out.to_csv(args.output, index=False)

    out.to_csv(args.output, index=False)
    scored = out[out["overall"].notna()].copy()
    pairwise_count, reward_count = build_training_exports(scored, args.pairwise_output, args.reward_output)
    args.report.write_text(build_report(scored, pairwise_count, reward_count, args.output), encoding="utf-8")

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "report": str(args.report),
        "pairwise_output": str(args.pairwise_output),
        "reward_output": str(args.reward_output),
        "model": args.model,
        "completed_seed_count": int(scored["seed_id"].nunique()),
        "scored_rows": int(len(scored)),
        "newly_judged_seed_count": generated,
        "errors": errors,
        "pairwise_examples": pairwise_count,
        "reward_examples": reward_count,
        "variants": VARIANTS,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.pairwise_output)
    print("Wrote", args.reward_output)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
