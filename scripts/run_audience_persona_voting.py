#!/usr/bin/env python3
"""Run audience persona voting agents over multi-objective headline candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_candidate_matrix.csv"
DEFAULT_SELECTION = PROJECT_ROOT / "data" / "processed" / "headline_multi_agent_objective_selection.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes_metadata.json"

PERSONAS = {
    "trust_sensitive_reader": "Prioritizes factual accuracy, non-clickbait wording, and whether the headline feels safe to trust.",
    "growth_reader": "Represents user-growth goals: wants a headline that attracts clicks without feeling manipulative.",
    "busy_news_reader": "Skims quickly and rewards clarity, compactness, and immediate usefulness.",
    "editorial_reviewer": "Acts like a newsroom editor, balancing newsworthiness, specificity, style, and faithfulness.",
}

SCORE_FIELDS = ["trust", "engagement", "clarity", "audience_fit", "overall"]

INSTRUCTIONS = """You are coordinating a small panel of audience persona agents for news headline selection.

Each persona must independently evaluate the same candidate headlines against the article summary.

Return only valid JSON with this exact schema:
{
  "personas": {
    "trust_sensitive_reader": {
      "scores": {
        "c1": {
          "trust": 1,
          "engagement": 1,
          "clarity": 1,
          "audience_fit": 1,
          "overall": 1,
          "rationale": "short reason"
        }
      },
      "ranking": ["c1", "c2"],
      "best_candidate": "c1"
    },
    "growth_reader": {...},
    "busy_news_reader": {...},
    "editorial_reviewer": {...}
  },
  "consensus_ranking": ["c1", "c2"],
  "consensus_best_candidate": "c1"
}

Scoring rules:
- Scores are integers from 1 to 5.
- trust: the headline is faithful, avoids unsupported claims, and does not feel deceptive.
- engagement: the headline is interesting enough to make the target reader want to read.
- clarity: the headline is understandable and concise.
- audience_fit: the headline fits that persona's stated preference.
- overall: that persona's final judgment.

Important:
- Do not reward unsupported specificity or invented details.
- Do not require every persona to agree.
- If two candidates are identical, give them identical scores for the same persona.
- Candidate ids must be exactly the provided ids such as c1, c2, c3."""


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


def build_candidate_sets(matrix: pd.DataFrame, selection: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    rows = []
    required_variants = ["original", "zero_shot", "round2_final", "agentic_selected", "generic_sft", "specificity_sft"]

    for seed_id, group in matrix.groupby("seed_id", sort=True):
        chosen_headlines = set(
            selection[selection["seed_id"].eq(seed_id)]["selected_headline"].map(clean_text).tolist()
        )
        required = group[group["variant"].isin(required_variants)].copy()
        selected = group[group["headline"].map(clean_text).isin(chosen_headlines)].copy()
        candidate_pool = pd.concat([required, selected], ignore_index=True)
        candidate_pool = candidate_pool.drop_duplicates(["headline"]).head(max_candidates)

        for idx, row in enumerate(candidate_pool.itertuples(index=False), start=1):
            obj = row._asdict()
            rows.append(
                {
                    "seed_id": int(seed_id),
                    "candidate_label": f"c{idx}",
                    "variant": obj["variant"],
                    "candidate_id": obj["candidate_id"],
                    "candidate_source": obj["candidate_source"],
                    "category": obj["category"],
                    "summary": obj["summary"],
                    "headline": clean_text(obj["headline"]),
                    "clickbait_penalty": obj["clickbait_penalty"],
                    "local_final_score": obj["local_final_score"],
                    "llm_overall": obj.get("llm_overall"),
                }
            )
    return pd.DataFrame(rows)


def call_persona_panel(
    *,
    api_key: str,
    base_url: str,
    model: str,
    seed_candidates: pd.DataFrame,
    timeout: int,
    retries: int,
    max_output_tokens: int,
) -> dict:
    url = base_url.rstrip("/") + "/responses"
    first = seed_candidates.iloc[0]
    candidates_text = "\n".join(
        f"- {row.candidate_label}: {clean_text(row.headline)}"
        for row in seed_candidates.itertuples(index=False)
    )
    persona_text = "\n".join(f"- {name}: {description}" for name, description in PERSONAS.items())
    user_input = (
        f"Category: {clean_text(first['category']) or 'unknown'}\n\n"
        f"Summary:\n{clean_text(first['summary'])}\n\n"
        f"Personas:\n{persona_text}\n\n"
        f"Candidate headlines:\n{candidates_text}\n"
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(f"Persona panel failed after {retries + 1} attempt(s): {last_error}") from exc
    raise RuntimeError(f"Persona panel failed: {last_error}")


def normalize_panel(panel: dict, seed_candidates: pd.DataFrame, model: str) -> list[dict]:
    candidate_lookup = {
        row.candidate_label: row._asdict()
        for row in seed_candidates.itertuples(index=False)
    }
    consensus_best = clean_text(panel.get("consensus_best_candidate", ""))
    consensus_ranking = panel.get("consensus_ranking", [])
    if not isinstance(consensus_ranking, list):
        consensus_ranking = []

    rows = []
    personas = panel.get("personas", {})
    for persona, description in PERSONAS.items():
        persona_obj = personas.get(persona, {})
        scores = persona_obj.get("scores", {})
        ranking = persona_obj.get("ranking", [])
        best_candidate = clean_text(persona_obj.get("best_candidate", ""))
        for label, candidate in candidate_lookup.items():
            score_obj = scores.get(label, {})
            row = dict(candidate)
            row["persona"] = persona
            row["persona_description"] = description
            for field in SCORE_FIELDS:
                value = score_obj.get(field)
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    value = max(1, min(5, value))
                row[field] = value
            row["rationale"] = clean_text(score_obj.get("rationale", ""))
            row["persona_ranking"] = json.dumps(ranking)
            row["persona_best_candidate"] = best_candidate
            row["is_persona_best"] = label == best_candidate
            row["consensus_ranking"] = json.dumps(consensus_ranking)
            row["consensus_best_candidate"] = consensus_best
            row["is_consensus_best"] = label == consensus_best
            row["judge_model"] = model
            row["judge_error"] = ""
            rows.append(row)
    return rows


def prepare_existing(output: Path) -> pd.DataFrame:
    if output.exists():
        return pd.read_csv(output)
    return pd.DataFrame()


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


def build_report(votes: pd.DataFrame, output: Path, pairwise_seed_count: int) -> str:
    persona_winners = (
        votes[votes["is_persona_best"].eq(True)]
        .groupby(["persona", "variant"])
        .size()
        .reset_index(name="best_count")
        .sort_values(["persona", "best_count"], ascending=[True, False])
    )
    consensus = (
        votes[votes["is_consensus_best"].eq(True)]
        .drop_duplicates(["seed_id", "candidate_label"])
        .groupby("variant")
        .size()
        .reset_index(name="consensus_best_count")
        .sort_values("consensus_best_count", ascending=False)
    )
    mean_scores = (
        votes.groupby(["persona", "variant"], dropna=False)[SCORE_FIELDS]
        .mean()
        .reset_index()
        .sort_values(["persona", "overall"], ascending=[True, False])
    )
    lines = [
        "# Audience Persona Voting Agents",
        "",
        f"- Scores: `{output}`",
        f"- Completed seed count: {pairwise_seed_count:,}",
        f"- Personas: {', '.join(PERSONAS)}",
        "",
        "## Persona Best Counts",
        "",
        markdown_table(persona_winners),
        "",
        "## Consensus Best Counts",
        "",
        markdown_table(consensus),
        "",
        "## Mean Persona Scores",
        "",
        markdown_table(mean_scores),
        "",
        "## Agentic RL Interpretation",
        "",
        "The persona panel adds separate reward channels for different user objectives. These scores can be used as additional reward dimensions in the multi-agent objective matrix or as preference data for a future selector policy.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--limit", type=int, help="Optional number of seed examples to judge.")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    matrix = pd.read_csv(args.matrix)
    selection = pd.read_csv(args.selection)
    candidates = build_candidate_sets(matrix, selection, args.max_candidates)
    seed_ids = sorted(candidates["seed_id"].unique().tolist())
    if args.limit:
        seed_ids = seed_ids[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    existing = prepare_existing(args.output)
    completed = set()
    if not existing.empty and not args.overwrite_existing:
        good = existing[existing["judge_error"].fillna("").astype(str).eq("")]
        completed = set(good["seed_id"].astype(int).unique().tolist())
    all_rows = [] if existing.empty or args.overwrite_existing else existing.to_dict(orient="records")

    generated = 0
    errors = 0
    for seed_id in seed_ids:
        if seed_id in completed:
            continue
        seed_candidates = candidates[candidates["seed_id"].eq(seed_id)].copy()
        try:
            panel = call_persona_panel(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                seed_candidates=seed_candidates,
                timeout=args.timeout,
                retries=args.retries,
                max_output_tokens=args.max_output_tokens,
            )
            rows = normalize_panel(panel, seed_candidates, args.model)
            all_rows.extend(rows)
            generated += 1
            print(f"[{generated}] seed_id={seed_id} consensus={panel.get('consensus_best_candidate')}")
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            for row in seed_candidates.to_dict(orient="records"):
                row["persona"] = ""
                row["judge_model"] = args.model
                row["judge_error"] = str(exc)
                all_rows.append(row)
            print(f"ERROR seed_id={seed_id}: {exc}")

        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    votes = pd.DataFrame(all_rows)
    votes.to_csv(args.output, index=False)
    completed_seed_count = int(votes[votes.get("judge_error", "").fillna("").astype(str).eq("")]["seed_id"].nunique()) if not votes.empty else 0
    args.report.write_text(build_report(votes[votes.get("judge_error", "").fillna("").astype(str).eq("")], args.output, completed_seed_count), encoding="utf-8")
    metadata = {
        "matrix": str(args.matrix),
        "selection": str(args.selection),
        "output": str(args.output),
        "report": str(args.report),
        "model": args.model,
        "personas": PERSONAS,
        "max_candidates": args.max_candidates,
        "completed_seed_count": completed_seed_count,
        "newly_judged_seed_count": generated,
        "errors": errors,
        "rows": int(len(votes)),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
