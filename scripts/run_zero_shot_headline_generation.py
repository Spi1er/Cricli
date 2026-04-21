#!/usr/bin/env python3
"""Generate zero-shot headline baselines with the OpenAI Responses API.

The script is resumable: if an output CSV already exists, rows with a
non-empty `zero_shot_title` are skipped.

Environment:
  OPENAI_API_KEY   required
  OPENAI_MODEL     optional, defaults to gpt-5
  OPENAI_BASE_URL  optional, defaults to https://api.openai.com/v1
"""

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_eval_seed_100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_zero_shot_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_zero_shot_100_metadata.json"


INSTRUCTIONS = """You are a careful news headline editor.

Write one concise, faithful, non-clickbait headline for the provided news summary.

Rules:
- 6 to 14 words.
- Preserve the main factual claim.
- Avoid exaggeration, curiosity gaps, vague teasers, or misleading phrasing.
- Do not write question headlines.
- Do not add facts that are not supported by the summary.
- Do not use quotes unless the exact quoted phrase appears in the summary.
- Output only the headline, with no markdown and no explanation."""


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def clean_headline(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.strip("\"'` ")
    text = re.sub(r"^(headline|title)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip()


def extract_output_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def call_openai_responses(
    *,
    api_key: str,
    base_url: str,
    model: str,
    summary: str,
    category: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    retries: int,
    reasoning_effort: str | None,
) -> tuple[str, dict]:
    url = base_url.rstrip("/") + "/responses"
    user_input = f"Category: {category or 'unknown'}\n\nSummary:\n{summary}"
    payload: dict[str, object] = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                sleep_s = min(30, 2**attempt)
                time.sleep(sleep_s)
                continue
            if response.status_code >= 400:
                error_text = response.text[:2000]
                raise RuntimeError(f"{response.status_code} {response.reason}: {error_text}")
            body = response.json()
            title = clean_headline(extract_output_text(body))
            return title, body
        except Exception as exc:  # noqa: BLE001 - keep resumable API runs robust.
            last_error = exc
            if attempt < retries:
                sleep_s = min(30, 2**attempt)
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"OpenAI request failed after {retries + 1} attempt(s): {last_error}") from exc

    raise RuntimeError(f"OpenAI request failed: {last_error}")


def prepare_output(seed_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    out_cols = [
        "seed_id",
        "nid",
        "news_id",
        "summary",
        "original_title",
        "category",
        "subvert",
        "original_clickbait_penalty",
        "original_predicted_clickbait",
        "zero_shot_title",
        "zero_shot_model",
        "zero_shot_error",
    ]

    if output_path.exists():
        existing = pd.read_csv(output_path)
        for col in out_cols:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[out_cols].copy()
        for col in ["zero_shot_title", "zero_shot_model", "zero_shot_error"]:
            existing[col] = existing[col].fillna("").astype(object)
        return existing

    out = pd.DataFrame(
        {
            "seed_id": seed_df["seed_id"],
            "nid": seed_df["nid"],
            "news_id": seed_df["news_id"],
            "summary": seed_df["summary"],
            "original_title": seed_df["title"],
            "category": seed_df["category"],
            "subvert": seed_df["subvert"],
            "original_clickbait_penalty": seed_df["clickbait_penalty"],
            "original_predicted_clickbait": seed_df["predicted_clickbait"],
            "zero_shot_title": "",
            "zero_shot_model": "",
            "zero_shot_error": "",
        }
    )
    for col in ["zero_shot_title", "zero_shot_model", "zero_shot_error"]:
        out[col] = out[col].astype(object)
    return out[out_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--limit", type=int, help="Optional max number of new rows to generate.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between successful requests.")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="none")
    parser.add_argument("--dry-run", action="store_true", help="Write prepared output with prompts skipped.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Regenerate rows even when zero_shot_title is already present.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is not set. Export it or run with --dry-run.")

    seed_df = pd.read_csv(args.input)
    out_df = prepare_output(seed_df, args.output)
    for col in ["zero_shot_title", "zero_shot_model", "zero_shot_error"]:
        out_df[col] = out_df[col].fillna("").astype(object)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    errors = 0
    for idx, row in out_df.iterrows():
        existing_title = clean_text(row.get("zero_shot_title", ""))
        if existing_title and not args.overwrite_existing:
            continue
        if args.limit is not None and generated >= args.limit:
            break

        summary = clean_text(row["summary"])
        category = clean_text(row["category"])
        if args.dry_run:
            out_df.at[idx, "zero_shot_error"] = "dry_run_skipped"
            continue

        try:
            title, _ = call_openai_responses(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                summary=summary,
                category=category,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                retries=args.retries,
                reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
            )
            out_df.at[idx, "zero_shot_title"] = title
            out_df.at[idx, "zero_shot_model"] = args.model
            out_df.at[idx, "zero_shot_error"] = ""
            generated += 1
            print(f"[{generated}] seed_id={row['seed_id']} title={title}")
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001 - keep partial outputs.
            errors += 1
            out_df.at[idx, "zero_shot_error"] = str(exc)
            print(f"ERROR seed_id={row['seed_id']}: {exc}")

        out_df.to_csv(args.output, index=False)

    out_df.to_csv(args.output, index=False)
    completed = out_df["zero_shot_title"].fillna("").astype(str).str.len().gt(0).sum()
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "total_rows": len(out_df),
        "completed_rows": int(completed),
        "newly_generated_rows": generated,
        "errors": errors,
        "dry_run": args.dry_run,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
