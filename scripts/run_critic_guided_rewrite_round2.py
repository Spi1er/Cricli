#!/usr/bin/env python3
"""Second-round critic-guided rewrite for headlines still above threshold."""

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_critic_guided_scored_100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_100.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_100_metadata.json"


INSTRUCTIONS = """You are a strict news copy editor reducing clickbait language.

Rewrite the headline into a plain, factual news headline. The previous rewrite was still flagged as clickbait by a trained critic.

Hard rules:
- 6 to 14 words.
- Start with the concrete subject, person, organization, place, or event.
- Do not start with a number.
- Do not use listicle framing.
- Do not use: amazing, fascinating, surprising, genius, powerful, best, perfect, shocking, lovers, feast, inspired, you, your, everyone.
- Do not use "facts about", "things about", "things to", "how to", "why you", or "what you".
- Avoid lifestyle-magazine or promotional wording.
- Preserve the main factual claim from the summary.
- Do not write question headlines.
- Do not add facts unsupported by the summary.
- Output only the rewritten headline, with no markdown and no explanation."""


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def clean_headline(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(headline|title|rewritten headline)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"'` ")
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"\.$", "", text).strip()
    return text


def extract_output_text(response: dict) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def call_openai(
    *,
    api_key: str,
    base_url: str,
    model: str,
    summary: str,
    current_title: str,
    original_title: str,
    penalty: float,
    category: str,
    max_output_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/responses"
    user_input = (
        f"Category: {category or 'unknown'}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Original headline:\n{original_title}\n\n"
        f"Current rewritten headline:\n{current_title}\n\n"
        f"Critic feedback:\n"
        f"The current rewritten headline still has clickbait probability {penalty:.3f}. "
        "Replace it with a plainer news headline. Avoid listicle, lifestyle, promotional, or curiosity-gap wording."
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
            return clean_headline(extract_output_text(response.json()))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(f"OpenAI request failed after {retries + 1} attempt(s): {last_error}") from exc
    raise RuntimeError(f"OpenAI request failed: {last_error}")


def prepare_output(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    out = df.copy()
    for col in ["round2_title", "round2_model", "round2_error"]:
        if col not in out.columns:
            out[col] = ""
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for col in out.columns:
            if col not in existing.columns:
                existing[col] = out[col]
        out = existing[out.columns].copy()
    for col in ["round2_title", "round2_model", "round2_error"]:
        out[col] = out[col].fillna("").astype(object)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    source = pd.read_csv(args.input)
    out = prepare_output(source, args.output)
    targets = out["rewritten_clickbait_penalty"] >= args.threshold

    generated = 0
    errors = 0
    for idx, row in out[targets].iterrows():
        if clean_text(row.get("round2_title", "")) and not args.overwrite_existing:
            continue
        if args.limit is not None and generated >= args.limit:
            break

        try:
            title = call_openai(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                summary=clean_text(row["summary"]),
                current_title=clean_text(row["rewritten_title_clean"]),
                original_title=clean_text(row["original_title"]),
                penalty=float(row["rewritten_clickbait_penalty"]),
                category=clean_text(row["category"]),
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            out.at[idx, "round2_title"] = title
            out.at[idx, "round2_model"] = args.model
            out.at[idx, "round2_error"] = ""
            generated += 1
            print(f"[{generated}] seed_id={row['seed_id']} round2={title}")
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            out.at[idx, "round2_error"] = str(exc)
            print(f"ERROR seed_id={row['seed_id']}: {exc}")

        out.to_csv(args.output, index=False)

    out.to_csv(args.output, index=False)
    completed = out.loc[targets, "round2_title"].fillna("").astype(str).str.len().gt(0).sum()
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "threshold": args.threshold,
        "target_rows": int(targets.sum()),
        "completed_target_rows": int(completed),
        "newly_rewritten_rows": generated,
        "errors": errors,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
