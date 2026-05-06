#!/usr/bin/env python3
"""Review headline candidates for one article summary.

Product-facing workflow:
summary -> candidate generation -> unified scoring -> recommendation -> HTML report

Generation:
- API generation through the OpenAI Responses API when OPENAI_API_KEY is set.
- Deterministic fallback generation with --dry-run or --force-fallback.

Scoring:
- Unified demo scorecard: Quality, Risk/Safety, Audience Fit, Objective Fit.
- Uses the local clickbait critic if `models/clickbait_penalty_distilbert` exists.
- Falls back to heuristic clickbait risk when local model weights are missing.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
import requests

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # noqa: BLE001 - optional local critic dependency.
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "single_article_review_candidates.csv"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "demo" / "single_article_review.html"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "single_article_review_metadata.json"
DEFAULT_CLICKBAIT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"

OBJECTIVE_NAMES = {
    "trust_safety": "Trust / Safety",
    "growth": "Growth",
    "editorial": "Editorial",
    "specificity": "Specificity",
}
OBJECTIVE_DESCRIPTIONS = {
    "trust_safety": "Prefer factual, clear, low-risk headlines for trust-sensitive publishing surfaces.",
    "growth": "Prefer engaging headlines while keeping clickbait and trust risk under control.",
    "editorial": "Prefer balanced, compact, publication-ready headlines.",
    "specificity": "Prefer concrete, source-supported details without losing clarity.",
}
DECISION_WEIGHTS = {
    "trust_safety": {"quality": 0.30, "risk": 0.35, "audience": 0.15, "objective": 0.20},
    "growth": {"quality": 0.25, "risk": 0.20, "audience": 0.25, "objective": 0.30},
    "editorial": {"quality": 0.35, "risk": 0.25, "audience": 0.15, "objective": 0.25},
    "specificity": {"quality": 0.30, "risk": 0.15, "audience": 0.10, "objective": 0.45},
}
GENERATION_ROLES = ["balanced", "growth", "trust_safety", "specificity", "concise", "alternative"]
STOPWORDS = {
    "a", "about", "after", "again", "all", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "has", "have", "he", "her", "his", "how", "in", "into",
    "is", "it", "its", "more", "new", "no", "not", "of", "on", "or", "over", "that", "the", "their", "them",
    "this", "to", "up", "was", "were", "what", "when", "where", "which", "who", "will", "with", "you", "your",
}
CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"will shock you",
    r"shocking",
    r"secret",
    r"what happens next",
    r"this is why",
    r"here's why",
    r"can't stop",
    r"mind[- ]blowing",
    r"goes viral",
    r"everyone is talking",
    r"\b\d+\s+(things|ways|reasons|facts)\b",
]


@dataclass
class Candidate:
    role: str
    headline: str
    source: str


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return " ".join(str(value).split())


def clean_headline(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(headline|title)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip(" -\t\"'`")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "run"


def suffix_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def output_paths(args: argparse.Namespace, objective: str, multi_objective: bool) -> tuple[Path, Path, Path]:
    suffix_parts = []
    if args.run_name:
        suffix_parts.append(slugify(args.run_name))
    if multi_objective:
        suffix_parts.append(objective)
    suffix = "_".join(suffix_parts)
    return (
        suffix_path(args.output_csv, suffix),
        suffix_path(args.output_html, suffix),
        suffix_path(args.metadata, suffix),
    )


def index_output_path(path: Path, run_name: str) -> Path:
    suffix = slugify(run_name) + "_all" if run_name else "all"
    return suffix_path(path, suffix)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?", text)


def keyword_set(text: str) -> set[str]:
    terms = set()
    for word in words(text.lower()):
        word = word.strip("' -")
        if len(word) >= 3 and word not in STOPWORDS:
            terms.add(word)
    return terms


def extract_key_phrases(summary: str, limit: int = 8) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"(?:[A-Z][A-Za-z0-9'&.-]+(?:\s+|$)){1,5}", summary):
        phrase = clean_text(match.group(0)).strip(".,;:()[]{}")
        if len(phrase) >= 3 and phrase.lower() not in STOPWORDS and phrase not in phrases:
            phrases.append(phrase)
    for number in re.findall(r"\b\d+(?:\.\d+)?(?:%|hp|L|M|B)?\b", summary):
        if number not in phrases:
            phrases.append(number)
    for word in words(summary):
        if len(word) >= 7 and word.lower() not in STOPWORDS and word not in phrases:
            phrases.append(word)
    return phrases[:limit]


def first_sentence(summary: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", clean_text(summary))
    return parts[0] if parts and parts[0] else clean_text(summary)


def trim_title(text: str, max_words: int = 14) -> str:
    text = clean_headline(text)
    token_list = text.split()
    if len(token_list) <= max_words:
        return text.rstrip(".")
    return " ".join(token_list[:max_words]).rstrip(".,;:")


def title_case_light(text: str) -> str:
    text = clean_text(text).rstrip(".")
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    pieces = []
    for i, token in enumerate(text.split()):
        if token.isupper() or any(ch.isdigit() for ch in token):
            pieces.append(token)
        elif i > 0 and token.lower() in small:
            pieces.append(token.lower())
        else:
            pieces.append(token[:1].upper() + token[1:])
    return " ".join(pieces)


def fallback_candidates(summary: str, category: str, original_title: str, limit: int) -> list[Candidate]:
    sentence = first_sentence(summary)
    phrases = extract_key_phrases(summary)
    topic = phrases[0] if phrases else "Article"
    second = phrases[1] if len(phrases) > 1 else "Key Details"
    third = phrases[2] if len(phrases) > 2 else "Latest Developments"
    raw: list[Candidate] = []
    if original_title:
        raw.append(Candidate("original", trim_title(original_title), "user_original_title"))
    raw.extend(
        [
            Candidate("balanced", title_case_light(trim_title(sentence)), "heuristic_fallback"),
            Candidate("growth", title_case_light(trim_title(f"{topic} Highlights {second} in {category or 'Latest'} Story")), "heuristic_fallback"),
            Candidate("trust_safety", title_case_light(trim_title(f"{topic} Report Details {second}")), "heuristic_fallback"),
            Candidate("specificity", title_case_light(trim_title(f"{topic} Involves {second} and {third}")), "heuristic_fallback"),
            Candidate("concise", title_case_light(trim_title(f"{topic} {second}")), "heuristic_fallback"),
            Candidate("alternative", title_case_light(trim_title(f"What to Know About {topic} and {second}")), "heuristic_fallback"),
        ]
    )
    deduped: list[Candidate] = []
    seen = set()
    for candidate in raw:
        norm = re.sub(r"[^a-z0-9]+", " ", candidate.headline.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return deduped


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def parse_candidates_from_text(text: str) -> list[Candidate]:
    cleaned = clean_text(text)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            payload = payload.get("candidates", [])
        candidates: list[Candidate] = []
        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                if isinstance(item, dict):
                    headline = clean_headline(str(item.get("headline", "")))
                    role = clean_text(item.get("role") or GENERATION_ROLES[idx % len(GENERATION_ROLES)])
                else:
                    headline = clean_headline(str(item))
                    role = GENERATION_ROLES[idx % len(GENERATION_ROLES)]
                if headline:
                    candidates.append(Candidate(role, headline, "api_generated"))
        if candidates:
            return candidates
    except json.JSONDecodeError:
        pass
    candidates = []
    for idx, line in enumerate(text.splitlines()):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        line = re.sub(r"^[A-Za-z_ /-]+:\s*", "", line).strip()
        headline = clean_headline(line)
        if headline:
            candidates.append(Candidate(GENERATION_ROLES[idx % len(GENERATION_ROLES)], headline, "api_generated"))
    return candidates


def call_openai_candidates(args: argparse.Namespace, summary: str) -> tuple[list[Candidate], str]:
    instructions = """You are a careful headline editor for a content operations team.
Return ONLY valid JSON: [{"role":"balanced","headline":"..."}, ...]
Generate diverse candidate headlines with these roles when possible: balanced, growth, trust_safety, specificity, concise, alternative.
Rules: 6 to 14 words; faithful to the summary; no unsupported facts; no vague teasers; no exaggerated clickbait; no markdown."""
    payload: dict[str, Any] = {
        "model": args.model,
        "instructions": instructions,
        "input": f"Category: {args.category or 'unknown'}\nNumber of candidates: {args.num_candidates}\n\nSummary:\n{summary}",
        "max_output_tokens": args.max_output_tokens,
        "store": False,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.reasoning_effort != "none":
        payload["reasoning"] = {"effort": args.reasoning_effort}
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    url = args.base_url.rstrip("/") + "/responses"
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < args.retries:
                time.sleep(min(30, 2**attempt))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} {response.reason}: {response.text[:1500]}")
            raw_text = extract_output_text(response.json())
            return parse_candidates_from_text(raw_text)[: args.num_candidates], raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(f"OpenAI request failed after {args.retries + 1} attempt(s): {last_error}") from exc
    raise RuntimeError(f"OpenAI request failed: {last_error}")


def infer_device() -> str:
    if torch is None:
        return "unavailable"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_device(device: str) -> str:
    if torch is None:
        return "unavailable"
    if device == "auto":
        return infer_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is unavailable.")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Requested --device mps, but MPS is unavailable.")
    return device


def load_clickbait_scorer(model_path: Path, device: str):
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, "torch/transformers unavailable"
    if not model_path.exists():
        return None, f"missing model path: {display_path(model_path)}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return (tokenizer, model), "loaded"
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to load clickbait model: {exc}"


def model_clickbait_scores(titles: list[str], scorer, device: str, max_length: int) -> list[float] | None:
    if scorer is None or torch is None:
        return None
    tokenizer, model = scorer
    encoded = tokenizer(titles, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    return [float(x) for x in probs]


def heuristic_clickbait_penalty(title: str) -> float:
    lowered = title.lower()
    penalty = 0.02
    for pattern in CLICKBAIT_PATTERNS:
        if re.search(pattern, lowered):
            penalty += 0.18
    if "?" in title:
        penalty += 0.12
    if "!" in title:
        penalty += 0.08
    if re.search(r"\b(best|worst|amazing|incredible|unbelievable|ultimate)\b", lowered):
        penalty += 0.08
    return clamp(penalty)


def heuristic_dimensions(summary: str, title: str) -> dict[str, float]:
    summary_terms = keyword_set(summary)
    title_terms = keyword_set(title)
    support = len(title_terms & summary_terms) / max(1, len(title_terms))
    token_count = len(title.split())
    length_score = 1.0 if 6 <= token_count <= 14 else 0.78 if 4 <= token_count <= 18 else 0.52
    punctuation_penalty = 0.08 * max(0, title.count(":") - 1) + 0.06 * max(0, title.count(",") - 2)
    if title.isupper():
        punctuation_penalty += 0.20
    clarity = clamp(length_score - punctuation_penalty)
    proper_or_number = len(re.findall(r"\b(?:[A-Z][A-Za-z0-9'&.-]+|\d+(?:\.\d+)?)\b", title))
    specificity = clamp(0.35 + 0.08 * min(5, proper_or_number) + 0.25 * support)
    action_words = len(re.findall(r"\b(announces|reveals|faces|wins|defeats|offers|launches|opens|finds|issues|returns|supports|delivers|investigates|plans|approves|reports|warns|shows|highlights)\b", title.lower()))
    attractiveness = clamp(0.45 + 0.16 * min(2, action_words) + 0.18 * clarity + 0.10 * specificity)
    non_clickbait = 1.0 - heuristic_clickbait_penalty(title)
    faithfulness = clamp(0.35 + 0.65 * support)
    overall = clamp(0.30 * faithfulness + 0.25 * clarity + 0.20 * specificity + 0.15 * attractiveness + 0.10 * non_clickbait)
    return {
        "faithfulness": faithfulness,
        "clarity": clarity,
        "specificity": specificity,
        "attractiveness": attractiveness,
        "non_clickbait": non_clickbait,
        "overall": overall,
        "support_score": support,
    }


def objective_fit(objective: str, dims: dict[str, float], risk_score: float) -> float:
    support = dims["support_score"]
    if objective == "trust_safety":
        return clamp(0.35 * dims["faithfulness"] + 0.25 * dims["clarity"] + 0.25 * risk_score + 0.15 * support)
    if objective == "growth":
        return clamp(0.40 * dims["attractiveness"] + 0.20 * dims["clarity"] + 0.20 * dims["specificity"] + 0.20 * risk_score)
    if objective == "specificity":
        return clamp(0.45 * dims["specificity"] + 0.35 * support + 0.10 * dims["clarity"] + 0.10 * risk_score)
    return clamp(0.35 * dims["overall"] + 0.25 * dims["clarity"] + 0.20 * support + 0.20 * risk_score)


def audience_fit(objective: str, dims: dict[str, float], risk_score: float) -> float:
    if objective == "growth":
        return clamp(0.45 * dims["attractiveness"] + 0.25 * dims["clarity"] + 0.15 * dims["specificity"] + 0.15 * risk_score)
    if objective == "trust_safety":
        return clamp(0.40 * risk_score + 0.30 * dims["faithfulness"] + 0.20 * dims["clarity"] + 0.10 * dims["overall"])
    if objective == "specificity":
        return clamp(0.35 * dims["specificity"] + 0.30 * dims["support_score"] + 0.20 * dims["clarity"] + 0.15 * risk_score)
    return clamp(0.30 * dims["overall"] + 0.30 * dims["clarity"] + 0.20 * dims["faithfulness"] + 0.20 * risk_score)


def quality_score(dims: dict[str, float]) -> float:
    return clamp(0.35 * dims["overall"] + 0.25 * dims["faithfulness"] + 0.20 * dims["clarity"] + 0.20 * dims["support_score"])


def unified_score(record: dict[str, Any], objective: str) -> float:
    weights = DECISION_WEIGHTS[objective]
    return clamp(
        weights["quality"] * float(record["quality_score"])
        + weights["risk"] * float(record["risk_score"])
        + weights["audience"] * float(record["audience_score"])
        + weights["objective"] * float(record["objective_fit_score"])
    )


def explain_recommended(row: dict[str, Any], objective: str, generation_mode: str) -> str:
    parts = [OBJECTIVE_DESCRIPTIONS[objective]]
    if row["risk_score"] >= 0.85:
        parts.append("Risk is low after folding clickbait into the safety score.")
    if row["quality_score"] >= 0.75:
        parts.append("The quality score is strong for this article.")
    if row["objective_fit_score"] >= 0.75:
        parts.append("It has strong fit for the selected business objective.")
    parts.append(f"Generation mode: {generation_mode}.")
    return " ".join(parts)


def explain_loser(row: dict[str, Any], recommended: dict[str, Any], objective: str) -> str:
    reasons = []
    checks = [
        ("risk_score", "higher risk", 0.10),
        ("quality_score", "lower quality", 0.05),
        ("audience_score", "weaker audience fit", 0.08),
        ("objective_fit_score", f"weaker {OBJECTIVE_NAMES[objective]} fit", 0.05),
        ("support_score", "less summary support", 0.12),
    ]
    for key, label, threshold in checks:
        gap = float(recommended[key]) - float(row[key])
        if gap >= threshold:
            reasons.append(f"{label} by {gap:.2f}")
    if reasons:
        return "Not selected because it has " + "; ".join(reasons[:3]) + "."
    if row["source"] == "user_original_title":
        return "Kept as the user's original/reference title; the recommendation has a stronger combined score."
    return "Close alternative, but the recommendation gives a better overall tradeoff for this objective."


def score_candidates(
    summary: str,
    category: str,
    objective: str,
    candidates: list[Candidate],
    clickbait_scorer,
    device: str,
    max_length: int,
    generation_mode: str,
) -> pd.DataFrame:
    titles = [candidate.headline for candidate in candidates]
    model_clickbait = None
    if clickbait_scorer is not None:
        try:
            model_clickbait = model_clickbait_scores(titles, clickbait_scorer, device, max_length)
        except Exception as exc:  # noqa: BLE001
            print(f"Clickbait model scoring failed, using heuristic fallback: {exc}")
    rows = []
    for idx, candidate in enumerate(candidates):
        dims = heuristic_dimensions(summary, candidate.headline)
        clickbait_penalty = model_clickbait[idx] if model_clickbait is not None else heuristic_clickbait_penalty(candidate.headline)
        risk = 1.0 - clamp(clickbait_penalty)
        row = {
            "candidate_id": idx + 1,
            "role": candidate.role,
            "source": candidate.source,
            "headline": candidate.headline,
            "quality_score": quality_score(dims),
            "risk_score": risk,
            "audience_score": audience_fit(objective, dims, risk),
            "objective_fit_score": objective_fit(objective, dims, risk),
            "support_score": dims["support_score"],
            "clickbait_penalty": clickbait_penalty,
            "faithfulness": dims["faithfulness"],
            "clarity": dims["clarity"],
            "specificity": dims["specificity"],
            "attractiveness": dims["attractiveness"],
            "overall": dims["overall"],
            "clickbait_method": "local_clickbait_model" if model_clickbait is not None else "heuristic_clickbait",
        }
        row["unified_decision_score"] = unified_score(row, objective)
        rows.append(row)
    rows = sorted(rows, key=lambda row: (row["unified_decision_score"], row["quality_score"]), reverse=True)
    recommended = rows[0]
    for row in rows:
        row["is_recommended"] = row is recommended
        row["decision_explanation"] = explain_recommended(row, objective, generation_mode) if row is recommended else explain_loser(row, recommended, objective)
    return pd.DataFrame(rows)


def render_score_box(label: str, value: float, color: str) -> str:
    pct = max(0, min(100, value * 100))
    return f"""
      <div class=\"score-box\">
        <div class=\"score-top\"><span>{html.escape(label)}</span><b>{value:.2f}</b></div>
        <div class=\"bar\"><div class=\"fill\" style=\"width:{pct:.1f}%; background:{color};\"></div></div>
      </div>"""


def render_html_report(summary: str, category: str, objective: str, df: pd.DataFrame, metadata: dict[str, Any]) -> str:
    recommended = df[df["is_recommended"]].iloc[0]
    cards = []
    for _, row in df.iterrows():
        classes = "candidate-card recommended" if bool(row["is_recommended"]) else "candidate-card"
        selected = " <span class=\"selected\">Selected</span>" if bool(row["is_recommended"]) else ""
        cards.append(f"""
        <article class=\"{classes}\">
          <div class=\"card-head\">
            <div><span class=\"role\">{html.escape(str(row['role']))}</span>{selected}</div>
            <span class=\"source\">{html.escape(str(row['source']))}</span>
          </div>
          <h3>{html.escape(str(row['headline']))}</h3>
          <p class=\"explanation\">{html.escape(str(row['decision_explanation']))}</p>
          <div class=\"scores\">
            {render_score_box('Quality', float(row['quality_score']), '#12756d')}
            {render_score_box('Risk/Safety', float(row['risk_score']), '#315fba')}
            {render_score_box('Audience Fit', float(row['audience_score']), '#b7791f')}
            {render_score_box('Objective Fit', float(row['objective_fit_score']), '#be3a51')}
          </div>
          <div class=\"tags\">
            <span>Decision {float(row['unified_decision_score']):.2f}</span>
            <span>Support {float(row['support_score']):.2f}</span>
            <span>Clickbait {float(row['clickbait_penalty']):.2f}</span>
          </div>
        </article>""")
    metadata_items = "".join(
        f"<li><b>{html.escape(str(key))}:</b> {html.escape(str(value))}</li>"
        for key, value in metadata.items()
    )
    template = Template("""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Single Article Headline Review</title>
  <style>
    :root { --bg:#f6f5ef; --panel:#fff; --ink:#18212f; --muted:#687487; --line:#d9decc; --teal:#12756d; --soft:#dff1ed; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif; letter-spacing:0; }
    main { max-width:1180px; margin:0 auto; padding:28px 18px 48px; }
    .top { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:18px; display:grid; grid-template-columns:minmax(0,1fr) 280px; gap:18px; }
    h1 { margin:0 0 10px; font-size:clamp(28px,4vw,46px); line-height:1.03; }
    .summary { color:#334155; line-height:1.55; font-size:16px; }
    .meta { border:1px solid var(--line); border-radius:8px; background:#fbfcf9; padding:14px; }
    .meta ul { margin:0; padding-left:18px; color:#475569; line-height:1.65; }
    .recommendation { background:var(--panel); border:1px solid var(--line); border-left:5px solid var(--teal); border-radius:8px; padding:18px; margin:18px 0; box-shadow:0 10px 30px rgba(16,24,40,.08); }
    .eyebrow { color:var(--muted); text-transform:uppercase; font-size:12px; font-weight:760; margin:0 0 8px; }
    .recommendation h2 { margin:0; font-size:clamp(24px,3vw,34px); line-height:1.12; }
    .recommendation p { color:#3b4657; line-height:1.5; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:12px; }
    .candidate-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:15px; display:flex; flex-direction:column; gap:11px; }
    .candidate-card.recommended { border-color:var(--teal); box-shadow:0 0 0 2px var(--soft); }
    .card-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .role { font-size:13px; font-weight:760; text-transform:capitalize; }
    .source { color:var(--muted); font-size:11px; text-align:right; }
    .selected { color:var(--teal); border:1px solid #9dd1c8; background:var(--soft); border-radius:7px; padding:3px 7px; margin-left:6px; font-size:11px; }
    h3 { margin:0; font-size:20px; line-height:1.22; }
    .explanation { margin:0; color:#465266; line-height:1.42; border-left:3px solid var(--line); padding-left:10px; }
    .recommended .explanation { border-left-color:var(--teal); }
    .scores { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .score-box { border:1px solid var(--line); border-radius:8px; padding:9px; background:#fbfcf9; }
    .score-top { display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:12px; margin-bottom:7px; }
    .bar { height:7px; background:#e4e8dd; border-radius:999px; overflow:hidden; }
    .fill { height:100%; border-radius:999px; }
    .tags { display:flex; flex-wrap:wrap; gap:6px; margin-top:auto; }
    .tags span { border:1px solid var(--line); border-radius:7px; padding:4px 7px; color:var(--muted); font-size:11px; background:#fbfcf9; }
    @media (max-width:760px) { .top { grid-template-columns:1fr; } .scores { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <main>
    <section class=\"top\">
      <div>
        <p class=\"eyebrow\">$category_label · $objective_label</p>
        <h1>Single Article Headline Review</h1>
        <p class=\"summary\">$summary</p>
      </div>
      <aside class=\"meta\"><ul>$metadata_items</ul></aside>
    </section>
    <section class=\"recommendation\">
      <p class=\"eyebrow\">Recommended headline</p>
      <h2>$recommended_headline</h2>
      <p>$recommended_explanation</p>
    </section>
    <p class=\"eyebrow\">Candidate decision set</p>
    <section class=\"grid\">$cards</section>
  </main>
</body>
</html>
""")
    return template.substitute(
        category_label=html.escape(category or "unknown"),
        objective_label=html.escape(OBJECTIVE_NAMES[objective]),
        summary=html.escape(summary),
        metadata_items=metadata_items,
        recommended_headline=html.escape(str(recommended["headline"])),
        recommended_explanation=html.escape(str(recommended["decision_explanation"])),
        cards="\n".join(cards),
    )


def render_index_html(summary: str, category: str, results: list[dict[str, Any]], index_path: Path) -> str:
    cards = []
    for result in results:
        html_path = Path(result["html"])
        try:
            href = html_path.resolve().relative_to(index_path.parent.resolve()).as_posix()
        except ValueError:
            href = display_path(html_path)
        cards.append(f"""
        <article class=\"objective-card\">
          <p class=\"eyebrow\">{html.escape(result['objective_name'])}</p>
          <h2>{html.escape(result['recommended_headline'])}</h2>
          <p>{html.escape(result['decision_explanation'])}</p>
          <a href=\"{html.escape(href)}\">Open detail</a>
        </article>""")
    return Template("""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Single Article Review: All Objectives</title>
  <style>
    body { margin:0; background:#f6f5ef; color:#18212f; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif; letter-spacing:0; }
    main { max-width:1080px; margin:0 auto; padding:30px 18px 48px; }
    h1 { margin:0 0 10px; font-size:clamp(30px,4vw,46px); line-height:1.05; }
    .summary { color:#334155; line-height:1.55; border-bottom:1px solid #d9decc; padding-bottom:18px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-top:18px; }
    .objective-card { background:#fff; border:1px solid #d9decc; border-radius:8px; padding:16px; box-shadow:0 10px 30px rgba(16,24,40,.06); }
    .eyebrow { color:#687487; text-transform:uppercase; font-size:12px; font-weight:760; margin:0 0 8px; }
    h2 { margin:0; font-size:21px; line-height:1.2; }
    p { color:#465266; line-height:1.45; }
    a { color:#12756d; font-weight:700; text-decoration:none; }
  </style>
</head>
<body>
  <main>
    <p class=\"eyebrow\">$category · all objectives</p>
    <h1>Single Article Review</h1>
    <p class=\"summary\">$summary</p>
    <section class=\"grid\">$cards</section>
  </main>
</body>
</html>
""").substitute(category=html.escape(category or "unknown"), summary=html.escape(summary), cards="\n".join(cards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review headline candidates for one article summary.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", help="Article summary or article text.")
    source.add_argument("--summary-file", type=Path, help="Text file containing article summary or article text.")
    parser.add_argument("--title", default="", help="Optional existing/original title to include as a candidate.")
    parser.add_argument("--category", default="", help="Optional content category.")
    parser.add_argument("--objective", choices=[*sorted(OBJECTIVE_NAMES), "all"], default="editorial")
    parser.add_argument("--run-name", default="", help="Optional slug added to output filenames to avoid overwriting previous runs.")
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="none")
    parser.add_argument("--dry-run", action="store_true", help="Skip API and use deterministic fallback candidates.")
    parser.add_argument("--force-fallback", action="store_true", help="Use fallback candidates even when OPENAI_API_KEY is set.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--clickbait-model", type=Path, default=DEFAULT_CLICKBAIT_MODEL)
    parser.add_argument("--max-length", type=int, default=96)
    return parser.parse_args()


def generate_candidates(args: argparse.Namespace, summary: str) -> tuple[list[Candidate], str, str]:
    generation_mode = "heuristic_fallback"
    raw_api_output = ""
    if os.environ.get("OPENAI_API_KEY") and not args.dry_run and not args.force_fallback:
        try:
            candidates, raw_api_output = call_openai_candidates(args, summary)
            generation_mode = "api"
        except Exception as exc:  # noqa: BLE001
            print(f"API generation failed, using fallback candidates: {exc}")
            candidates = fallback_candidates(summary, args.category, args.title, args.num_candidates)
            generation_mode = "api_failed_then_fallback"
    else:
        candidates = fallback_candidates(summary, args.category, args.title, args.num_candidates)
    if args.title and all(candidate.source != "user_original_title" for candidate in candidates):
        candidates.insert(0, Candidate("original", clean_headline(args.title), "user_original_title"))
    candidates = candidates[: args.num_candidates]
    if not candidates:
        raise SystemExit("No candidates generated.")
    return candidates, generation_mode, raw_api_output


def write_one_objective(
    args: argparse.Namespace,
    summary: str,
    candidates: list[Candidate],
    generation_mode: str,
    raw_api_output: str,
    objective: str,
    clickbait_scorer,
    clickbait_status: str,
    device: str,
    multi_objective: bool,
) -> dict[str, Any]:
    scored = score_candidates(summary, args.category, objective, candidates, clickbait_scorer, device, args.max_length, generation_mode)
    output_csv, output_html, output_json = output_paths(args, objective, multi_objective)
    metadata = {
        "objective": objective,
        "objective_name": OBJECTIVE_NAMES[objective],
        "generation_mode": generation_mode,
        "model": args.model if generation_mode == "api" else "fallback",
        "candidate_count": len(scored),
        "device": device,
        "clickbait_model_status": clickbait_status,
        "run_name": args.run_name,
    }
    if raw_api_output:
        metadata["raw_api_output_preview"] = raw_api_output[:1000]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    output_html.write_text(render_html_report(summary, args.category, objective, scored, metadata), encoding="utf-8")
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    recommended = scored[scored["is_recommended"]].iloc[0]
    return {
        "objective": objective,
        "objective_name": OBJECTIVE_NAMES[objective],
        "recommended_headline": str(recommended["headline"]),
        "decision_explanation": str(recommended["decision_explanation"]),
        "unified_decision_score": round(float(recommended["unified_decision_score"]), 4),
        "csv": str(output_csv),
        "html": str(output_html),
        "metadata": str(output_json),
    }


def main() -> None:
    args = parse_args()
    summary = clean_text(args.summary_file.read_text(encoding="utf-8") if args.summary_file else args.summary)
    if not summary:
        raise SystemExit("Summary is empty.")

    candidates, generation_mode, raw_api_output = generate_candidates(args, summary)
    device = validate_device(args.device)
    clickbait_scorer, clickbait_status = load_clickbait_scorer(args.clickbait_model, device) if device != "unavailable" else (None, "torch unavailable")

    objectives = list(OBJECTIVE_NAMES) if args.objective == "all" else [args.objective]
    multi_objective = len(objectives) > 1
    results = []
    for objective in objectives:
        result = write_one_objective(
            args,
            summary,
            candidates,
            generation_mode,
            raw_api_output,
            objective,
            clickbait_scorer,
            clickbait_status,
            device,
            multi_objective,
        )
        results.append(result)
        print("Wrote", result["csv"])
        print("Wrote", result["html"])
        print("Wrote", result["metadata"])

    if multi_objective:
        index_path = index_output_path(args.output_html, args.run_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(render_index_html(summary, args.category, results, index_path), encoding="utf-8")
        print("Wrote", index_path)

    print(
        json.dumps(
            {
                "objective": args.objective,
                "generation_mode": generation_mode,
                "clickbait_model_status": clickbait_status,
                "results": [
                    {
                        "objective": result["objective"],
                        "recommended_headline": result["recommended_headline"],
                        "unified_decision_score": result["unified_decision_score"],
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
