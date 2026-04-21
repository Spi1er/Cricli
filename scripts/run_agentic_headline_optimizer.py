#!/usr/bin/env python3
"""Run an agentic headline optimization loop with local critics.

Workflow:
  1. Generate K candidate headlines with the OpenAI Responses API.
  2. Score each candidate with local clickbait, quality, and pairwise reward critics.
  3. Select the highest-scoring candidate for each seed item.

The script is resumable. Existing candidates are reused unless
`--overwrite-existing` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_generation_eval_seed_100.csv"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_candidates_100.csv"
DEFAULT_SELECTED = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_selected_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_metadata.json"
DEFAULT_CLICKBAIT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models" / "headline_quality_reward_distilbert"
DEFAULT_PAIRWISE_MODEL = PROJECT_ROOT / "models" / "headline_pairwise_reward_distilbert"

SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


INSTRUCTIONS = """You are a careful news headline editor.

Generate multiple candidate headlines for the provided news summary.

Rules:
- Each headline must be 6 to 14 words.
- Preserve the main factual claim.
- Make each candidate specific, informative, and non-clickbait.
- Avoid exaggeration, curiosity gaps, vague teasers, listicle framing, and promotional phrasing.
- Do not write question headlines.
- Do not add facts that are not supported by the summary.
- Return valid JSON only, with this schema:
  {"headlines": ["candidate 1", "candidate 2", "candidate 3"]}"""


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def clean_headline(text: object) -> str:
    text = clean_text(text)
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(headline|title)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
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


def parse_headlines(text: str, expected: int) -> list[str]:
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    candidates: list[str] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict) and isinstance(obj.get("headlines"), list):
            candidates = [clean_headline(item) for item in obj["headlines"]]
        elif isinstance(obj, list):
            candidates = [clean_headline(item) for item in obj]
    except json.JSONDecodeError:
        match = re.search(r"\{.*\"headlines\".*\}", text)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("headlines"), list):
                    candidates = [clean_headline(item) for item in obj["headlines"]]
            except json.JSONDecodeError:
                candidates = []
        if not candidates:
            lines = [re.sub(r"^\d+[\).\s-]*", "", line).strip() for line in text.splitlines()]
            candidates = [clean_headline(line) for line in lines if line.strip()]

    deduped = []
    seen = set()
    for title in candidates:
        key = title.lower()
        if title and key not in seen:
            deduped.append(title)
            seen.add(key)
    return deduped[:expected]


def call_openai_candidates(
    *,
    api_key: str,
    base_url: str,
    model: str,
    summary: str,
    category: str,
    num_candidates: int,
    temperature: float | None,
    max_output_tokens: int,
    timeout: int,
    retries: int,
    reasoning_effort: str | None,
) -> list[str]:
    url = base_url.rstrip("/") + "/responses"
    user_input = (
        f"Category: {category or 'unknown'}\n\n"
        f"Number of candidates: {num_candidates}\n\n"
        f"Summary:\n{summary}"
    )
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
            candidates = parse_headlines(extract_output_text(response.json()), num_candidates)
            if not candidates:
                raise RuntimeError("API response did not contain parseable candidates.")
            return candidates
        except Exception as exc:  # noqa: BLE001 - keep resumable API runs robust.
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(f"OpenAI request failed after {retries + 1} attempt(s): {last_error}") from exc
    raise RuntimeError(f"OpenAI request failed: {last_error}")


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_device(device: str) -> torch.device:
    if device == "auto":
        return infer_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is unavailable.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps, but MPS is unavailable.")
    return torch.device(device)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        return self.tokenizer(self.texts[index], truncation=True, max_length=self.max_length)


class DistilBertRewardRegressor(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768))
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(getattr(config, "seq_classif_dropout", 0.2))
        self.regressor = nn.Linear(hidden_size, len(SCORE_FIELDS))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = torch.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return torch.sigmoid(self.regressor(pooled))


class DistilBertPairwiseRewardModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768))
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(getattr(config, "seq_classif_dropout", 0.2))
        self.reward_head = nn.Linear(hidden_size, 1)
        self.post_init()

    def score(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = torch.relu(self.pre_classifier(pooled))
        pooled = self.dropout(pooled)
        return self.reward_head(pooled).squeeze(-1)


def batched_clickbait_scores(
    titles: list[str], model_path: Path, batch_size: int, max_length: int, device: torch.device
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    loader = DataLoader(
        TextDataset(titles, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = torch.softmax(model(**batch).logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def reward_text(row: pd.Series, title: str) -> str:
    return (
        f"Category: {clean_text(row.get('category', ''))}\n"
        f"Summary: {clean_text(row.get('summary', ''))}\n"
        f"Headline: {title}"
    )


def batched_quality_scores(
    texts: list[str], model_path: Path, batch_size: int, max_length: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBertRewardRegressor.from_pretrained(model_path, attn_implementation="eager").to(device).eval()
    loader = DataLoader(
        TextDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            preds.append(model(**batch).cpu().numpy())
    normalized = np.vstack(preds).astype(np.float32)
    scores_5 = np.clip(normalized * 4.0 + 1.0, 1.0, 5.0)
    reward = (
        0.25 * scores_5[:, 0]
        + 0.15 * scores_5[:, 1]
        + 0.15 * scores_5[:, 2]
        + 0.20 * scores_5[:, 3]
        + 0.15 * scores_5[:, 4]
        + 0.10 * scores_5[:, 5]
    )
    return scores_5, reward


def batched_pairwise_scores(
    texts: list[str], model_path: Path, batch_size: int, max_length: int, device: torch.device
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBertPairwiseRewardModel.from_pretrained(model_path, attn_implementation="eager").to(device).eval()
    loader = DataLoader(
        TextDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores.extend(model.score(batch["input_ids"], batch["attention_mask"]).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def load_existing_candidates(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def existing_seed_ids(df: pd.DataFrame) -> set[int]:
    if df.empty or "seed_id" not in df.columns:
        return set()
    valid = df["candidate_title"].fillna("").astype(str).str.len().gt(0)
    return set(df.loc[valid, "seed_id"].astype(int).tolist())


def dry_run_candidates(row: pd.Series, num_candidates: int) -> list[str]:
    base = clean_headline(row.get("title", row.get("original_title", "")))
    category = clean_text(row.get("category", "news")).title()
    options = [
        base,
        f"{category} Report Details Main Developments",
        f"Officials Outline Key Updates in {category} Story",
        f"New Details Emerge in {category} Report",
    ]
    return options[:num_candidates]


def generate_candidates(args: argparse.Namespace, seed_df: pd.DataFrame) -> pd.DataFrame:
    existing = load_existing_candidates(args.output_candidates)
    done_ids = existing_seed_ids(existing) if not args.overwrite_existing else set()
    rows = [] if args.overwrite_existing else existing.to_dict("records")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is not set. Export it or run with --dry-run.")

    generated = 0
    errors = 0
    for _, row in seed_df.iterrows():
        seed_id = int(row["seed_id"])
        if seed_id in done_ids:
            continue
        if args.limit is not None and generated >= args.limit:
            break

        try:
            if args.dry_run:
                candidates = dry_run_candidates(row, args.num_candidates)
            else:
                candidates = call_openai_candidates(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    summary=clean_text(row["summary"]),
                    category=clean_text(row["category"]),
                    num_candidates=args.num_candidates,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
                )
            for rank, title in enumerate(candidates, start=1):
                rows.append(
                    {
                        "seed_id": seed_id,
                        "candidate_id": f"{seed_id}_c{rank}",
                        "candidate_rank": rank,
                        "nid": row.get("nid", ""),
                        "news_id": row.get("news_id", ""),
                        "summary": clean_text(row.get("summary", "")),
                        "original_title": clean_text(row.get("title", row.get("original_title", ""))),
                        "category": clean_text(row.get("category", "")),
                        "subvert": clean_text(row.get("subvert", "")),
                        "original_clickbait_penalty": row.get("clickbait_penalty", row.get("original_clickbait_penalty", np.nan)),
                        "original_predicted_clickbait": row.get("predicted_clickbait", row.get("original_predicted_clickbait", np.nan)),
                        "candidate_title": clean_headline(title),
                        "candidate_model": "dry_run" if args.dry_run else args.model,
                        "candidate_error": "",
                    }
                )
            generated += 1
            print(f"[{generated}] seed_id={seed_id} candidates={len(candidates)}")
            if not args.dry_run:
                time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            rows.append(
                {
                    "seed_id": seed_id,
                    "candidate_id": f"{seed_id}_error",
                    "candidate_rank": 0,
                    "nid": row.get("nid", ""),
                    "news_id": row.get("news_id", ""),
                    "summary": clean_text(row.get("summary", "")),
                    "original_title": clean_text(row.get("title", row.get("original_title", ""))),
                    "category": clean_text(row.get("category", "")),
                    "subvert": clean_text(row.get("subvert", "")),
                    "original_clickbait_penalty": row.get("clickbait_penalty", row.get("original_clickbait_penalty", np.nan)),
                    "original_predicted_clickbait": row.get("predicted_clickbait", row.get("original_predicted_clickbait", np.nan)),
                    "candidate_title": "",
                    "candidate_model": args.model,
                    "candidate_error": str(exc),
                }
            )
            print(f"ERROR seed_id={seed_id}: {exc}")

        pd.DataFrame(rows).to_csv(args.output_candidates, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(args.output_candidates, index=False)
    args._generated_seed_count = generated
    args._generation_error_count = errors
    return out


def score_candidates(args: argparse.Namespace, candidates: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    valid_mask = candidates["candidate_title"].fillna("").astype(str).str.len().gt(0)
    valid = candidates.loc[valid_mask].copy().reset_index(drop=True)
    if valid.empty:
        return candidates

    titles = valid["candidate_title"].map(clean_headline).tolist()
    texts = [reward_text(row, title) for (_, row), title in zip(valid.iterrows(), titles)]

    clickbait = batched_clickbait_scores(titles, args.clickbait_model, args.batch_size, args.max_length, device)
    quality_dims, quality_reward = batched_quality_scores(texts, args.quality_model, args.batch_size, args.max_length, device)
    pairwise_reward = batched_pairwise_scores(texts, args.pairwise_model, args.batch_size, args.max_length, device)

    valid["candidate_clickbait_penalty"] = clickbait
    valid["candidate_predicted_clickbait"] = (valid["candidate_clickbait_penalty"] >= args.clickbait_threshold).astype(int)
    for i, field in enumerate(SCORE_FIELDS):
        valid[f"candidate_pred_{field}"] = quality_dims[:, i]
    valid["candidate_quality_reward"] = quality_reward
    valid["candidate_pairwise_reward"] = pairwise_reward
    valid["agentic_final_score"] = (
        args.quality_weight * valid["candidate_quality_reward"]
        + args.pairwise_weight * valid["candidate_pairwise_reward"]
        - args.clickbait_weight * valid["candidate_clickbait_penalty"]
    )

    scored = candidates.copy()
    for col in valid.columns:
        if col not in scored.columns:
            scored[col] = np.nan
    scored.loc[valid_mask, valid.columns] = valid.to_numpy()
    return scored


def select_best(scored: pd.DataFrame) -> pd.DataFrame:
    valid = scored[scored["candidate_title"].fillna("").astype(str).str.len().gt(0)].copy()
    valid = valid.sort_values(["seed_id", "agentic_final_score", "candidate_rank"], ascending=[True, False, True])
    selected = valid.groupby("seed_id", as_index=False).head(1).copy()
    selected = selected.rename(
        columns={
            "candidate_title": "agentic_selected_title",
            "candidate_id": "agentic_selected_candidate_id",
            "candidate_rank": "agentic_selected_candidate_rank",
            "candidate_clickbait_penalty": "agentic_clickbait_penalty",
            "candidate_predicted_clickbait": "agentic_predicted_clickbait",
            "candidate_quality_reward": "agentic_quality_reward",
            "candidate_pairwise_reward": "agentic_pairwise_reward",
        }
    )
    return selected.reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    string_df = df.copy()
    for col in string_df.columns:
        if pd.api.types.is_float_dtype(string_df[col]):
            string_df[col] = string_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            string_df[col] = string_df[col].map(lambda value: "" if pd.isna(value) else str(value))

    header = "| " + " | ".join(string_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(string_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).replace("|", "\\|") for col in string_df.columns) + " |"
        for _, row in string_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(args: argparse.Namespace, candidates: pd.DataFrame, selected: pd.DataFrame, device: torch.device) -> None:
    lines = [
        "# Agentic Headline Optimizer",
        "",
        "This run generates multiple candidates per seed headline, scores them with local critics, and selects the best candidate by weighted reward.",
        "",
        "## Configuration",
        "",
        f"- Device: `{device}`",
        f"- Generator model: `{args.model}`",
        f"- Candidates per seed: {args.num_candidates}",
        f"- Clickbait weight: {args.clickbait_weight}",
        f"- Quality weight: {args.quality_weight}",
        f"- Pairwise weight: {args.pairwise_weight}",
        f"- Dry run: {args.dry_run}",
        "",
        "## Summary",
        "",
        f"- Candidate rows: {len(candidates):,}",
        f"- Selected rows: {len(selected):,}",
    ]

    if not selected.empty:
        lines.extend(
            [
                f"- Mean selected clickbait penalty: {selected['agentic_clickbait_penalty'].mean():.4f}",
                f"- Selected clickbait rate: {selected['agentic_predicted_clickbait'].mean():.2%}",
                f"- Mean selected quality reward: {selected['agentic_quality_reward'].mean():.4f}",
                f"- Mean selected pairwise reward: {selected['agentic_pairwise_reward'].mean():.4f}",
                f"- Mean selected final score: {selected['agentic_final_score'].mean():.4f}",
            ]
        )
        if "original_clickbait_penalty" in selected.columns:
            original = pd.to_numeric(selected["original_clickbait_penalty"], errors="coerce")
            lines.append(f"- Mean original clickbait penalty for selected seeds: {original.mean():.4f}")

        preview_cols = [
            "seed_id",
            "category",
            "original_title",
            "agentic_selected_title",
            "agentic_clickbait_penalty",
            "agentic_quality_reward",
            "agentic_pairwise_reward",
            "agentic_final_score",
        ]
        preview = selected[preview_cols].head(12).copy()
        lines.extend(["", "## Selected Examples", "", markdown_table(preview), ""])

    lines.extend(
        [
            "## Next Training Use",
            "",
            "- Use selected candidates as policy outputs for comparison against zero-shot and critic-guided rewrite baselines.",
            "- Use candidate rankings as synthetic preference data: chosen = selected candidate, rejected = lower-scoring candidates from the same seed.",
            "- Use the final score as a local reward for best-of-N sampling, rejection sampling, or later RL-style policy optimization.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--clickbait-model", type=Path, default=DEFAULT_CLICKBAIT_MODEL)
    parser.add_argument("--quality-model", type=Path, default=DEFAULT_QUALITY_MODEL)
    parser.add_argument("--pairwise-model", type=Path, default=DEFAULT_PAIRWISE_MODEL)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Optional max number of new seed rows to generate.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="none")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--clickbait-threshold", type=float, default=0.5)
    parser.add_argument("--clickbait-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    args.output_selected.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    device = validate_device(args.device)
    seed_df = pd.read_csv(args.input)
    candidates = generate_candidates(args, seed_df)
    scored = score_candidates(args, candidates, device)
    scored.to_csv(args.output_candidates, index=False)

    selected = select_best(scored)
    selected.to_csv(args.output_selected, index=False)
    write_report(args, scored, selected, device)

    metadata = {
        "input": str(args.input),
        "output_candidates": str(args.output_candidates),
        "output_selected": str(args.output_selected),
        "report": str(args.report),
        "model": args.model,
        "base_url": args.base_url,
        "num_candidates": args.num_candidates,
        "dry_run": args.dry_run,
        "device": str(device),
        "generated_seed_count": getattr(args, "_generated_seed_count", 0),
        "generation_error_count": getattr(args, "_generation_error_count", 0),
        "candidate_rows": int(len(scored)),
        "selected_rows": int(len(selected)),
        "clickbait_weight": args.clickbait_weight,
        "quality_weight": args.quality_weight,
        "pairwise_weight": args.pairwise_weight,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output_candidates)
    print("Wrote", args.output_selected)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
