#!/usr/bin/env python3
"""Compare agentic headline selection against earlier baselines.

This script is local-only: it does not call the OpenAI API. It re-scores all
headline variants with the same local critics so the comparison is consistent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_scored_100.csv"
DEFAULT_AGENTIC = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_selected_100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "headline_agentic_vs_baselines_eval.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_agentic_vs_baselines_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_agentic_vs_baselines_metadata.json"
DEFAULT_CLICKBAIT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models" / "headline_quality_reward_distilbert"
DEFAULT_PAIRWISE_MODEL = PROJECT_ROOT / "models" / "headline_pairwise_reward_distilbert"

SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]
VARIANT_ORDER = ["original", "zero_shot", "round1_final", "round2_final", "agentic_selected"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def clean_headline(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Some API responses may arrive as a JSON object encoded inside a CSV cell.
    if text.startswith("{") and '"headlines"' in text:
        try:
            obj = json.loads(text)
            headlines = obj.get("headlines", [])
            if headlines:
                text = clean_text(headlines[0])
        except json.JSONDecodeError:
            match = re.search(r'"headlines"\s*:\s*\[\s*"([^"]+)"', text)
            if match:
                text = match.group(1)

    text = re.sub(r"^(headline|title|rewritten headline)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"'` ")
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"\.$", "", text).strip()
    return text


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


def reward_text(row: pd.Series) -> str:
    return (
        f"Category: {clean_text(row['category'])}\n"
        f"Summary: {clean_text(row['summary'])}\n"
        f"Headline: {clean_headline(row['headline'])}"
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
    quality_reward = (
        0.25 * scores_5[:, 0]
        + 0.15 * scores_5[:, 1]
        + 0.15 * scores_5[:, 2]
        + 0.20 * scores_5[:, 3]
        + 0.15 * scores_5[:, 4]
        + 0.10 * scores_5[:, 5]
    )
    return scores_5, quality_reward


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


def build_long_dataframe(baselines: pd.DataFrame, agentic: pd.DataFrame) -> pd.DataFrame:
    agentic_cols = ["seed_id", "agentic_selected_title", "agentic_selected_candidate_id", "agentic_selected_candidate_rank"]
    merged = baselines.merge(agentic[agentic_cols], on="seed_id", how="left")

    rows = []
    for row in merged.itertuples(index=False):
        base = {
            "seed_id": row.seed_id,
            "nid": row.nid,
            "news_id": row.news_id,
            "category": row.category,
            "subvert": row.subvert,
            "summary": row.summary,
        }
        variants = {
            "original": row.original_title,
            "zero_shot": row.zero_shot_title_clean,
            "round1_final": row.final_title,
            "round2_final": row.round2_final_title,
            "agentic_selected": getattr(row, "agentic_selected_title", ""),
        }
        for variant, title in variants.items():
            out = dict(base)
            out["variant"] = variant
            out["headline"] = clean_headline(title)
            out["headline_word_count"] = len(out["headline"].split())
            out["missing_headline"] = out["headline"] == ""
            out["source_candidate_id"] = getattr(row, "agentic_selected_candidate_id", "") if variant == "agentic_selected" else ""
            out["source_candidate_rank"] = getattr(row, "agentic_selected_candidate_rank", "") if variant == "agentic_selected" else ""
            rows.append(out)
    return pd.DataFrame(rows)


def score_long_dataframe(args: argparse.Namespace, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    scored = df.copy()
    titles = scored["headline"].tolist()
    texts = [reward_text(row) for _, row in scored.iterrows()]

    scored["clickbait_penalty"] = batched_clickbait_scores(
        titles, args.clickbait_model, args.batch_size, args.max_length, device
    )
    scored["predicted_clickbait"] = (scored["clickbait_penalty"] >= args.clickbait_threshold).astype(int)

    quality_dims, quality_reward = batched_quality_scores(
        texts, args.quality_model, args.batch_size, args.max_length, device
    )
    for i, field in enumerate(SCORE_FIELDS):
        scored[f"pred_{field}"] = quality_dims[:, i]
    scored["quality_reward"] = quality_reward
    scored["pairwise_reward"] = batched_pairwise_scores(
        texts, args.pairwise_model, args.batch_size, args.max_length, device
    )
    scored["final_score"] = (
        args.quality_weight * scored["quality_reward"]
        + args.pairwise_weight * scored["pairwise_reward"]
        - args.clickbait_weight * scored["clickbait_penalty"]
    )
    return scored


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored.groupby("variant", observed=True)
        .agg(
            rows=("seed_id", "count"),
            mean_clickbait_penalty=("clickbait_penalty", "mean"),
            clickbait_rate=("predicted_clickbait", "mean"),
            mean_quality_reward=("quality_reward", "mean"),
            mean_pairwise_reward=("pairwise_reward", "mean"),
            mean_final_score=("final_score", "mean"),
            mean_pred_faithfulness=("pred_faithfulness", "mean"),
            mean_pred_clarity=("pred_clarity", "mean"),
            mean_pred_specificity=("pred_specificity", "mean"),
            mean_pred_attractiveness=("pred_attractiveness", "mean"),
            mean_pred_non_clickbait=("pred_non_clickbait", "mean"),
            mean_pred_overall=("pred_overall", "mean"),
            mean_word_count=("headline_word_count", "mean"),
        )
        .reset_index()
    )
    order = {variant: i for i, variant in enumerate(VARIANT_ORDER)}
    summary["order"] = summary["variant"].map(order)
    return summary.sort_values("order").drop(columns=["order"])


def paired_deltas(scored: pd.DataFrame) -> pd.DataFrame:
    wide = scored.pivot(index="seed_id", columns="variant", values="final_score")
    rows = []
    for baseline in ["original", "zero_shot", "round1_final", "round2_final"]:
        if baseline in wide and "agentic_selected" in wide:
            delta = wide["agentic_selected"] - wide[baseline]
            rows.append(
                {
                    "comparison": f"agentic_selected - {baseline}",
                    "mean_delta_final_score": float(delta.mean()),
                    "median_delta_final_score": float(delta.median()),
                    "agentic_win_rate": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


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


def write_report(args: argparse.Namespace, scored: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, device: torch.device) -> None:
    compact_cols = [
        "variant",
        "rows",
        "mean_clickbait_penalty",
        "clickbait_rate",
        "mean_quality_reward",
        "mean_pairwise_reward",
        "mean_final_score",
        "mean_pred_overall",
    ]

    best_rows = (
        scored.sort_values(["seed_id", "final_score"], ascending=[True, False])
        .groupby("seed_id", as_index=False)
        .head(1)
    )
    best_counts = best_rows["variant"].value_counts().rename_axis("variant").reset_index(name="best_count")
    best_counts["best_rate"] = best_counts["best_count"] / scored["seed_id"].nunique()

    agentic_examples = scored[scored["variant"].eq("agentic_selected")].sort_values("final_score", ascending=False)
    example_cols = ["seed_id", "category", "headline", "clickbait_penalty", "quality_reward", "pairwise_reward", "final_score"]

    lines = [
        "# Agentic vs Baselines Local Evaluation",
        "",
        "This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.",
        "",
        "## Configuration",
        "",
        f"- Device: `{device}`",
        f"- Clickbait weight: {args.clickbait_weight}",
        f"- Quality weight: {args.quality_weight}",
        f"- Pairwise weight: {args.pairwise_weight}",
        f"- Output: `{args.output}`",
        "",
        "## Variant Summary",
        "",
        markdown_table(summary[compact_cols]),
        "",
        "## Paired Final-Score Deltas",
        "",
        markdown_table(deltas),
        "",
        "## Best Variant by Local Final Score",
        "",
        markdown_table(best_counts),
        "",
        "## Top Agentic Selected Examples",
        "",
        markdown_table(agentic_examples[example_cols].head(12)),
        "",
        "## Interpretation",
        "",
        "Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--agentic", type=Path, default=DEFAULT_AGENTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--clickbait-model", type=Path, default=DEFAULT_CLICKBAIT_MODEL)
    parser.add_argument("--quality-model", type=Path, default=DEFAULT_QUALITY_MODEL)
    parser.add_argument("--pairwise-model", type=Path, default=DEFAULT_PAIRWISE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--clickbait-threshold", type=float, default=0.5)
    parser.add_argument("--clickbait-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    device = validate_device(args.device)
    baselines = pd.read_csv(args.baselines)
    agentic = pd.read_csv(args.agentic)
    long_df = build_long_dataframe(baselines, agentic)
    scored = score_long_dataframe(args, long_df, device)
    summary = summarize(scored)
    deltas = paired_deltas(scored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    write_report(args, scored, summary, deltas, device)

    metadata = {
        "baselines": str(args.baselines),
        "agentic": str(args.agentic),
        "output": str(args.output),
        "report": str(args.report),
        "device": str(device),
        "rows": int(len(scored)),
        "seed_count": int(scored["seed_id"].nunique()),
        "variants": VARIANT_ORDER,
        "clickbait_weight": args.clickbait_weight,
        "quality_weight": args.quality_weight,
        "pairwise_weight": args.pairwise_weight,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
