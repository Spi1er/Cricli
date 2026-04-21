#!/usr/bin/env python3
"""Benchmark local critic latency for headline scoring.

Benchmarks:
- clickbait penalty critic: title -> P(clickbait)
- multi-dimensional reward critic: summary + title -> six quality scores
- pairwise reward critic: summary + chosen/rejected title -> preference margin

Outputs:
- data/processed/critic_latency_benchmark.json
- data/processed/critic_latency_benchmark.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertPreTrainedModel
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADLINES = PROJECT_ROOT / "data" / "processed" / "headline_generation_rewrite_round2_critic_guided_scored_100.csv"
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "processed" / "headline_quality_pairwise_preferences.jsonl"
DEFAULT_CLICKBAIT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models" / "headline_quality_reward_distilbert"
DEFAULT_PAIRWISE_MODEL = PROJECT_ROOT / "models" / "headline_pairwise_reward_distilbert"
DEFAULT_JSON = PROJECT_ROOT / "data" / "processed" / "critic_latency_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "critic_latency_benchmark.md"

SCORE_FIELDS = ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        return self.tokenizer(self.texts[index], truncation=True, max_length=self.max_length)


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def encode(self, row: dict, title_key: str) -> dict:
        text = f"Category: {row['category']}\nSummary: {row['summary']}\nHeadline: {row[title_key]}"
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        chosen = self.encode(row, "chosen_title")
        rejected = self.encode(row, "rejected_title")
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


class PairwiseCollator:
    def __init__(self, tokenizer) -> None:
        self.pad = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        chosen = [{"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]} for f in features]
        rejected = [{"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]} for f in features]
        chosen_batch = self.pad(chosen)
        rejected_batch = self.pad(rejected)
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


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

    def forward(self, chosen_input_ids=None, chosen_attention_mask=None, rejected_input_ids=None, rejected_attention_mask=None):
        return self.score(chosen_input_ids, chosen_attention_mask) - self.score(rejected_input_ids, rejected_attention_mask)


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
        raise RuntimeError("Requested cuda, but CUDA is unavailable.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested mps, but MPS is unavailable.")
    return torch.device(device)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_loader(name: str, loader: DataLoader, fn, device: torch.device, warmup: int) -> dict:
    count = 0
    batch_times = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            start = time.perf_counter()
            _ = fn(batch)
            sync_device(device)
            elapsed = time.perf_counter() - start
            batch_size = next(iter(batch.values())).shape[0]
            if i >= warmup:
                batch_times.append(elapsed)
                count += batch_size

    total = float(sum(batch_times))
    examples_per_second = count / total if total > 0 else 0.0
    ms_per_example = 1000.0 / examples_per_second if examples_per_second > 0 else float("inf")
    return {
        "name": name,
        "measured_examples": count,
        "measured_batches": len(batch_times),
        "total_seconds": total,
        "examples_per_second": examples_per_second,
        "ms_per_example": ms_per_example,
        "mean_batch_seconds": float(np.mean(batch_times)) if batch_times else 0.0,
        "p95_batch_seconds": float(np.percentile(batch_times, 95)) if batch_times else 0.0,
    }


def headline_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for row in df.itertuples(index=False):
        texts.extend(
            [
                str(row.original_title),
                str(row.zero_shot_title_clean),
                str(row.round2_final_title),
            ]
        )
    return texts


def quality_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for row in df.itertuples(index=False):
        for title in [row.original_title, row.zero_shot_title_clean, row.round2_final_title]:
            texts.append(f"Category: {row.category}\nSummary: {row.summary}\nHeadline: {title}")
    return texts


def load_pair_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(results: list[dict], output_json: Path, report_path: Path, device: torch.device, batch_size: int) -> None:
    lines = [
        "# Critic Latency Benchmark",
        "",
        f"- Device: `{device}`",
        f"- Batch size: {batch_size}",
        f"- JSON: `{output_json}`",
        "",
        "| Critic | Examples | Examples/sec | ms/example | Total measured sec |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['measured_examples']:,} | "
            f"{item['examples_per_second']:.2f} | {item['ms_per_example']:.2f} | "
            f"{item['total_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These local critics run without API calls. Use the examples/sec numbers to compare against API-based LLM judging latency and cost. The pairwise critic processes a pair as one example, but internally runs two headline encodings.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headlines", type=Path, default=DEFAULT_HEADLINES)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--clickbait-model", type=Path, default=DEFAULT_CLICKBAIT_MODEL)
    parser.add_argument("--quality-model", type=Path, default=DEFAULT_QUALITY_MODEL)
    parser.add_argument("--pairwise-model", type=Path, default=DEFAULT_PAIRWISE_MODEL)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    device = validate_device(args.device)
    df = pd.read_csv(args.headlines)
    pair_rows = load_pair_rows(args.pairs)

    results = []

    click_tokenizer = AutoTokenizer.from_pretrained(args.clickbait_model)
    click_model = AutoModelForSequenceClassification.from_pretrained(args.clickbait_model).to(device).eval()
    click_loader = DataLoader(
        TextDataset(headline_texts(df), click_tokenizer, args.max_length),
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=click_tokenizer),
    )
    results.append(
        benchmark_loader(
            "clickbait_penalty_distilbert",
            click_loader,
            lambda batch: torch.softmax(click_model(**batch).logits, dim=-1)[:, 1],
            device,
            args.warmup_batches,
        )
    )

    quality_tokenizer = AutoTokenizer.from_pretrained(args.quality_model)
    quality_model = DistilBertRewardRegressor.from_pretrained(args.quality_model, attn_implementation="eager").to(device).eval()
    quality_loader = DataLoader(
        TextDataset(quality_texts(df), quality_tokenizer, args.max_length),
        batch_size=args.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=quality_tokenizer),
    )
    results.append(
        benchmark_loader(
            "headline_quality_reward_distilbert",
            quality_loader,
            lambda batch: quality_model(**batch),
            device,
            args.warmup_batches,
        )
    )

    pair_tokenizer = AutoTokenizer.from_pretrained(args.pairwise_model)
    pair_model = DistilBertPairwiseRewardModel.from_pretrained(args.pairwise_model, attn_implementation="eager").to(device).eval()
    pair_loader = DataLoader(
        PairDataset(pair_rows, pair_tokenizer, args.max_length),
        batch_size=args.batch_size,
        collate_fn=PairwiseCollator(pair_tokenizer),
    )
    results.append(
        benchmark_loader(
            "headline_pairwise_reward_distilbert",
            pair_loader,
            lambda batch: pair_model(**batch),
            device,
            args.warmup_batches,
        )
    )

    output = {
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "warmup_batches": args.warmup_batches,
        "benchmarks": results,
    }
    args.json_output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_report(results, args.json_output, args.report, device, args.batch_size)
    print(json.dumps(output, indent=2))
    print("Wrote", args.json_output)
    print("Wrote", args.report)


if __name__ == "__main__":
    main()
