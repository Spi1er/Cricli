#!/usr/bin/env python3
"""Predict clickbait penalty with the fine-tuned DistilBERT classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "clickbait_penalty_distilbert"


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
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()):
            raise RuntimeError("Requested --device mps, but this PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Requested --device mps, but torch.backends.mps.is_available() is False. "
                "Check Python/PyTorch install and macOS MPS support."
            )
        torch.ones(1, device="mps")
    if device not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Unsupported device: {device}")
    return torch.device(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("titles", nargs="*", help="Title strings to score.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--file", type=Path, help="Optional text file with one title per line.")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    titles = list(args.titles)
    if args.file:
        titles.extend(line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not titles:
        raise SystemExit("Provide at least one title or --file.")

    device = validate_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        titles,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    for title, probability in zip(titles, probs):
        probability = float(probability)
        output = {
            "title": title,
            "clickbait_probability": probability,
            "clickbait_penalty": probability,
            "predicted_label": "clickbait" if probability >= 0.5 else "non_clickbait",
        }
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
