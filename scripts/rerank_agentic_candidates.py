#!/usr/bin/env python3
"""Rerank existing agentic candidates with a chosen set of local critics.

This avoids another API generation pass: it takes a candidates CSV, re-scores
the same candidates with local critics, and writes a new selected-title CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_agentic_headline_optimizer import (
    DEFAULT_CLICKBAIT_MODEL,
    DEFAULT_PAIRWISE_MODEL,
    DEFAULT_QUALITY_MODEL,
    score_candidates,
    select_best,
    validate_device,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_candidates_100.csv"
DEFAULT_SELECTED = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_selected_v2_100.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_v2_profile.md"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_v2_metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-candidates", type=Path, default=PROJECT_ROOT / "data" / "processed" / "headline_generation_agentic_candidates_v2_scored_100.csv")
    parser.add_argument("--output-selected", type=Path, default=DEFAULT_SELECTED)
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
    parser.add_argument("--reward-preset", choices=["balanced", "faithfulness_specificity"], default="balanced")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    args.model = "existing_candidates"
    args.num_candidates = None
    args.dry_run = False
    args.prompt_style = "existing_candidates"

    device = validate_device(args.device)
    candidates = pd.read_csv(args.candidates)
    scored = score_candidates(args, candidates, device)
    selected = select_best(scored)

    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    args.output_selected.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_candidates, index=False)
    selected.to_csv(args.output_selected, index=False)
    write_report(args, scored, selected, device)

    metadata = {
        "candidates": str(args.candidates),
        "output_candidates": str(args.output_candidates),
        "output_selected": str(args.output_selected),
        "report": str(args.report),
        "device": str(device),
        "candidate_rows": int(len(scored)),
        "selected_rows": int(len(selected)),
        "clickbait_model": str(args.clickbait_model),
        "quality_model": str(args.quality_model),
        "pairwise_model": str(args.pairwise_model),
        "clickbait_weight": args.clickbait_weight,
        "quality_weight": args.quality_weight,
        "pairwise_weight": args.pairwise_weight,
        "reward_preset": args.reward_preset,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote", args.output_candidates)
    print("Wrote", args.output_selected)
    print("Wrote", args.report)
    print("Wrote", args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
