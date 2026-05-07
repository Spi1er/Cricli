#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for LLM-judge comparisons.

This script reads an existing LLM-as-judge score CSV and reports paired
bootstrap intervals for every variant pair on each judge dimension. It does not
train a model; it is an evaluation-summary utility for the final report and
presentation.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE = PROJECT_ROOT / "data" / "processed" / "headline_quality_llm_judge_agentic_v3_specificity_scores.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "bootstrap_significance.md"
DEFAULT_DIMS = ("faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall")


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    iterations: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float, bool]:
    diffs = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample_idx = rng.integers(0, len(diffs), len(diffs))
        means[idx] = float(np.mean(diffs[sample_idx]))
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(diffs)), low, high, (low > 0) or (high < 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired bootstrap significance report.")
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE, help="LLM-judge score CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Markdown report output path.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["zero_shot", "optimized", "agentic_selected", "original"],
        help="Variant order to compare.",
    )
    parser.add_argument("--dims", nargs="+", default=list(DEFAULT_DIMS), help="Judge dimensions to compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.judge)
    seed_col = "seed_id" if "seed_id" in df.columns else "id"
    if seed_col not in df.columns:
        raise ValueError("Judge CSV must contain `seed_id` or `id`.")
    if "variant" not in df.columns:
        raise ValueError("Judge CSV must contain `variant`.")

    dims = [dim for dim in args.dims if dim in df.columns]
    if not dims:
        raise ValueError(f"No requested judge dimensions found in {args.judge}.")

    rows: list[dict[str, object]] = []
    for left, right in combinations(args.variants, 2):
        left_df = df[df["variant"] == left].set_index(seed_col)
        right_df = df[df["variant"] == right].set_index(seed_col)
        joined = left_df[dims].join(right_df[dims], lsuffix="_left", rsuffix="_right", how="inner")
        if joined.empty:
            continue
        for dim in dims:
            a = joined[f"{dim}_left"].to_numpy(dtype=float)
            b = joined[f"{dim}_right"].to_numpy(dtype=float)
            delta, low, high, significant = paired_bootstrap(
                a,
                b,
                iterations=args.bootstrap,
                alpha=args.alpha,
                seed=args.seed,
            )
            rows.append(
                {
                    "variant_a": left,
                    "variant_b": right,
                    "dimension": dim,
                    "n": int(len(a)),
                    "mean_a": round(float(np.mean(a)), 3),
                    "mean_b": round(float(np.mean(b)), 3),
                    "delta_a_minus_b": round(delta, 3),
                    "ci_low": round(low, 3),
                    "ci_high": round(high, 3),
                    "significant": significant,
                }
            )

    out_df = pd.DataFrame(rows)
    lines = [
        "# Bootstrap Significance Report",
        "",
        f"- Judge file: `{args.judge}`",
        f"- Bootstrap iterations: {args.bootstrap}",
        f"- Confidence level: {int((1 - args.alpha) * 100)}%",
        "",
        "Each row reports the paired difference `variant_a - variant_b`; `significant` is true when the interval excludes zero.",
        "",
        out_df.to_markdown(index=False),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
