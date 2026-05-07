#!/usr/bin/env python3
"""Run the product-facing Cricli demo build.

This is the clean demo entrypoint for teammates and project review:

1. Check local assets and dependency state.
2. Build a compact product-facing headline review dataset for the static HTML fallback.
3. Build a full 100-seed case explorer dataset for the Gradio live demo.
4. Render the English static demo.
The research scripts remain available, but this command is the recommended
surface for report/demo preparation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_step(command: list[str], title: str) -> None:
    print(f"\n== {title} ==", flush=True)
    print(" ".join(rel(Path(part)) if part.startswith(str(PROJECT_ROOT)) else part for part in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the product-facing Cricli demo.")
    parser.add_argument("--limit-seeds", type=int, default=10, help="Number of article examples to expose in the main demo.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for dependency checks.")
    parser.add_argument("--skip-asset-check", action="store_true", help="Skip the local asset/dependency check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_asset_check:
        run_step(
            [python, "scripts/check_project_assets.py", "--python", args.python],
            "Check local assets",
        )

    run_step(
        [
            python,
            "scripts/build_headline_review_demo_cases.py",
            "--limit-seeds",
            str(args.limit_seeds),
        ],
        "Build compact HTML demo cases",
    )
    run_step(
        [
            python,
            "scripts/build_headline_review_demo_cases.py",
            "--output",
            "data/processed/headline_review_demo_cases_full.csv",
            "--report",
            "data/processed/headline_review_demo_cases_full_profile.md",
            "--metadata",
            "data/processed/headline_review_demo_cases_full_metadata.json",
        ],
        "Build full Gradio case explorer",
    )
    run_step([python, "scripts/build_headline_review_demo_html.py"], "Build HTML demo")

    print("\nDemo build complete.", flush=True)
    print(f"- Demo: {rel(PROJECT_ROOT / 'demo' / 'headline_review_console.html')}")
    print(f"- Compact cases: {rel(PROJECT_ROOT / 'data' / 'processed' / 'headline_review_demo_cases.csv')}")
    print(f"- Full Gradio cases: {rel(PROJECT_ROOT / 'data' / 'processed' / 'headline_review_demo_cases_full.csv')}")


if __name__ == "__main__":
    main()
