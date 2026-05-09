"""Smoke tests for demo-facing Cricli workflows.

The goal is to catch broken imports, schema drift, parser regressions, and the
fallback review path without requiring model weights or network access.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DemoWorkflowTests(unittest.TestCase):
    def test_live_url_html_extractor_parses_article_like_html(self) -> None:
        from demo.gradio_app_live_url import extract_article_from_html

        raw_html = """
        <html>
          <head>
            <title>Senior Santa Program Delivers Gifts | Local News</title>
            <meta name="description" content="Brevard County volunteers bring holiday cheer to seniors in long-term care facilities through the TRIAD Senior Santa program.">
          </head>
          <body>
            <h1>Senior Santa Program Delivers Gifts</h1>
            <p>Brevard County volunteers collect gifts and deliver them before Christmas to seniors living in long-term care facilities every year.</p>
            <p>The TRIAD Senior Santa program coordinates with local agencies and care homes to identify residents who may not otherwise receive gifts.</p>
          </body>
        </html>
        """
        article = extract_article_from_html(raw_html, "https://example.com/community/senior-santa")
        self.assertEqual(article["title"], "Senior Santa Program Delivers Gifts")
        self.assertEqual(article["domain"], "example.com")
        self.assertGreaterEqual(article["text_block_count"], 2)
        self.assertIn("TRIAD Senior Santa", article["summary"])

    def test_single_article_fallback_scores_and_recommends_one_candidate(self) -> None:
        from scripts import review_single_article as review

        summary = (
            "Brevard County volunteers bring holiday cheer to seniors in long-term care facilities every year "
            "through the TRIAD Senior Santa program, collecting gifts and delivering them before Christmas."
        )
        candidates = review.fallback_candidates(summary, "community", "", 4)
        scored = review.score_candidates(
            summary=summary,
            category="community",
            objective="editorial",
            candidates=candidates,
            clickbait_scorer=None,
            device="unavailable",
            max_length=96,
            generation_mode="heuristic_fallback",
        )
        self.assertEqual(len(scored), 4)
        self.assertEqual(int(scored["is_recommended"].sum()), 1)
        self.assertTrue(scored.iloc[0]["headline"])
        self.assertIn("unified_decision_score", scored.columns)

    def test_asset_check_script_runs_with_temp_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            json_out = tmp / "asset_check.json"
            report_out = tmp / "asset_check.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "check_project_assets.py"),
                    "--python",
                    sys.executable,
                    "--json-output",
                    str(json_out),
                    "--report",
                    str(report_out),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertIn("processed_data", payload)
            self.assertIn("recommended_fixes", payload)
            self.assertTrue(report_out.exists())


if __name__ == "__main__":
    unittest.main()
