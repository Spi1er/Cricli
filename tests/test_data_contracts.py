"""Lightweight data-contract tests for tracked Cricli artifacts.

These tests intentionally avoid model weights and API calls. They verify that
GitHub-tracked evaluation artifacts keep the schemas needed by the demo,
analysis scripts, and final presentation.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv_rows(relative_path: str) -> list[dict[str, str]]:
    path = PROJECT_ROOT / relative_path
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class DataContractTests(unittest.TestCase):
    def test_generation_seed_has_expected_schema_and_size(self) -> None:
        rows = read_csv_rows("data/processed/headline_generation_eval_seed_100.csv")
        self.assertEqual(len(rows), 100)
        required = {"seed_id", "category", "summary", "title"}
        self.assertTrue(required.issubset(rows[0].keys()))
        self.assertEqual(len({row["seed_id"] for row in rows}), 100)
        self.assertTrue(all(row["summary"].strip() for row in rows))

    def test_final_llm_judge_scores_cover_all_variants(self) -> None:
        rows = read_csv_rows("data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv")
        required_variants = {"original", "zero_shot", "optimized", "agentic_selected"}
        self.assertEqual(len(rows), 400)
        self.assertEqual({row["variant"] for row in rows}, required_variants)
        self.assertEqual(len({row["seed_id"] for row in rows}), 100)
        for row in rows[:20]:
            for field in ["faithfulness", "clarity", "specificity", "attractiveness", "non_clickbait", "overall"]:
                score = float(row[field])
                self.assertGreaterEqual(score, 1.0)
                self.assertLessEqual(score, 5.0)

    def test_demo_cases_have_recommendations_for_each_objective(self) -> None:
        rows = read_csv_rows("data/processed/headline_review_demo_cases.csv")
        required = {"seed_id", "objective", "headline", "is_recommended", "decision_explanation"}
        self.assertTrue(rows)
        self.assertTrue(required.issubset(rows[0].keys()))
        objectives = {row["objective"] for row in rows}
        self.assertEqual(objectives, {"trust_safety", "growth", "editorial", "specificity"})
        recommended = [row for row in rows if row["is_recommended"].lower() == "true"]
        self.assertTrue(recommended)
        self.assertTrue(all(row["decision_explanation"].strip() for row in recommended))

    def test_persona_votes_have_expected_personas_and_seed_coverage(self) -> None:
        rows = read_csv_rows("data/processed/headline_audience_persona_votes.csv")
        personas = {row["persona"] for row in rows if not row.get("judge_error")}
        expected = {"trust_sensitive_reader", "growth_reader", "busy_news_reader", "editorial_reviewer"}
        self.assertEqual(personas, expected)
        self.assertEqual(len({row["seed_id"] for row in rows if not row.get("judge_error")}), 100)
        required = {"trust", "engagement", "clarity", "audience_fit", "overall"}
        self.assertTrue(required.issubset(rows[0].keys()))


if __name__ == "__main__":
    unittest.main()
