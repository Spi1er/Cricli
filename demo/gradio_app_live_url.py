#!/usr/bin/env python3
"""Gradio app for Cricli's headline review workflow.

The app has three product-facing modes:

1. Load curated presentation scenarios for a clean live walkthrough.
2. Review saved evaluation cases from the full product-facing case file.
3. Review a custom article summary by reusing `scripts/review_single_article.py`.

It does not introduce a new model. The custom-summary path uses the same
candidate-generation and scoring functions as the existing single-article
script. If `OPENAI_API_KEY` is not set, that script's deterministic fallback
candidate generator is used.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import review_single_article as review  # noqa: E402


DEMO_CASES = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_cases.csv"
FULL_DEMO_CASES = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_cases_full.csv"
PERSONA_VOTES = PROJECT_ROOT / "data" / "processed" / "headline_audience_persona_votes.csv"

OBJECTIVE_ORDER = ["trust_safety", "growth", "editorial", "specificity"]
OBJECTIVE_LABELS = {
    "trust_safety": "Trust / Safety",
    "growth": "Growth",
    "editorial": "Editorial",
    "specificity": "Specificity",
}
LABEL_TO_OBJECTIVE = {label: key for key, label in OBJECTIVE_LABELS.items()}
SCORE_FIELDS = [
    "quality_score",
    "risk_score",
    "audience_score",
    "objective_fit_score",
    "support_score",
    "unified_decision_score",
]
MAX_URL_FETCH_BYTES = 2_000_000
MAX_EXTRACTED_SUMMARY_CHARS = 1_200
MIN_EXTRACTED_SUMMARY_CHARS = 80
APP_CSS = """
:root,
.gradio-container {
  --body-background-fill: #f6f5ef !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f9faf5 !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: #d9decc !important;
  --border-color-primary: #d9decc !important;
  --border-color-accent: #12756d !important;
  --body-text-color: #18212f !important;
  --body-text-color-subdued: #687487 !important;
  --link-text-color: #12756d !important;
  --button-primary-background-fill: #12756d !important;
  --button-primary-background-fill-hover: #0f6a62 !important;
  --button-primary-text-color: #ffffff !important;
  --button-secondary-background-fill: #ffffff !important;
  --button-secondary-background-fill-hover: #f9faf5 !important;
  --button-secondary-text-color: #2b3444 !important;
  --input-background-fill: #ffffff !important;
  --input-border-color: #d9decc !important;
  --table-border-color: #d9decc !important;
  --code-background-fill: #f9faf5 !important;
  --code-text-color: #18212f !important;
}
html,
body,
body.dark,
gradio-app,
body > gradio-app,
#root,
.main,
.app,
.dark,
.gradio-container {
  background: #f6f5ef !important;
  color: #18212f !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  min-height: 100vh !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
.gradio-container .tabs {
  background: transparent !important;
}
.gradio-container .tabitem,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .accordion {
  border-color: #d9decc !important;
}
.gradio-container .tabitem {
  background: transparent !important;
}
.gradio-container button.primary {
  background: #12756d !important;
  border-color: #12756d !important;
  color: #ffffff !important;
}
.gradio-container button.secondary,
.gradio-container button:not(.primary) {
  border-color: #d9decc !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .input-container {
  background: #ffffff !important;
  border-color: #d9decc !important;
  color: #18212f !important;
}
.gradio-container label,
.gradio-container .label,
.gradio-container .caption {
  color: #687487 !important;
}
.gradio-container table {
  background: #ffffff !important;
}
.gradio-container th {
  background: #f9faf5 !important;
  color: #465266 !important;
}
.gradio-container td {
  background: #ffffff !important;
  color: #18212f !important;
}
.gradio-container code {
  background: #eef3ed !important;
  color: #18212f !important;
  border-radius: 5px !important;
  padding: 1px 5px !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
  color: #12756d !important;
  border-color: #12756d !important;
}
.cricli-table-wrap {
  overflow-x: auto;
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #ffffff;
  margin-bottom: 10px;
}
.cricli-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border: 0 !important;
  font-size: 0.92rem;
}
.cricli-table th {
  background: #f9faf5;
  color: #465266;
  text-align: left;
  font-weight: 760;
  border: 0 !important;
  border-bottom: 1px solid #d9decc !important;
  padding: 9px 10px;
  white-space: nowrap;
}
.cricli-table td {
  background: #ffffff;
  color: #18212f;
  border: 0 !important;
  border-bottom: 1px solid #edf0e6 !important;
  padding: 9px 10px;
  vertical-align: top;
}
.cricli-table tr:last-child td {
  border-bottom: 0;
}
.cricli-table .numeric {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  color: #315fba;
  font-weight: 650;
}
.cricli-table-empty {
  border: 1px dashed #d9decc;
  border-radius: 8px;
  color: #687487;
  padding: 14px;
  background: #fbfcf9;
}
.cricli-section-label {
  color: #687487;
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0;
  margin: 12px 0 6px;
  text-transform: uppercase;
}
.cricli-field label,
.cricli-field label span,
.cricli-field .label-wrap,
.cricli-field .label-wrap span,
.cricli-field [data-testid="block-label"],
.cricli-field [data-testid="block-label"] span,
.cricli-field .wrap > span,
.cricli-field .block-label,
.cricli-field .container > span,
.cricli-field span.svelte-jdcl7l,
.gradio-container span.svelte-jdcl7l,
.gradio-container span.svelte-e5lyqv {
  color: #687487 !important;
  font-size: 0.78rem !important;
  font-weight: 760 !important;
  letter-spacing: 0 !important;
  text-transform: uppercase !important;
  opacity: 1 !important;
}
.cricli-field:focus-within,
.cricli-field *:focus,
.cricli-field *:focus-visible {
  outline-color: #12756d !important;
}
.cricli-hero {
  border: 1px solid #d9decc;
  border-left: 5px solid #12756d;
  border-radius: 8px;
  padding: 18px 20px;
  margin-bottom: 12px;
  background: #ffffff;
  box-shadow: 0 10px 30px rgba(16, 24, 40, 0.06);
}
.cricli-hero h1 {
  margin: 0 0 6px;
  color: #18212f;
}
.cricli-muted { color: #687487; margin: 0; }
.cricli-note {
  color: #687487;
  font-size: 0.92rem;
  margin: 4px 0 12px;
}
.cricli-note-card {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #fbfcf9;
  color: #465266;
  padding: 14px 18px;
  margin: 10px 0 14px;
}
.cricli-tight h3 { margin-top: 0; }
.cricli-tight {
  background: #fbfcf9;
  border: 1px solid #d9decc;
  border-radius: 8px;
  padding: 12px 14px;
}
.cricli-rec-card {
  border: 1px solid #d9decc;
  border-left: 5px solid #12756d;
  border-radius: 8px;
  background: #ffffff;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(16, 24, 40, 0.06);
}
.cricli-rec-eyebrow {
  color: #687487;
  font-size: 0.78rem;
  font-weight: 760;
  text-transform: uppercase;
  margin-bottom: 7px;
}
.cricli-rec-title {
  color: #18212f;
  font-size: 1.65rem;
  font-weight: 780;
  line-height: 1.12;
  margin-bottom: 10px;
}
.cricli-rec-reason {
  color: #3b4657;
  line-height: 1.48;
  margin-bottom: 14px;
}
.cricli-score-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 10px;
}
.cricli-score-grid.compact {
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}
.cricli-score-box {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #fbfcf9;
  padding: 9px 10px;
}
.cricli-score-top {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: #687487;
  font-size: 0.78rem;
  margin-bottom: 8px;
}
.cricli-score-top b {
  color: #59667a;
}
.cricli-bar {
  height: 7px;
  background: #e4e8dd;
  border-radius: 999px;
  overflow: hidden;
}
.cricli-fill {
  height: 100%;
  border-radius: 999px;
  background: #12756d;
}
.cricli-fill.risk { background: #315fba; }
.cricli-fill.audience { background: #b7791f; }
.cricli-fill.objective { background: #be3a51; }
.cricli-fill.support { background: #6b7280; }
.cricli-objective-grid,
.cricli-source-grid,
.cricli-candidate-grid {
  display: grid;
  gap: 12px;
}
.cricli-objective-grid {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.cricli-source-grid {
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}
.cricli-candidate-grid {
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.cricli-objective-card,
.cricli-source-card,
.cricli-candidate-card {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px;
}
.cricli-candidate-card.selected {
  border-color: #12756d;
  box-shadow: 0 0 0 2px #dff1ed;
}
.cricli-card-top {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: #687487;
  font-size: 0.78rem;
  margin-bottom: 8px;
}
.cricli-card-top b,
.cricli-candidate-card .cricli-card-title,
.cricli-source-card .cricli-card-title,
.cricli-objective-card .cricli-card-title {
  color: #18212f !important;
}
.cricli-gradio-section {
  color: #687487;
  font-size: 0.78rem;
  font-weight: 760;
  margin: 14px 0 8px;
  text-transform: uppercase;
}
.gradio-container .accordion,
.gradio-container details {
  background: #ffffff !important;
  border-color: #d9decc !important;
}
.gradio-container summary,
.gradio-container .accordion span {
  color: #465266 !important;
}
.gradio-container .accordion,
.gradio-container details,
.gradio-container .accordion label,
.gradio-container .accordion summary,
.gradio-container .accordion .label-wrap,
.gradio-container .accordion .label-wrap span,
.gradio-container .accordion [data-testid="block-label"],
.gradio-container details summary,
.gradio-container details summary * {
  color: #18212f !important;
  opacity: 1 !important;
}
.gradio-container .accordion svg,
.gradio-container details summary svg {
  color: #687487 !important;
  opacity: 1 !important;
}
.cricli-card-title {
  color: #18212f;
  font-size: 1rem;
  font-weight: 760;
  line-height: 1.28;
  margin-bottom: 8px;
}
.cricli-card-note {
  border-left: 3px solid #d9decc;
  color: #465266;
  font-size: 0.86rem;
  line-height: 1.42;
  padding-left: 9px;
  margin-bottom: 10px;
}
.cricli-candidate-card.selected .cricli-card-note {
  border-left-color: #12756d;
}
.cricli-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.cricli-chip {
  border: 1px solid #d9decc;
  border-radius: 7px;
  background: #fbfcf9;
  color: #59667a;
  font-size: 0.74rem;
  padding: 4px 7px;
}
.cricli-chip.selected {
  border-color: #9dd1c8;
  background: #dff1ed;
  color: #12756d;
}
.cricli-sidebar,
.cricli-mainpane {
  gap: 14px !important;
}
.cricli-sidebar .cricli-source-grid,
.cricli-sidebar .cricli-objective-grid {
  grid-template-columns: 1fr;
}
.cricli-mainpane .cricli-candidate-grid {
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.cricli-persona-accordion,
.cricli-persona-accordion .accordion,
.cricli-persona-accordion details {
  border-color: #d9decc !important;
  box-shadow: none !important;
}
.cricli-persona-accordion,
.cricli-persona-accordion label,
.cricli-persona-accordion summary,
.cricli-persona-accordion .label-wrap,
.cricli-persona-accordion .label-wrap span,
.cricli-persona-accordion [data-testid="block-label"],
.cricli-persona-accordion [data-testid="block-label"] span,
.cricli-persona-accordion details summary,
.cricli-persona-accordion details summary * {
  color: #687487 !important;
  font-size: 0.78rem !important;
  font-weight: 760 !important;
  letter-spacing: 0 !important;
  text-transform: uppercase !important;
  opacity: 1 !important;
}
.cricli-persona-accordion:focus,
.cricli-persona-accordion:focus-within,
.cricli-persona-accordion *:focus,
.cricli-persona-accordion *:focus-visible {
  border-color: #d9decc !important;
  box-shadow: none !important;
  outline: 0 !important;
}
.cricli-persona-accordion .cricli-table-wrap {
  border-color: #d9decc !important;
}
.cricli-persona-accordion .cricli-table th,
.cricli-persona-accordion .cricli-table td {
  border-color: #edf0e6 !important;
}
.cricli-html-shell {
  gap: 0 !important;
  align-items: stretch !important;
  min-height: 100vh;
}
.cricli-html-sidebar {
  background: #fcfcf7;
  border-right: 1px solid #d9decc;
  padding: 18px 16px;
  min-height: 100vh;
}
.cricli-html-sidebar > .form,
.cricli-html-sidebar .block {
  background: transparent !important;
}
.cricli-html-main {
  background: #f6f5ef;
  padding: 22px clamp(18px, 3vw, 42px) 42px;
  min-height: 100vh;
}
.cricli-html-brand-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.cricli-html-brand-title {
  color: #18212f;
  font-size: 19px;
  font-weight: 760;
  line-height: 1.08;
}
.cricli-html-brand-pill,
.cricli-html-chip {
  border: 1px solid #d9decc;
  border-radius: 7px;
  background: #fbfcf9;
  color: #59667a;
  font-size: 0.74rem;
  padding: 4px 7px;
  white-space: nowrap;
}
.cricli-html-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
.cricli-html-stat {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #ffffff;
  padding: 9px;
}
.cricli-html-stat b {
  display: block;
  color: #18212f;
  font-size: 18px;
  line-height: 1;
}
.cricli-html-stat span {
  display: block;
  color: #687487;
  font-size: 11px;
  margin-top: 3px;
}
.cricli-html-scenario-panel,
.cricli-html-current-card {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #f9faf5 !important;
  padding: 10px;
}
.cricli-html-scenario-panel,
.cricli-html-scenario-panel *,
.cricli-html-scenario-panel > div,
.cricli-html-scenario-panel .form {
  background-color: #f9faf5 !important;
}
.cricli-html-scenario-panel input,
.cricli-html-scenario-panel select,
.cricli-html-scenario-panel .wrap,
.cricli-html-scenario-panel .input-container {
  background-color: #ffffff !important;
}
.cricli-html-current-card {
  background: #ffffff !important;
}
.cricli-html-article-top {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 6px;
}
.cricli-html-category {
  color: #12756d;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.cricli-html-seed {
  color: #687487;
  font-size: 12px;
}
.cricli-html-snippet {
  color: #394456;
  font-size: 13px;
  line-height: 1.35;
  margin: 0;
}
.cricli-html-page-title {
  color: #18212f;
  font-size: clamp(25px, 3vw, 38px);
  line-height: 1.04;
  margin: 0 0 14px;
}
.cricli-html-scenario-banner {
  border: 1px solid #c7d2fe;
  border-left: 5px solid #315fba;
  border-radius: 8px;
  background: #f4f7ff;
  padding: 12px 14px;
  margin-bottom: 16px;
}
.cricli-html-scenario-banner b {
  display: block;
  color: #18212f;
  margin-bottom: 4px;
}
.cricli-html-scenario-banner span {
  color: #44516a;
  line-height: 1.45;
}
.cricli-html-objective-tabs {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0;
}
.cricli-html-objective-tab {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #ffffff;
  color: #2b3444;
  font-weight: 650;
  padding: 9px 12px;
}
.cricli-html-objective-tab.active {
  background: #18212f;
  border-color: #18212f;
  color: #ffffff;
}
.cricli-objective-radio {
  margin: 16px 0 !important;
}
.cricli-objective-radio,
.cricli-objective-radio .wrap,
.cricli-objective-radio .block,
.cricli-objective-radio .form {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.cricli-objective-radio .wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}
.cricli-objective-radio label {
  align-items: center !important;
  background: #ffffff !important;
  border: 1px solid #d9decc !important;
  border-radius: 8px !important;
  color: #2b3444 !important;
  cursor: pointer !important;
  display: inline-flex !important;
  font-weight: 650 !important;
  margin: 0 !important;
  min-height: 38px !important;
  padding: 9px 12px !important;
  width: auto !important;
}
.cricli-objective-radio label span {
  color: #2b3444 !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  text-transform: none !important;
}
.cricli-objective-radio input[type="radio"] {
  display: none !important;
}
.cricli-objective-radio label:has(input[type="radio"]:checked) {
  background: #18212f !important;
  border-color: #18212f !important;
  color: #ffffff !important;
}
.cricli-objective-radio label:has(input[type="radio"]:checked) span {
  color: #ffffff !important;
}
.cricli-html-article-context {
  border-top: 1px solid #d9decc;
  border-bottom: 1px solid #d9decc;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 260px;
  gap: 18px;
  margin-bottom: 20px;
  padding: 17px 0;
}
.cricli-html-summary-text {
  color: #323d4d;
  font-size: 16px;
  line-height: 1.55;
  margin: 7px 0 0;
}
.cricli-html-metric-stack {
  display: grid;
  gap: 10px;
}
.cricli-html-metric {
  border: 1px solid #d9decc;
  border-radius: 8px;
  background: #ffffff;
  padding: 10px 12px;
}
.cricli-html-metric span {
  color: #687487;
  display: block;
  font-size: 12px;
}
.cricli-html-metric b {
  color: #18212f;
  display: block;
  font-size: 20px;
  margin-top: 2px;
}
.cricli-html-section {
  color: #687487;
  font-size: 0.78rem;
  font-weight: 760;
  margin: 16px 0 8px;
  text-transform: uppercase;
}
.cricli-html-main .cricli-rec-card {
  margin-bottom: 16px;
}
.cricli-html-main .cricli-candidate-grid {
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
}
.cricli-html-main .cricli-source-grid {
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
@media (max-width: 980px) {
  .cricli-html-shell {
    display: block !important;
  }
  .cricli-html-sidebar {
    border-right: 0;
    border-bottom: 1px solid #d9decc;
    min-height: auto;
  }
  .cricli-html-article-context {
    grid-template-columns: 1fr;
  }
  .cricli-html-main .cricli-candidate-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 760px) {
  .cricli-score-grid {
    grid-template-columns: 1fr;
  }
}
"""

SCENARIOS = {
    "Objective Changes Recommendation": {
        "seed_id": 10,
        "objective": "growth",
        "user": "Content editor preparing a sports headline.",
        "business_question": "Should the team prioritize editorial balance, growth, trust, or specificity?",
        "takeaway": "There is no universal best headline. The best choice depends on the publishing objective.",
    },
    "Trust / Safety Prefers Conservative Editorial": {
        "seed_id": 2,
        "objective": "trust_safety",
        "user": "Editor reviewing a food and lifestyle headline.",
        "business_question": "Should a strong GenAI title be used directly, or should the team keep a safer editorial title?",
        "takeaway": "GenAI can be excellent, but trust-sensitive surfaces may still prefer a conservative editorial baseline.",
    },
    "Specificity Selects Concrete Alternative": {
        "seed_id": 9,
        "objective": "specificity",
        "user": "News editor checking whether the title is specific enough.",
        "business_question": "Can the selector move away from the default best baseline when the objective changes?",
        "takeaway": "Objective-specific selection can choose a more concrete candidate when specificity is the explicit goal.",
    },
    "Validated GenAI Baseline": {
        "seed_id": 1,
        "objective": "editorial",
        "user": "Editor reviewing whether a direct GenAI title is ready to publish.",
        "business_question": "Can Cricli confirm when the GenAI baseline is already the best tradeoff?",
        "takeaway": "Cricli is not designed to reject GenAI by default. It can validate a strong GenAI headline when the scores and persona signals support it.",
    },
    "SFT Adds Specificity": {
        "seed_id": 14,
        "objective": "specificity",
        "user": "Finance editor checking whether the headline contains enough concrete detail.",
        "business_question": "Can an SFT candidate add value when the objective emphasizes specificity?",
        "takeaway": "The SFT generator is not the main product claim, but it can contribute useful candidates to the review pool under specific objectives.",
    },
}


def import_gradio():
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Gradio is not installed. Install it with `pip install gradio` or `pip install -r requirements-demo.txt`."
        ) from exc
    return gr


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return " ".join(str(value).split())


class ArticleHTMLExtractor(HTMLParser):
    """Small article extractor for the live demo path.

    This keeps the demo self-contained. It is intentionally conservative: the
    pipeline only needs a useful summary seed, not a full production crawler.
    """

    TEXT_TAGS = {"p", "h1", "h2", "h3", "li"}
    IGNORED_TAGS = {"script", "style", "noscript", "svg", "form", "footer", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta_title = ""
        self.meta_description = ""
        self.page_title = ""
        self.h1_candidates: list[str] = []
        self.text_blocks: list[str] = []
        self._ignore_depth = 0
        self._active_tag: str | None = None
        self._active_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if tag == "meta":
            attrs_dict = {str(key).lower(): value or "" for key, value in attrs if key}
            key = clean_text(attrs_dict.get("property") or attrs_dict.get("name")).lower()
            content = clean_text(attrs_dict.get("content"))
            if content and key in {"og:title", "twitter:title"} and not self.meta_title:
                self.meta_title = content
            if content and key in {"description", "og:description", "twitter:description"} and not self.meta_description:
                self.meta_description = content
            return
        if tag == "title" or tag in self.TEXT_TAGS:
            self._active_tag = tag
            self._active_parts = []

    def handle_data(self, data: str) -> None:
        if self._ignore_depth or not self._active_tag:
            return
        text = clean_text(data)
        if text:
            self._active_parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.IGNORED_TAGS and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if tag != self._active_tag:
            return
        text = clean_text(" ".join(self._active_parts))
        if tag == "title" and text and not self.page_title:
            self.page_title = text
        elif tag == "h1" and len(text) >= 12:
            self.h1_candidates.append(text)
        elif tag in self.TEXT_TAGS and len(text) >= 35:
            self.text_blocks.append(text)
        self._active_tag = None
        self._active_parts = []


def _dedupe_text_blocks(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = clean_text(value)
        key = re.sub(r"\W+", "", text.lower())[:160]
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _clean_page_title(value: str) -> str:
    title = clean_text(value)
    for separator in (" | ", " - "):
        if separator in title:
            first = clean_text(title.split(separator)[0])
            if len(first) >= 18:
                return first
    return title


def fetch_url_html(url: str) -> tuple[str, str]:
    cleaned_url = clean_text(url)
    if cleaned_url and "://" not in cleaned_url:
        cleaned_url = f"https://{cleaned_url}"
    parsed = urlparse(cleaned_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http(s) article URL.")
    request = Request(
        parsed.geturl(),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(MAX_URL_FETCH_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            final_url = response.geturl()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching article.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch article: {exc.reason}.") from exc
    return final_url, raw.decode(charset, errors="replace")


def extract_article_from_html(raw_html: str, source_url: str) -> dict[str, Any]:
    extractor = ArticleHTMLExtractor()
    extractor.feed(raw_html)
    text_blocks = _dedupe_text_blocks(extractor.text_blocks)
    title = _clean_page_title(extractor.meta_title or (extractor.h1_candidates[0] if extractor.h1_candidates else "") or extractor.page_title)
    description = clean_text(extractor.meta_description)

    summary_parts = []
    if description and len(description) >= 45:
        summary_parts.append(description)
    summary_parts.extend(text_blocks)

    summary = ""
    for part in _dedupe_text_blocks(summary_parts):
        candidate = clean_text(f"{summary} {part}")
        if len(candidate) > MAX_EXTRACTED_SUMMARY_CHARS:
            remaining = MAX_EXTRACTED_SUMMARY_CHARS - len(summary) - 1
            if remaining > 80:
                summary = clean_text(f"{summary} {part[:remaining]}")
            break
        summary = candidate

    if len(summary) < MIN_EXTRACTED_SUMMARY_CHARS:
        raise ValueError("Could not extract enough article text. Paste a short summary instead.")

    domain = urlparse(source_url).netloc.replace("www.", "")
    return {
        "title": title,
        "summary": summary,
        "domain": domain,
        "text_block_count": len(text_blocks),
        "source_url": source_url,
    }


def fetch_article_url_ui(
    url: str,
    current_summary: str,
    current_title: str,
    current_category: str,
) -> tuple[str, str, str, str]:
    url = clean_text(url)
    if not url:
        return current_summary, current_title, current_category, "Enter a URL to fetch, or paste a summary directly."
    try:
        final_url, raw_html = fetch_url_html(url)
        article = extract_article_from_html(raw_html, final_url)
    except (ValueError, RuntimeError) as exc:
        return current_summary, current_title, current_category, f"Fetch failed: {exc}"

    title = clean_text(article.get("title")) or clean_text(current_title)
    category = clean_text(current_category) or "news"
    status = (
        f"Fetched from `{article['domain']}`. "
        f"Extracted {article['text_block_count']} text blocks and filled the review fields."
    )
    return article["summary"], title, category, status


def score_label(value: object) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def table_html(df: pd.DataFrame, empty_text: str = "No rows to display.") -> str:
    if df is None or df.empty:
        return f'<div class="cricli-table-empty">{html.escape(empty_text)}</div>'
    parts = ['<div class="cricli-table-wrap"><table class="cricli-table"><thead><tr>']
    for column in df.columns:
        parts.append(f"<th>{html.escape(str(column))}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for value in row:
            text_value = "" if pd.isna(value) else str(value)
            numeric_class = " numeric" if isinstance(value, (int, float)) or _looks_numeric(text_value) else ""
            parts.append(f'<td class="{numeric_class.strip()}">{html.escape(text_value)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def labeled_table_html(label: str, df: pd.DataFrame, empty_text: str = "No rows to display.") -> str:
    return f'<div class="cricli-section-label">{html.escape(label)}</div>{table_html(df, empty_text)}'


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def numeric_value(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def score_box_html(label: str, value: object, class_name: str = "") -> str:
    score = numeric_value(value)
    pct = max(0.0, min(100.0, score * 100.0))
    escaped_label = html.escape(label)
    escaped_value = html.escape(score_label(value))
    fill_class = f"cricli-fill {class_name}".strip()
    return (
        '<div class="cricli-score-box">'
        '<div class="cricli-score-top">'
        f"<span>{escaped_label}</span><b>{escaped_value}</b>"
        "</div>"
        '<div class="cricli-bar">'
        f'<div class="{fill_class}" style="width: {pct:.0f}%"></div>'
        "</div>"
        "</div>"
    )


def recommendation_card_html(row: pd.Series, heading: str = "Recommended Headline") -> str:
    explanation = clean_text(row.get("decision_explanation", row.get("recommendation_summary", "")))
    cards = "".join(
        [
            score_box_html("Quality", row.get("quality_score")),
            score_box_html("Risk safety", row.get("risk_score"), "risk"),
            score_box_html("Audience fit", row.get("audience_score"), "audience"),
            score_box_html("Objective fit", row.get("objective_fit_score"), "objective"),
        ]
    )
    return (
        '<div class="cricli-rec-card">'
        f'<div class="cricli-rec-eyebrow">{html.escape(heading)}</div>'
        f'<div class="cricli-rec-title">{html.escape(clean_text(row.get("headline")))}</div>'
        f'<div class="cricli-rec-reason">{html.escape(explanation)}</div>'
        f'<div class="cricli-score-grid">{cards}</div>'
        "</div>"
    )


def score_summary_cards_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return table_html(pd.DataFrame())
    class_by_score = {
        "Risk Score": "risk",
        "Audience Score": "audience",
        "Objective Fit Score": "objective",
        "Support Score": "support",
    }
    cards = []
    for row in df.itertuples(index=False):
        label = str(getattr(row, "Score"))
        value = getattr(row, "Value")
        cards.append(score_box_html(label.replace(" Score", ""), value, class_by_score.get(label, "")))
    return '<div class="cricli-score-grid compact">' + "".join(cards) + "</div>"


def objective_cards_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return table_html(pd.DataFrame())
    cards = []
    for _, row in df.iterrows():
        objective = html.escape(str(row.get("Objective", "")))
        headline = html.escape(str(row.get("Recommended Headline", "")))
        source = html.escape(str(row.get("Source", "")))
        score = html.escape(str(row.get("Decision Score", "")))
        cards.append(
            '<div class="cricli-objective-card">'
            f'<div class="cricli-card-top"><b>{objective}</b><span>{score}</span></div>'
            f'<div class="cricli-card-title">{headline}</div>'
            f'<div class="cricli-chip-row"><span class="cricli-chip">{source}</span></div>'
            "</div>"
        )
    return '<div class="cricli-objective-grid">' + "".join(cards) + "</div>"


def source_pool_cards_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return table_html(pd.DataFrame())
    cards = []
    for _, row in df.iterrows():
        source = html.escape(str(row.get("Candidate Source", "")))
        origin = html.escape(str(row.get("Generator / Origin", "")))
        roles = html.escape(str(row.get("Visible Roles", "")))
        recommended = html.escape(str(row.get("Recommended Under", "")))
        cards.append(
            '<div class="cricli-source-card">'
            f'<div class="cricli-card-top"><b>{source}</b><span>{recommended}</span></div>'
            f'<div class="cricli-card-note">{origin}</div>'
            f'<div class="cricli-chip-row"><span class="cricli-chip">{roles}</span></div>'
            "</div>"
        )
    return '<div class="cricli-source-grid">' + "".join(cards) + "</div>"


def candidate_cards_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return table_html(pd.DataFrame())
    cards = []
    for _, row in df.iterrows():
        role = str(row.get("Role", ""))
        source = str(row.get("Source", ""))
        headline = str(row.get("Headline", ""))
        explanation = str(row.get("Explanation", ""))
        selected = role == "Recommended"
        card_class = "cricli-candidate-card selected" if selected else "cricli-candidate-card"
        score_cards = "".join(
            [
                score_box_html("Quality", row.get("Quality")),
                score_box_html("Risk safety", row.get("Risk/Safety"), "risk"),
                score_box_html("Audience fit", row.get("Audience"), "audience"),
                score_box_html("Objective fit", row.get("Objective Fit"), "objective"),
            ]
        )
        cards.append(
            f'<div class="{card_class}">'
            f'<div class="cricli-card-top"><b>{html.escape(role)}</b><span>{html.escape(source)}</span></div>'
            f'<div class="cricli-card-title">{html.escape(headline)}</div>'
            f'<div class="cricli-card-note">{html.escape(explanation)}</div>'
            f'<div class="cricli-score-grid compact">{score_cards}</div>'
            '<div class="cricli-chip-row">'
            + (('<span class="cricli-chip selected">Selected</span>') if selected else "")
            + f'<span class="cricli-chip">Decision {html.escape(str(row.get("Decision", "")))}</span>'
            + f'<span class="cricli-chip">Clickbait {html.escape(str(row.get("Clickbait Penalty", "")))}</span>'
            + "</div>"
            "</div>"
        )
    return '<div class="cricli-candidate-grid">' + "".join(cards) + "</div>"


def scenario_notes_html(name: str) -> str:
    scenario = SCENARIOS[name]
    objective = OBJECTIVE_LABELS[scenario["objective"]]
    return (
        '<div class="cricli-tight">'
        f'<div class="cricli-card-title">{html.escape(name)}</div>'
        '<div class="cricli-chip-row">'
        f'<span class="cricli-chip">Seed {int(scenario["seed_id"])}</span>'
        f'<span class="cricli-chip">{html.escape(objective)}</span>'
        "</div>"
        f'<div class="cricli-card-note"><b>User:</b> {html.escape(scenario["user"])}</div>'
        f'<div class="cricli-card-note"><b>Business question:</b> {html.escape(scenario["business_question"])}</div>'
        f'<div class="cricli-card-note"><b>Presenter takeaway:</b> {html.escape(scenario["takeaway"])}</div>'
        "</div>"
    )


def load_demo_cases() -> pd.DataFrame:
    source = FULL_DEMO_CASES if FULL_DEMO_CASES.exists() else DEMO_CASES
    if not source.exists():
        raise FileNotFoundError(f"Missing demo cases: {source}")
    df = pd.read_csv(source)
    df["is_recommended"] = df["is_recommended"].astype(bool)
    return df


def load_persona_votes() -> pd.DataFrame:
    if not PERSONA_VOTES.exists():
        return pd.DataFrame()
    return pd.read_csv(PERSONA_VOTES)


def article_choices(df: pd.DataFrame) -> list[str]:
    choices = []
    for seed_id, group in df.groupby("seed_id", sort=True):
        first = group.iloc[0]
        summary = clean_text(first["summary"])
        choices.append(f"{int(seed_id)} | {clean_text(first['category'])} | {summary[:90]}")
    return choices


def seed_from_choice(choice: str) -> int:
    return int(str(choice).split("|", 1)[0].strip())


def objective_key(label_or_key: str) -> str:
    return LABEL_TO_OBJECTIVE.get(label_or_key, label_or_key)


def recommended_markdown(row: pd.Series, heading: str = "Recommended Headline") -> str:
    score_lines = []
    for field in SCORE_FIELDS:
        if field in row:
            score_lines.append(f"- {field.replace('_', ' ').title()}: {score_label(row[field])}")
    explanation = clean_text(row.get("decision_explanation", row.get("recommendation_summary", "")))
    return "\n".join(
        [
            f"### {heading}",
            f"## {clean_text(row['headline'])}",
            "",
            f"**Source:** {clean_text(row.get('variant', row.get('source', '')))}",
            "",
            explanation,
            "",
            *score_lines,
        ]
    )


def candidate_table(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "display_label",
        "headline",
        "variant",
        "quality_score",
        "risk_score",
        "clickbait_penalty",
        "audience_score",
        "objective_fit_score",
        "unified_decision_score",
        "decision_explanation",
    ]
    existing = [column for column in columns if column in rows.columns]
    out = rows[existing].copy()
    out = out.rename(
        columns={
            "display_label": "Role",
            "headline": "Headline",
            "variant": "Source",
            "quality_score": "Quality",
            "risk_score": "Risk/Safety",
            "clickbait_penalty": "Clickbait Penalty",
            "audience_score": "Audience",
            "objective_fit_score": "Objective Fit",
            "unified_decision_score": "Decision",
            "decision_explanation": "Explanation",
        }
    )
    return out


def score_breakdown(row: pd.Series) -> pd.DataFrame:
    rows = []
    for field in SCORE_FIELDS:
        if field in row:
            rows.append({"Score": field.replace("_", " ").title(), "Value": score_label(row[field])})
    if "clickbait_penalty" in row:
        rows.append({"Score": "Clickbait Penalty", "Value": score_label(row["clickbait_penalty"])})
    if "persona_adjustment" in row:
        rows.append({"Score": "Persona Adjustment", "Value": score_label(row["persona_adjustment"])})
    return pd.DataFrame(rows)


def source_pool_table(df: pd.DataFrame, seed_id: int) -> pd.DataFrame:
    sub = df[df["seed_id"].astype(int) == int(seed_id)].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in sub.groupby("variant", dropna=False):
        source = clean_text(group["candidate_source"].dropna().iloc[0]) if group["candidate_source"].notna().any() else ""
        roles = sorted({clean_text(value) for value in group.get("display_label", pd.Series(dtype=str)).dropna()})
        objectives = sorted({OBJECTIVE_LABELS.get(clean_text(value), clean_text(value)) for value in group["objective"].dropna()})
        recommended_objectives = sorted(
            {
                OBJECTIVE_LABELS.get(clean_text(value), clean_text(value))
                for value in group[group["is_recommended"].astype(bool)]["objective"].dropna()
            }
        )
        rows.append(
            {
                "Candidate Source": clean_text(variant),
                "Generator / Origin": source,
                "Visible Roles": ", ".join(roles),
                "Objectives Shown": ", ".join(objectives),
                "Recommended Under": ", ".join(recommended_objectives) if recommended_objectives else "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["Recommended Under", "Candidate Source"], ascending=[False, True])


def persona_table(persona_df: pd.DataFrame, seed_id: int) -> pd.DataFrame:
    if persona_df.empty or "seed_id" not in persona_df.columns:
        return pd.DataFrame()
    sub = persona_df[persona_df["seed_id"].astype(int) == int(seed_id)].copy()
    columns = [
        "persona",
        "variant",
        "headline",
        "trust",
        "engagement",
        "clarity",
        "audience_fit",
        "overall",
        "is_best",
        "rationale",
    ]
    existing = [column for column in columns if column in sub.columns]
    return sub[existing]


def objective_preview(df: pd.DataFrame, seed_id: int) -> pd.DataFrame:
    rows = []
    for objective in OBJECTIVE_ORDER:
        sub = df[(df["seed_id"].astype(int) == int(seed_id)) & (df["objective"] == objective) & (df["is_recommended"])]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Objective": OBJECTIVE_LABELS.get(objective, objective),
                "Recommended Headline": clean_text(row["headline"]),
                "Source": clean_text(row["variant"]),
                "Decision Score": score_label(row["unified_decision_score"]),
            }
        )
    return pd.DataFrame(rows)


def scenario_notes(name: str) -> str:
    scenario = SCENARIOS[name]
    objective = OBJECTIVE_LABELS[scenario["objective"]]
    return "\n".join(
        [
            f"### {name}",
            f"**Demo seed:** `{scenario['seed_id']}`",
            f"**Recommended objective to show first:** `{objective}`",
            "",
            f"**User:** {scenario['user']}",
            "",
            f"**Business question:** {scenario['business_question']}",
            "",
            f"**Presenter takeaway:** {scenario['takeaway']}",
            "",
            "Use the objective preview table to show how the recommendation changes across business goals.",
        ]
    )


def review_scenario(name: str) -> tuple[str, pd.DataFrame, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_demo_cases()
    persona_df = load_persona_votes()
    scenario = SCENARIOS[name]
    seed_id = int(scenario["seed_id"])
    objective = scenario["objective"]
    rows = df[(df["seed_id"].astype(int) == seed_id) & (df["objective"] == objective)].copy()
    if rows.empty:
        empty = pd.DataFrame()
        return scenario_notes(name), empty, "No saved case found.", empty, empty, empty, empty
    rows = rows.sort_values(["is_recommended", "display_order"], ascending=[False, True])
    recommended = rows[rows["is_recommended"]].iloc[0] if rows["is_recommended"].any() else rows.iloc[0]
    return (
        scenario_notes(name),
        objective_preview(df, seed_id),
        recommended_markdown(recommended, heading="Scenario Recommendation"),
        score_breakdown(recommended),
        source_pool_table(df, seed_id),
        candidate_table(rows.sort_values("display_order")),
        persona_table(persona_df, seed_id),
    )


def review_scenario_ui(name: str) -> tuple[str, str, str, str, str, str, str]:
    df = load_demo_cases()
    persona_df = load_persona_votes()
    scenario = SCENARIOS[name]
    seed_id = int(scenario["seed_id"])
    objective = scenario["objective"]
    rows = df[(df["seed_id"].astype(int) == seed_id) & (df["objective"] == objective)].copy()
    if rows.empty:
        empty = pd.DataFrame()
        return scenario_notes_html(name), table_html(empty), "No saved case found.", table_html(empty), table_html(empty), table_html(empty), table_html(empty)
    rows = rows.sort_values(["is_recommended", "display_order"], ascending=[False, True])
    recommended = rows[rows["is_recommended"]].iloc[0] if rows["is_recommended"].any() else rows.iloc[0]
    return (
        scenario_notes_html(name),
        objective_cards_html(objective_preview(df, seed_id)),
        recommendation_card_html(recommended, heading="Recommended Headline"),
        score_summary_cards_html(score_breakdown(recommended)),
        source_pool_cards_html(source_pool_table(df, seed_id)),
        candidate_cards_html(candidate_table(rows.sort_values("display_order"))),
        table_html(persona_table(persona_df, seed_id)),
    )


def review_saved_case(article_choice: str, objective_label: str) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_demo_cases()
    persona_df = load_persona_votes()
    seed_id = seed_from_choice(article_choice)
    objective = objective_key(objective_label)
    rows = df[(df["seed_id"].astype(int) == seed_id) & (df["objective"] == objective)].copy()
    if rows.empty:
        empty = pd.DataFrame()
        return "No saved case found.", empty, empty, empty, empty, empty
    rows = rows.sort_values(["is_recommended", "display_order"], ascending=[False, True])
    recommended = rows[rows["is_recommended"]].iloc[0] if rows["is_recommended"].any() else rows.iloc[0]
    return (
        recommended_markdown(recommended),
        score_breakdown(recommended),
        source_pool_table(df, seed_id),
        candidate_table(rows.sort_values("display_order")),
        persona_table(persona_df, seed_id),
        objective_preview(df, seed_id),
    )


def review_saved_case_ui(article_choice: str, objective_label: str) -> tuple[str, str, str, str, str, str]:
    df = load_demo_cases()
    persona_df = load_persona_votes()
    seed_id = seed_from_choice(article_choice)
    objective = objective_key(objective_label)
    rows = df[(df["seed_id"].astype(int) == seed_id) & (df["objective"] == objective)].copy()
    if rows.empty:
        empty = pd.DataFrame()
        return "No saved case found.", table_html(empty), table_html(empty), table_html(empty), table_html(empty), table_html(empty)
    rows = rows.sort_values(["is_recommended", "display_order"], ascending=[False, True])
    recommended = rows[rows["is_recommended"]].iloc[0] if rows["is_recommended"].any() else rows.iloc[0]
    return (
        recommendation_card_html(recommended, heading="Recommended Headline"),
        score_summary_cards_html(score_breakdown(recommended)),
        source_pool_cards_html(source_pool_table(df, seed_id)),
        candidate_cards_html(candidate_table(rows.sort_values("display_order"))),
        table_html(persona_table(persona_df, seed_id)),
        objective_cards_html(objective_preview(df, seed_id)),
    )


def custom_args(category: str, title: str, objective: str, num_candidates: int, use_api: bool) -> argparse.Namespace:
    return argparse.Namespace(
        category=category,
        title=title,
        objective=objective,
        run_name="gradio",
        num_candidates=int(num_candidates),
        output_csv=PROJECT_ROOT / "data" / "processed" / "single_article_review_candidates.csv",
        output_html=PROJECT_ROOT / "demo" / "single_article_review.html",
        metadata=PROJECT_ROOT / "data" / "processed" / "single_article_review_metadata.json",
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=None,
        max_output_tokens=700,
        timeout=60,
        retries=2,
        reasoning_effort="none",
        dry_run=not use_api,
        force_fallback=not use_api,
        device="auto",
        clickbait_model=review.DEFAULT_CLICKBAIT_MODEL,
        max_length=96,
    )


def review_custom_summary(
    summary: str,
    category: str,
    original_title: str,
    objective_label: str,
    num_candidates: int,
    use_api_when_available: bool,
) -> tuple[str, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = clean_text(summary)
    if not summary:
        return "Summary is empty.", pd.DataFrame(), pd.DataFrame(), {}

    objective = objective_key(objective_label)
    args = custom_args(category, original_title, objective, num_candidates, use_api_when_available)
    candidates, generation_mode, _raw_output = review.generate_candidates(args, summary)
    device = review.validate_device(args.device)
    clickbait_scorer, clickbait_status = (
        review.load_clickbait_scorer(args.clickbait_model, device)
        if device != "unavailable"
        else (None, "torch unavailable")
    )
    scored = review.score_candidates(
        summary=summary,
        category=category,
        objective=objective,
        candidates=candidates,
        clickbait_scorer=clickbait_scorer,
        device=device,
        max_length=args.max_length,
        generation_mode=generation_mode,
    )
    recommended = scored[scored["is_recommended"]].iloc[0] if scored["is_recommended"].any() else scored.iloc[0]
    metadata = {
        "objective": objective,
        "generation_mode": generation_mode,
        "model": args.model if generation_mode == "api" else "fallback",
        "clickbait_model_status": clickbait_status,
        "candidate_count": int(len(scored)),
    }
    table = scored.rename(
        columns={
            "role": "Role",
            "headline": "Headline",
            "source": "Source",
            "quality_score": "Quality",
            "risk_score": "Risk/Safety",
            "clickbait_penalty": "Clickbait Penalty",
            "audience_score": "Audience",
            "objective_fit_score": "Objective Fit",
            "unified_decision_score": "Decision",
            "decision_explanation": "Explanation",
        }
    )
    visible = [
        "Role",
        "Headline",
        "Source",
        "Quality",
        "Risk/Safety",
        "Clickbait Penalty",
        "Audience",
        "Objective Fit",
        "Decision",
        "Explanation",
    ]
    return recommended_markdown(recommended), score_breakdown(recommended), table[visible], metadata


def review_custom_summary_ui(
    summary: str,
    category: str,
    original_title: str,
    objective_label: str,
    num_candidates: int,
    use_api_when_available: bool,
) -> tuple[str, str, str, dict[str, Any]]:
    recommendation, scores, candidates, metadata = review_custom_summary(
        summary,
        category,
        original_title,
        objective_label,
        num_candidates,
        use_api_when_available,
    )
    return (
        recommendation,
        score_summary_cards_html(scores),
        candidate_cards_html(candidates),
        metadata,
    )


def fetch_and_review_url_ui(
    url: str,
    category: str,
    objective_label: str,
    num_candidates: int,
    use_api_when_available: bool,
) -> tuple[str, str, str, str, str, str, str, dict[str, Any]]:
    summary, title, category_value, fetch_status = fetch_article_url_ui(url, "", "", category)
    if fetch_status.startswith("Fetch failed") or fetch_status.startswith("Enter a URL"):
        empty = table_html(pd.DataFrame())
        return summary, title, category_value, fetch_status, "", empty, empty, {"error": fetch_status}

    recommendation, scores_html, candidates_html, metadata = review_custom_summary_ui(
        summary,
        category_value,
        title,
        objective_label,
        num_candidates,
        use_api_when_available,
    )
    metadata = {
        **metadata,
        "source_url": clean_text(url),
        "fetch_status": fetch_status,
    }
    return summary, title, category_value, fetch_status, recommendation, scores_html, candidates_html, metadata


def article_choice_for_seed(df: pd.DataFrame, seed_id: int) -> str | None:
    prefix = f"{int(seed_id)} |"
    return next((choice for choice in article_choices(df) if choice.startswith(prefix)), None)


def sidebar_stats_html(df: pd.DataFrame, visible_count: int | None = None) -> str:
    article_count = df["seed_id"].nunique() if "seed_id" in df.columns else 0
    objective_count = df["objective"].nunique() if "objective" in df.columns else 0
    row_count = len(df) if visible_count is None else visible_count
    return (
        '<div class="cricli-html-stats">'
        f'<div class="cricli-html-stat"><b>{int(article_count)}</b><span>articles</span></div>'
        f'<div class="cricli-html-stat"><b>{int(objective_count)}</b><span>objectives</span></div>'
        f'<div class="cricli-html-stat"><b>{int(row_count)}</b><span>visible rows</span></div>'
        "</div>"
    )


def sidebar_article_html(df: pd.DataFrame, seed_id: int | None) -> str:
    if seed_id is None:
        return '<div class="cricli-html-current-card"><p class="cricli-html-snippet">No article selected.</p></div>'
    sub = df[df["seed_id"].astype(int) == int(seed_id)]
    if sub.empty:
        return '<div class="cricli-html-current-card"><p class="cricli-html-snippet">No article selected.</p></div>'
    first = sub.iloc[0]
    summary = clean_text(first.get("summary"))
    snippet = summary[:430] + (" ..." if len(summary) > 430 else "")
    return (
        '<div class="cricli-html-current-card">'
        '<div class="cricli-html-article-top">'
        f'<span class="cricli-html-category">{html.escape(clean_text(first.get("category")))}</span>'
        f'<span class="cricli-html-seed">#{int(seed_id)}</span>'
        "</div>"
        f'<p class="cricli-html-snippet">{html.escape(snippet)}</p>'
        "</div>"
    )


def objective_tabs_html(active_objective: str) -> str:
    tabs = []
    for objective in OBJECTIVE_ORDER:
        active = " active" if objective == active_objective else ""
        tabs.append(
            f'<span class="cricli-html-objective-tab{active}">{html.escape(OBJECTIVE_LABELS[objective])}</span>'
        )
    return '<div class="cricli-html-objective-tabs">' + "".join(tabs) + "</div>"


def scenario_banner_html(scenario_name: str | None, seed_id: int, objective: str) -> str:
    if not scenario_name or scenario_name not in SCENARIOS:
        return ""
    scenario = SCENARIOS[scenario_name]
    if int(scenario["seed_id"]) != int(seed_id) or scenario["objective"] != objective:
        return ""
    return (
        '<div class="cricli-html-scenario-banner">'
        f"<b>{html.escape(scenario_name)}</b>"
        f"<span>{html.escape(scenario['takeaway'])}</span>"
        "</div>"
    )


def selected_rows(df: pd.DataFrame, seed_id: int, objective: str) -> pd.DataFrame:
    rows = df[(df["seed_id"].astype(int) == int(seed_id)) & (df["objective"] == objective)].copy()
    if rows.empty:
        return rows
    return rows.sort_values(["is_recommended", "display_order"], ascending=[False, True])


def console_header_html(df: pd.DataFrame, article_choice: str | None, objective_label: str, scenario_name: str | None) -> str:
    if not article_choice:
        return '<div class="cricli-table-empty">No article selected.</div>'
    seed_id = seed_from_choice(article_choice)
    objective = objective_key(objective_label)
    rows = selected_rows(df, seed_id, objective)
    if rows.empty:
        return '<div class="cricli-table-empty">No saved case found.</div>'

    return (
        '<div class="cricli-html-review">'
        f'<h1 class="cricli-html-page-title">{html.escape(OBJECTIVE_LABELS[objective])} recommendation</h1>'
        f"{scenario_banner_html(scenario_name, seed_id, objective)}"
        "</div>"
    )


def console_body_html(df: pd.DataFrame, article_choice: str | None, objective_label: str, scenario_name: str | None) -> str:
    if not article_choice:
        return '<div class="cricli-table-empty">No article selected.</div>'
    seed_id = seed_from_choice(article_choice)
    objective = objective_key(objective_label)
    rows = selected_rows(df, seed_id, objective)
    if rows.empty:
        return '<div class="cricli-table-empty">No saved case found.</div>'

    article_rows = df[df["seed_id"].astype(int) == int(seed_id)]
    first = rows.iloc[0]
    recommended = rows[rows["is_recommended"]].iloc[0] if rows["is_recommended"].any() else rows.iloc[0]
    summary = clean_text(first.get("summary"))
    category = clean_text(first.get("category")).upper()
    pool_size = clean_text(first.get("hidden_candidate_pool_size")) or str(len(article_rows["headline"].dropna().unique()))
    source_count = clean_text(first.get("hidden_candidate_source_count")) or str(article_rows["candidate_source"].dropna().nunique())
    candidates = candidate_cards_html(candidate_table(rows.sort_values("display_order")))

    return (
        '<div class="cricli-html-article-context">'
        "<div>"
        f'<div class="cricli-section-label">{html.escape(category)} ARTICLE #{int(seed_id)}</div>'
        f'<p class="cricli-html-summary-text">{html.escape(summary)}</p>'
        "</div>"
        '<div class="cricli-html-metric-stack">'
        f'<div class="cricli-html-metric"><span>Internal candidates</span><b>{html.escape(pool_size)}</b></div>'
        f'<div class="cricli-html-metric"><span>Generator sources</span><b>{html.escape(source_count)}</b></div>'
        "</div>"
        "</div>"
        f"{recommendation_card_html(recommended)}"
        '<div class="cricli-html-section">Visible Decision Set</div>'
        f"{candidates}"
    )


def console_main_html(df: pd.DataFrame, article_choice: str | None, objective_label: str, scenario_name: str | None) -> str:
    return console_header_html(df, article_choice, objective_label, scenario_name) + console_body_html(
        df,
        article_choice,
        objective_label,
        scenario_name,
    )


def console_side_outputs(
    df: pd.DataFrame,
    persona_df: pd.DataFrame,
    article_choice: str | None,
    objective_label: str,
    scenario_name: str | None,
) -> tuple[str, str, str, str, str]:
    seed_id = seed_from_choice(article_choice) if article_choice else None
    sidebar = sidebar_article_html(df, seed_id)
    header = console_header_html(df, article_choice, objective_label, scenario_name)
    body = console_body_html(df, article_choice, objective_label, scenario_name)
    if seed_id is None:
        empty = table_html(pd.DataFrame())
        return sidebar, header, body, empty, empty
    return (
        sidebar,
        header,
        body,
        source_pool_cards_html(source_pool_table(df, seed_id)),
        table_html(persona_table(persona_df, seed_id)),
    )


def build_app():
    gr = import_gradio()
    cases = load_demo_cases()
    persona_df = load_persona_votes()
    choices = article_choices(cases)
    objective_labels = [OBJECTIVE_LABELS[key] for key in OBJECTIVE_ORDER]
    scenario_choices = list(SCENARIOS)
    default_scenario_name = (
        "Trust / Safety Prefers Conservative Editorial"
        if "Trust / Safety Prefers Conservative Editorial" in SCENARIOS
        else scenario_choices[0]
    )
    default_article_choice = article_choice_for_seed(cases, SCENARIOS[default_scenario_name]["seed_id"]) or (choices[0] if choices else None)
    default_objective_label = OBJECTIVE_LABELS[SCENARIOS[default_scenario_name]["objective"]]
    default_sidebar, default_header, default_body, default_sources, default_personas = console_side_outputs(
        cases,
        persona_df,
        default_article_choice,
        default_objective_label,
        default_scenario_name,
    )

    def visible_row_count(filtered_choices: list[str]) -> int:
        if not filtered_choices:
            return 0
        seed_ids = {seed_from_choice(choice) for choice in filtered_choices}
        return int(cases[cases["seed_id"].astype(int).isin(seed_ids)].shape[0])

    def apply_filter(
        query: str,
        current_article: str | None,
        current_objective: str,
        current_scenario: str,
    ) -> tuple[object, str, str, str, str, str, str]:
        query = clean_text(query).lower()
        filtered = [choice for choice in choices if query in choice.lower()] if query else choices
        if not filtered:
            filtered = choices
        value = current_article if current_article in filtered else (filtered[0] if filtered else None)
        sidebar, header, body, sources, personas = console_side_outputs(
            cases,
            persona_df,
            value,
            current_objective,
            current_scenario,
        )
        return (
            gr.update(choices=filtered, value=value),
            sidebar_stats_html(cases, visible_row_count(filtered)),
            sidebar,
            header,
            body,
            sources,
            personas,
        )

    def update_console(
        current_article: str | None,
        current_objective: str,
        current_scenario: str,
    ) -> tuple[str, str, str, str, str]:
        return console_side_outputs(
            cases,
            persona_df,
            current_article,
            current_objective,
            current_scenario,
        )

    def sync_objective_console(
        current_article: str | None,
        current_objective: str,
        current_scenario: str,
    ) -> tuple[str, str, str, str, str, str]:
        sidebar, header, body, sources, personas = console_side_outputs(
            cases,
            persona_df,
            current_article,
            current_objective,
            current_scenario,
        )
        return current_objective, sidebar, header, body, sources, personas

    def load_scenario_console(name: str) -> tuple[object, str, str, str, str, str, str, str, str]:
        scenario_config = SCENARIOS[name]
        article_value = article_choice_for_seed(cases, scenario_config["seed_id"]) or default_article_choice
        objective_value = OBJECTIVE_LABELS[scenario_config["objective"]]
        sidebar, header, body, sources, personas = console_side_outputs(
            cases,
            persona_df,
            article_value,
            objective_value,
            name,
        )
        return (
            gr.update(choices=choices, value=article_value),
            objective_value,
            objective_value,
            "",
            sidebar_stats_html(cases),
            sidebar,
            header,
            body,
            sources,
            personas,
        )

    with gr.Blocks(title="Cricli Headline Review Console") as app:
        with gr.Row(elem_classes=["cricli-html-shell"]):
            with gr.Column(scale=1, min_width=320, elem_classes=["cricli-html-sidebar"]):
                gr.HTML(
                    """
                    <div class="cricli-html-brand-row">
                      <div class="cricli-html-brand-title">Headline Review<br>Console</div>
                      <div class="cricli-html-brand-pill">Local demo</div>
                    </div>
                    """
                )
                article = gr.Dropdown(
                    choices=choices,
                    value=default_article_choice,
                    label="Select Article or Headline",
                    elem_classes=["cricli-field"],
                )
                filter_text = gr.Textbox(
                    placeholder="Filter article cards",
                    show_label=False,
                    elem_classes=["cricli-field"],
                )
                stats = gr.HTML(value=sidebar_stats_html(cases))
                with gr.Group(elem_classes=["cricli-html-scenario-panel"]):
                    scenario = gr.Dropdown(
                        choices=scenario_choices,
                        value=default_scenario_name,
                        label="Presentation Scenarios",
                        elem_classes=["cricli-field"],
                    )
                    gr.HTML(
                        '<p class="cricli-html-snippet">Use these curated cases for a cleaner presentation path.</p>'
                    )
                objective = gr.Dropdown(
                    choices=objective_labels,
                    value=default_objective_label,
                    label="Business Objective",
                    elem_classes=["cricli-field"],
                )
                current_article = gr.HTML(value=default_sidebar)

            with gr.Column(scale=5, min_width=640, elem_classes=["cricli-html-main"]):
                main_header = gr.HTML(value=default_header)
                objective_tabs = gr.Radio(
                    choices=objective_labels,
                    value=default_objective_label,
                    show_label=False,
                    container=False,
                    elem_classes=["cricli-objective-radio"],
                )
                main_body = gr.HTML(value=default_body)
                with gr.Accordion("Candidate Source Pool", open=False, elem_classes=["cricli-persona-accordion"]):
                    source_pool = gr.HTML(value=default_sources)
                with gr.Accordion("Persona Votes", open=False, elem_classes=["cricli-persona-accordion"]):
                    persona_votes = gr.HTML(value=default_personas)
                with gr.Accordion("Custom Review", open=False, elem_classes=["cricli-persona-accordion"]):
                    article_url = gr.Textbox(
                        label="Live Article URL",
                        placeholder="https://example.com/news/article",
                        elem_classes=["cricli-field"],
                    )
                    with gr.Row():
                        fetch_url = gr.Button("Fetch Article", variant="secondary")
                        fetch_and_review = gr.Button("Fetch + Run Review", variant="primary")
                    url_fetch_status = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=2):
                            summary = gr.Textbox(label="Article Summary", lines=7, elem_classes=["cricli-field"])
                            original_title = gr.Textbox(
                                label="Existing / Original Title",
                                lines=1,
                                elem_classes=["cricli-field"],
                            )
                        with gr.Column(scale=1, min_width=300):
                            category = gr.Textbox(label="Category", value="news", elem_classes=["cricli-field"])
                            custom_objective = gr.Dropdown(
                                choices=objective_labels,
                                value="Editorial",
                                label="Objective",
                                elem_classes=["cricli-field"],
                            )
                            num_candidates = gr.Slider(
                                3,
                                8,
                                value=6,
                                step=1,
                                label="Candidate Count",
                                elem_classes=["cricli-field"],
                            )
                            use_api = gr.Checkbox(label="Use API Generation When OPENAI_API_KEY Is Set", value=False)
                            run_custom = gr.Button("Run Review", variant="primary")
                    custom_recommendation = gr.Markdown()
                    with gr.Row():
                        custom_scores = gr.HTML(value=table_html(pd.DataFrame()))
                        with gr.Column():
                            gr.Markdown('<div class="cricli-gradio-section">Run Metadata</div>')
                            custom_metadata = gr.JSON(show_label=False)
                    gr.Markdown('<div class="cricli-gradio-section">Candidate Headlines</div>')
                    custom_candidates = gr.HTML(value=table_html(pd.DataFrame()))

        article.change(
            update_console,
            inputs=[article, objective, scenario],
            outputs=[current_article, main_header, main_body, source_pool, persona_votes],
        )
        objective.change(
            sync_objective_console,
            inputs=[article, objective, scenario],
            outputs=[objective_tabs, current_article, main_header, main_body, source_pool, persona_votes],
        )
        objective_tabs.change(
            sync_objective_console,
            inputs=[article, objective_tabs, scenario],
            outputs=[objective, current_article, main_header, main_body, source_pool, persona_votes],
        )
        filter_text.change(
            apply_filter,
            inputs=[filter_text, article, objective, scenario],
            outputs=[article, stats, current_article, main_header, main_body, source_pool, persona_votes],
        )
        scenario.change(
            load_scenario_console,
            inputs=[scenario],
            outputs=[
                article,
                objective,
                objective_tabs,
                filter_text,
                stats,
                current_article,
                main_header,
                main_body,
                source_pool,
                persona_votes,
            ],
        )
        fetch_url.click(
            fetch_article_url_ui,
            inputs=[article_url, summary, original_title, category],
            outputs=[summary, original_title, category, url_fetch_status],
        )
        fetch_and_review.click(
            fetch_and_review_url_ui,
            inputs=[article_url, category, custom_objective, num_candidates, use_api],
            outputs=[
                summary,
                original_title,
                category,
                url_fetch_status,
                custom_recommendation,
                custom_scores,
                custom_candidates,
                custom_metadata,
            ],
        )
        run_custom.click(
            review_custom_summary_ui,
            inputs=[summary, category, original_title, custom_objective, num_candidates, use_api],
            outputs=[custom_recommendation, custom_scores, custom_candidates, custom_metadata],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Cricli Gradio demo.")
    parser.add_argument("--server-name", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_app().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )
