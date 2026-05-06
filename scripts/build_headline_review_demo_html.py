# Build a self-contained local HTML demo for the simplified headline review workflow.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "headline_review_demo_cases.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "headline_review_console.html"

NUMERIC_COLUMNS = {
    "seed_id",
    "display_order",
    "quality_score",
    "risk_score",
    "audience_score",
    "support_score",
    "evidence_support_score",
    "objective_fit_score",
    "unified_decision_score",
    "recommendation_score",
    "base_objective_score",
    "persona_adjustment",
    "clickbait_penalty",
    "pred_overall",
    "persona_mean_overall",
    "summary_support_rate",
    "llm_overall",
    "hidden_candidate_pool_size",
    "hidden_candidate_source_count",
}

PREFERRED_OBJECTIVE_ORDER = [
    "trust_safety",
    "safety",
    "growth",
    "editorial",
    "specificity",
]


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        return clean_value(value.item())
    return str(value)


def row_to_option(row: pd.Series) -> dict[str, Any]:
    option: dict[str, Any] = {}
    for key, value in row.items():
        option[key] = clean_value(value)
    option["is_recommended"] = bool(row.get("is_recommended", False))
    return option


def objective_sort_key(objective: str) -> tuple[int, str]:
    if objective in PREFERRED_OBJECTIVE_ORDER:
        return (PREFERRED_OBJECTIVE_ORDER.index(objective), objective)
    return (len(PREFERRED_OBJECTIVE_ORDER), objective)


def build_payload(df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "seed_id",
        "objective",
        "objective_name",
        "category",
        "summary",
        "display_label",
        "display_order",
        "headline",
        "is_recommended",
        "recommendation_summary",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df.copy()
    df["seed_id"] = df["seed_id"].astype(int)
    df["display_order"] = df["display_order"].astype(int)
    df = df.sort_values(["seed_id", "objective", "display_order", "headline"])

    objective_rows = (
        df[["objective", "objective_name"]]
        .drop_duplicates()
        .sort_values("objective", key=lambda col: col.map(objective_sort_key))
    )
    objectives = [
        {"objective": str(row.objective), "objective_name": str(row.objective_name)}
        for row in objective_rows.itertuples(index=False)
    ]

    articles: dict[int, dict[str, Any]] = {}
    for (seed_id, objective), group in df.groupby(["seed_id", "objective"], sort=False):
        first = group.iloc[0]
        article = articles.setdefault(
            int(seed_id),
            {
                "seed_id": int(seed_id),
                "category": clean_value(first.get("category")),
                "summary": clean_value(first.get("summary")),
                "objectives": {},
            },
        )
        options = [row_to_option(row) for _, row in group.iterrows()]
        recommended = next((option for option in options if option.get("is_recommended")), options[0])
        article["objectives"][str(objective)] = {
            "objective": str(objective),
            "objective_name": clean_value(first.get("objective_name")),
            "hidden_candidate_pool_size": clean_value(first.get("hidden_candidate_pool_size")),
            "hidden_candidate_source_count": clean_value(first.get("hidden_candidate_source_count")),
            "recommended_headline": recommended.get("headline"),
            "recommended_summary": recommended.get("recommendation_summary"),
            "recommended_score": recommended.get("recommendation_score"),
            "options": options,
        }

    return {
        "objectives": objectives,
        "articles": [articles[key] for key in sorted(articles)],
        "metadata": {
            "article_count": len(articles),
            "row_count": int(len(df)),
            "objective_count": len(objectives),
        },
    }


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    template = Template(r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Headline Review Console</title>
  <style>
    :root {
      --bg: #f6f5ef;
      --panel: #ffffff;
      --panel-soft: #f9faf5;
      --ink: #18212f;
      --muted: #687487;
      --line: #d9decc;
      --teal: #12756d;
      --teal-soft: #dff1ed;
      --blue: #315fba;
      --gold: #b7791f;
      --rose: #be3a51;
      --shadow: 0 10px 30px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input { font: inherit; }
    .app-shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #fcfcf7;
      padding: 18px 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 100vh;
      position: sticky;
      top: 0;
    }
    .brand-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand-title {
      font-size: 19px;
      font-weight: 760;
      line-height: 1.1;
    }
    .brand-pill {
      border: 1px solid var(--line);
      color: var(--muted);
      padding: 5px 8px;
      border-radius: 7px;
      font-size: 12px;
      white-space: nowrap;
    }
    .search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 11px 12px;
      outline: none;
    }
    .search:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 3px var(--teal-soft);
    }
    .mini-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .mini-stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 9px;
    }
    .mini-stat b { display: block; font-size: 18px; }
    .mini-stat span { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }
    .article-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      overflow: auto;
      padding-right: 2px;
    }
    .article-button {
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 11px;
      cursor: pointer;
      min-height: 74px;
      transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }
    .article-button:hover { border-color: #aeb9a1; transform: translateY(-1px); }
    .article-button.active { border-color: var(--teal); box-shadow: 0 0 0 2px var(--teal-soft); }
    .article-topline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .seed { color: var(--muted); font-size: 12px; }
    .category {
      color: var(--teal);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .article-snippet {
      color: #394456;
      font-size: 13px;
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    main {
      min-width: 0;
      padding: 22px clamp(18px, 3vw, 42px) 42px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }
    .page-title {
      font-size: clamp(25px, 3vw, 38px);
      line-height: 1.04;
      margin: 0;
      max-width: 860px;
    }
    .objective-tabs {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      margin: 18px 0;
    }
    .tab {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 9px 12px;
      cursor: pointer;
      color: #2b3444;
      font-weight: 650;
    }
    .tab.active {
      background: var(--ink);
      border-color: var(--ink);
      color: #fff;
    }
    .article-context {
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 17px 0;
      margin-bottom: 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 18px;
    }
    .summary-text {
      font-size: 16px;
      line-height: 1.55;
      margin: 7px 0 0;
      color: #323d4d;
    }
    .context-meta {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      align-content: start;
    }
    .metric-chip {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 10px 11px;
    }
    .metric-chip span { color: var(--muted); font-size: 12px; display: block; }
    .metric-chip b { display: block; margin-top: 3px; font-size: 18px; }
    .recommendation {
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 5px solid var(--teal);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 18px;
    }
    .section-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
      margin: 0 0 8px;
    }
    .recommended-title {
      margin: 0;
      font-size: clamp(23px, 3vw, 34px);
      line-height: 1.12;
    }
    .reason {
      margin: 12px 0 0;
      line-height: 1.48;
      color: #3b4657;
    }
    .score-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }
    .score-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcf9;
      min-width: 0;
    }
    .score-box-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .bar {
      height: 7px;
      background: #e4e8dd;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill { height: 100%; background: var(--teal); border-radius: 999px; width: 0%; }
    .bar-fill.risk { background: var(--blue); }
    .bar-fill.audience { background: var(--gold); }
    .bar-fill.objective { background: var(--rose); }
    .bar-fill.support { background: var(--rose); }
    .option-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .option-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      min-height: 220px;
      display: flex;
      flex-direction: column;
      gap: 11px;
    }
    .option-card.recommended {
      border-color: var(--teal);
      box-shadow: 0 0 0 2px var(--teal-soft);
    }
    .option-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }
    .option-label {
      font-size: 13px;
      font-weight: 760;
      color: var(--ink);
    }
    .option-source {
      font-size: 11px;
      color: var(--muted);
      text-align: right;
      line-height: 1.3;
      max-width: 140px;
    }
    .headline {
      font-size: 18px;
      line-height: 1.25;
      font-weight: 720;
      margin: 0;
    }
    .decision-note {
      margin: 0;
      color: #465266;
      font-size: 13px;
      line-height: 1.42;
      border-left: 3px solid var(--line);
      padding-left: 10px;
    }
    .option-card.recommended .decision-note { border-left-color: var(--teal); }
    .tag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: auto;
    }
    .tag {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 4px 7px;
      color: var(--muted);
      font-size: 11px;
      background: #fbfcf9;
    }
    .tag.rec { color: var(--teal); border-color: #9dd1c8; background: var(--teal-soft); }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.55);
    }
    @media (max-width: 920px) {
      .app-shell { grid-template-columns: 1fr; }
      .sidebar { position: relative; min-height: 0; border-right: 0; border-bottom: 1px solid var(--line); }
      .article-list { max-height: 260px; }
      .article-context { grid-template-columns: 1fr; }
      .score-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      main { padding: 18px 14px 32px; }
      .score-grid { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .mini-stats { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand-row">
        <div class="brand-title">Headline Review Console</div>
        <div class="brand-pill">Local demo</div>
      </div>
      <input id="search" class="search" type="search" placeholder="Search articles or headlines">
      <div class="mini-stats">
        <div class="mini-stat"><b id="statArticles">0</b><span>articles</span></div>
        <div class="mini-stat"><b id="statObjectives">0</b><span>objectives</span></div>
        <div class="mini-stat"><b id="statRows">0</b><span>visible rows</span></div>
      </div>
      <div id="articleList" class="article-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <h1 class="page-title" id="pageTitle">Headline recommendation</h1>
      </div>
      <div id="objectiveTabs" class="objective-tabs"></div>
      <section class="article-context">
        <div>
          <p class="section-label" id="articleMeta">Article</p>
          <p class="summary-text" id="summaryText"></p>
        </div>
        <div class="context-meta">
          <div class="metric-chip"><span>Internal candidates</span><b id="candidatePool">0</b></div>
          <div class="metric-chip"><span>Generator sources</span><b id="sourcePool">0</b></div>
        </div>
      </section>
      <section class="recommendation">
        <p class="section-label">Recommended headline</p>
        <h2 class="recommended-title" id="recommendedTitle"></h2>
        <p class="reason" id="recommendedReason"></p>
        <div id="recommendedScores" class="score-grid"></div>
      </section>
      <p class="section-label">Visible decision set</p>
      <section id="optionGrid" class="option-grid"></section>
    </main>
  </div>
  <script>
    const PAYLOAD = $data_json;
    const OBJECTIVES = PAYLOAD.objectives || [];
    const ARTICLES = PAYLOAD.articles || [];
    const state = {
      query: "",
      objective: OBJECTIVES[0] ? OBJECTIVES[0].objective : "",
      seedId: ARTICLES[0] ? ARTICLES[0].seed_id : null
    };

    function byId(id) { return document.getElementById(id); }

    function text(value, fallback) {
      if (value === null || value === undefined || value === "") return fallback || "";
      return String(value);
    }

    function fmt(value, digits) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits === undefined ? 2 : digits);
    }

    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
      return Math.max(0, Math.min(100, Number(value) * 100));
    }

    function articleText(article) {
      const chunks = [article.category, article.summary];
      Object.values(article.objectives || {}).forEach(obj => {
        (obj.options || []).forEach(option => chunks.push(option.headline, option.display_label, option.roles, option.decision_explanation));
      });
      return chunks.filter(Boolean).join(" ").toLowerCase();
    }

    function getArticle(seedId) {
      return ARTICLES.find(article => article.seed_id === seedId) || ARTICLES[0];
    }

    function getObjective(article) {
      if (!article) return null;
      if (article.objectives && article.objectives[state.objective]) return article.objectives[state.objective];
      const firstKey = Object.keys(article.objectives || {})[0];
      return firstKey ? article.objectives[firstKey] : null;
    }

    function getRecommended(obj) {
      if (!obj || !obj.options || !obj.options.length) return null;
      return obj.options.find(option => option.is_recommended) || obj.options[0];
    }

    function filteredArticles() {
      const query = state.query.trim().toLowerCase();
      if (!query) return ARTICLES;
      return ARTICLES.filter(article => articleText(article).includes(query));
    }

    function makeButton(className, textValue, onClick) {
      const button = document.createElement("button");
      button.className = className;
      button.type = "button";
      button.textContent = textValue;
      button.addEventListener("click", onClick);
      return button;
    }

    function renderStats() {
      byId("statArticles").textContent = PAYLOAD.metadata.article_count || ARTICLES.length;
      byId("statObjectives").textContent = PAYLOAD.metadata.objective_count || OBJECTIVES.length;
      byId("statRows").textContent = PAYLOAD.metadata.row_count || 0;
    }

    function renderList() {
      const list = byId("articleList");
      list.innerHTML = "";
      const rows = filteredArticles();
      if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No matching articles.";
        list.appendChild(empty);
        return;
      }
      if (!rows.find(article => article.seed_id === state.seedId)) {
        state.seedId = rows[0].seed_id;
      }
      rows.forEach(article => {
        const button = makeButton(
          "article-button" + (article.seed_id === state.seedId ? " active" : ""),
          "",
          () => { state.seedId = article.seed_id; render(); }
        );
        const top = document.createElement("div");
        top.className = "article-topline";
        const category = document.createElement("span");
        category.className = "category";
        category.textContent = text(article.category, "article");
        const seed = document.createElement("span");
        seed.className = "seed";
        seed.textContent = "#" + article.seed_id;
        top.append(category, seed);
        const snippet = document.createElement("div");
        snippet.className = "article-snippet";
        snippet.textContent = text(article.summary, "");
        button.append(top, snippet);
        list.appendChild(button);
      });
    }

    function renderTabs() {
      const tabs = byId("objectiveTabs");
      tabs.innerHTML = "";
      OBJECTIVES.forEach(obj => {
        const tab = makeButton(
          "tab" + (obj.objective === state.objective ? " active" : ""),
          obj.objective_name || obj.objective,
          () => { state.objective = obj.objective; render(); }
        );
        tabs.appendChild(tab);
      });
    }

    function scoreBox(label, value, className) {
      const box = document.createElement("div");
      box.className = "score-box";
      const top = document.createElement("div");
      top.className = "score-box-top";
      const left = document.createElement("span");
      left.textContent = label;
      const right = document.createElement("b");
      right.textContent = fmt(value, 2);
      top.append(left, right);
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("div");
      fill.className = "bar-fill" + (className ? " " + className : "");
      fill.style.width = pct(value) + "%";
      bar.appendChild(fill);
      box.append(top, bar);
      return box;
    }

    function renderScoreGrid(container, option) {
      container.innerHTML = "";
      container.append(
        scoreBox("Quality", option ? option.quality_score : null, ""),
        scoreBox("Risk safety", option ? option.risk_score : null, "risk"),
        scoreBox("Audience fit", option ? option.audience_score : null, "audience"),
        scoreBox("Objective fit", option ? option.objective_fit_score : null, "objective")
      );
    }

    function renderOption(option) {
      const card = document.createElement("article");
      card.className = "option-card" + (option.is_recommended ? " recommended" : "");
      const head = document.createElement("div");
      head.className = "option-head";
      const label = document.createElement("div");
      label.className = "option-label";
      label.textContent = text(option.display_label, "Candidate");
      const source = document.createElement("div");
      source.className = "option-source";
      source.textContent = text(option.candidate_source, option.variant);
      head.append(label, source);

      const headline = document.createElement("p");
      headline.className = "headline";
      headline.textContent = text(option.headline, "Untitled");

      const decision = document.createElement("p");
      decision.className = "decision-note";
      decision.textContent = text(option.decision_explanation, option.recommendation_summary || "");

      const scores = document.createElement("div");
      scores.className = "score-grid";
      renderScoreGrid(scores, option);

      const tags = document.createElement("div");
      tags.className = "tag-row";
      if (option.is_recommended) {
        const tag = document.createElement("span");
        tag.className = "tag rec";
        tag.textContent = "Selected";
        tags.appendChild(tag);
      }
      [option.roles, "Decision " + fmt(option.unified_decision_score, 2), "Support " + fmt(option.support_score, 2)].forEach(value => {
        if (!value) return;
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = value;
        tags.appendChild(tag);
      });

      card.append(head, headline, decision, scores, tags);
      return card;
    }

    function renderMain() {
      const article = getArticle(state.seedId);
      const obj = getObjective(article);
      const recommended = getRecommended(obj);
      if (!article || !obj) return;

      byId("pageTitle").textContent = text(obj.objective_name, "Headline") + " recommendation";
      byId("articleMeta").textContent = text(article.category, "Article") + " article #" + article.seed_id;
      byId("summaryText").textContent = text(article.summary, "");
      byId("candidatePool").textContent = text(obj.hidden_candidate_pool_size, "0");
      byId("sourcePool").textContent = text(obj.hidden_candidate_source_count, "0");
      byId("recommendedTitle").textContent = recommended ? text(recommended.headline, "") : "";
      byId("recommendedReason").textContent = recommended ? text(recommended.decision_explanation, recommended.recommendation_summary || "") : "";
      renderScoreGrid(byId("recommendedScores"), recommended);

      const grid = byId("optionGrid");
      grid.innerHTML = "";
      (obj.options || []).forEach(option => grid.appendChild(renderOption(option)));
    }

    function render() {
      renderStats();
      renderTabs();
      renderList();
      renderMain();
    }

    byId("search").addEventListener("input", event => {
      state.query = event.target.value;
      render();
    });

    render();
  </script>
</body>
</html>
''')
    return template.substitute(data_json=data_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML console from simplified headline review cases."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    payload = build_payload(df)
    html = render_html(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
