# Cricli: Headline Review & Selection Console

Cricli is a course project that turns headline generation into a controllable review and selection workflow for content teams.

The project is no longer framed as "our model writes better headlines than GenAI." Direct GenAI headline generation is already a strong baseline. The more realistic product value is:

```text
GenAI can generate headlines,
but editors and growth teams still need a decision layer
that compares candidates, checks risk, estimates audience fit,
and explains which headline should be published under a given objective.
```

## Product Use Case

Target users:

- News editors choosing titles for article pages or home feeds.
- Content operations teams reviewing AI-generated copy before publishing.
- Growth teams choosing titles for feeds, newsletters, push notifications, or campaign content.

Core problem:

- A single GenAI output may be fluent but not necessarily publishable.
- Different business goals prefer different headlines.
- A growth-oriented title may be engaging but risky.
- A trust-oriented title may be safer but less attractive.
- Editors need a clear reason for choosing one headline over another.

Cricli acts as a **headline review console**:

```text
article summary
-> candidate headlines from multiple sources
-> local critic scores
-> LLM-as-judge labels
-> audience/persona voting
-> objective-specific selector
-> recommended headline with score breakdown
```

## What The System Does

Given an article summary and a set of candidate headlines, the system evaluates each candidate from several perspectives:

- Clickbait risk
- Faithfulness to the article summary
- Clarity
- Specificity
- Attractiveness
- Non-clickbait quality
- Pairwise preference score
- Audience/persona preference
- Objective-specific fit

It then recommends a headline under different operating modes:

- **Trust / Safety:** prioritize faithful, clear, non-clickbait headlines.
- **Growth:** prioritize attractive and engaging headlines while controlling risk.
- **Editorial:** prioritize balanced, compact, publication-ready headlines.
- **Specificity:** prioritize concrete and supported details.

## Current Implementation

The current repository implements the research and data pipeline behind this console.

### Candidate Sources

- Original MIND human headlines.
- GPT zero-shot generated headlines.
- Critic-guided rewrite headlines.
- Best-of-N agentic candidate generation.
- Generic FLAN-T5-small SFT headline generator outputs.
- Specificity-aware FLAN-T5-small SFT headline generator outputs.

The SFT generator work is kept as an auxiliary candidate source. It is not the core product claim.

### Evaluators

- **Clickbait penalty critic:** DistilBERT classifier trained on clickbait/non-clickbait examples.
- **Quality reward critic:** DistilBERT reward regressor distilled from LLM-as-judge labels.
- **Pairwise reward critic:** local preference scorer trained from pairwise judge examples.
- **LLM-as-judge:** reference evaluator for faithfulness, clarity, specificity, attractiveness, non-clickbait, and overall quality.
- **Audience/persona voting:** simulates trust-sensitive, growth-oriented, busy-reader, and editorial-reviewer preferences.

### Selection Layer

- Multi-agent candidate matrix with 1,000 candidate actions over 100 fixed evaluation articles.
- Objective-specific selectors for trust/safety, growth, editorial quality, and specificity.
- Persona voting results over 100 evaluation articles.

## Key Findings So Far

1. **Direct GenAI is a strong headline generator.**

   Zero-shot GPT headlines are hard to beat on average LLM-judge quality. This is why the product should not claim to replace GenAI generation.

2. **Low clickbait does not guarantee a better headline.**

   Critic-guided rewrite lowered clickbait penalty, but sometimes reduced overall editorial quality.

3. **Local reward models are useful but biased.**

   The local reward critic can guide cheap reranking, but it sometimes overvalues formal, specific, summary-like titles. LLM judge and persona voting help reveal this misalignment.

4. **Headline quality is objective-dependent.**

   Trust, growth, editorial, and specificity objectives select different candidates. This supports the product idea of a review and selection console.

5. **Audience preference is not uniform.**

   Persona voting shows meaningful disagreement between trust-sensitive, growth, busy-reader, and editorial personas.

## Important Results

### Clickbait Penalty Critic

The trained DistilBERT clickbait critic achieved strong held-out performance:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9891 |
| F1 | 0.9890 |
| ROC-AUC | 0.9988 |

### Agentic / Reward-Guided Selection

Agentic v3 reduced clickbait risk and narrowed the LLM-judge gap to zero-shot:

| Variant | LLM Judge Overall | Clickbait Penalty |
| --- | ---: | ---: |
| Zero-shot | 4.77 | 0.087 |
| Optimized rewrite | 4.73 | 0.060 |
| Agentic selected v3 | 4.50 | 0.063 |

Interpretation: the system improves controllability and lowers clickbait risk, but direct zero-shot remains the strongest average generator.

### Audience Persona Voting

Persona voting was completed on 100 seed examples with 4 personas.

Consensus best counts:

| Variant | Count |
| --- | ---: |
| Zero-shot | 51 |
| Original human headline | 46 |
| Generic SFT | 2 |
| Agentic selected | 1 |

Interpretation: zero-shot and human headlines remain strong, but the voting layer gives a structured way to compare audience preferences before publication.

### SFT Generator Baseline

The SFT line was rerun with two local FLAN-T5-small generators. These models are useful as auxiliary candidate sources, but they are not the strongest headline generators in LLM-judge evaluation.

| Variant | LLM Judge Overall | Local Final Score |
| --- | ---: | ---: |
| Original human headline | 4.23 | 3.989 |
| Generic SFT | 3.78 | 4.711 |
| Specificity-aware SFT | 3.86 | 4.713 |

Interpretation: local critics score SFT outputs highly, but LLM judge still prefers original human headlines. This supports the reward-misalignment analysis and keeps SFT as a baseline rather than the project center.

## Repository Layout

```text
data/docs/                       Dataset manifest
data/processed/                  Processed datasets, judge outputs, reports, and selection matrices
docs/                            Project summary and structure notes
scripts/                         Data processing, training, judging, scoring, and analysis scripts
requirements-clickbait-bert.txt  Core dependencies used for critic training
requirements-demo.txt            Gradio demo dependency
```

Large model weights, checkpoints, raw data, and local virtual environments are intentionally excluded from Git.

## Local Reproduction Guide

This section is for teammates who want to clone the repository and reproduce the current project state on their own machine.

### 1. Clone The Repository

```bash
git clone https://github.com/Spi1er/Cricli.git
cd Cricli
```

Run project commands from the repository root, `Cricli`.

### 2. Create The Python Environment

Recommended Python version: Python 3.11 or 3.12. Python 3.13 worked in the original local environment for several scripts, but some ML packages may be easier to install on Python 3.11/3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-clickbait-bert.txt
pip install -r requirements-demo.txt
```

Check the core packages:

```bash
python - <<'PY'
import torch, transformers, pandas, sklearn
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("ok")
PY
```

On Apple Silicon, most local critic scripts can use MPS with `--device mps` when the script exposes a device argument. CPU also works, but training and scoring will be slower.

### 3. What Is Already In Git

The repository includes:

- Source code in `scripts/`.
- Project notes in `docs/`.
- Dataset manifest in `data/docs/`.
- Small processed CSV/JSON/JSONL/Markdown artifacts in `data/processed/`.
- Current evaluation reports, judge labels, persona votes, and objective-selection matrices.
- Static HTML demo and Gradio demo code.

The repository does not include:

- Local model weights under `models/`.
- Hugging Face base model cache.
- Raw MIND data under `data/raw/`.
- Local virtual environments.
- API keys.
- Local single-article scratch outputs.

This means teammates can immediately inspect the project results, but must retrain or restore local models before rerunning every scoring script end to end.

### 4. Quick Reproduction From Existing Artifacts

After installing dependencies, teammates can inspect the main outputs without regenerating API calls or retraining models:

```bash
sed -n '1,180p' docs/WORK_SUMMARY.md
sed -n '1,180p' docs/PROJECT_STRUCTURE.md
sed -n '1,180p' docs/REPRODUCIBILITY_CHECKLIST.md
sed -n '1,180p' data/processed/headline_multi_agent_objective_profile.md
sed -n '1,180p' data/processed/headline_audience_persona_votes_profile.md
sed -n '1,180p' data/processed/headline_quality_llm_judge_agentic_v3_specificity_profile.md
```

These files reproduce the current narrative, model comparison, persona voting, and objective-selection findings.

To check whether the local environment, data, models, and demos are ready, run:

```bash
python scripts/check_project_assets.py --python .venv/bin/python
```

This writes `docs/PROJECT_ASSET_CHECK.md` and `data/processed/project_asset_check.json`.

To rebuild the product-facing review-console demo from existing artifacts, run:

```bash
python scripts/run_product_demo.py --limit-seeds 10 --python .venv/bin/python
```

This builds two demo surfaces from the same processed results: a compact 10-article static HTML fallback and a full 100-seed Gradio case explorer. It also refreshes the asset check and renders the self-contained HTML output. The lower-level commands are still available when needed:

```bash
python scripts/build_headline_review_demo_cases.py --limit-seeds 10
python scripts/build_headline_review_demo_cases.py --output data/processed/headline_review_demo_cases_full.csv --report data/processed/headline_review_demo_cases_full_profile.md --metadata data/processed/headline_review_demo_cases_full_metadata.json
python scripts/build_headline_review_demo_html.py
```

The generated static demo is:

```text
demo/headline_review_console.html
```

It can be opened directly in a browser and does not require a web server.

### 5. Rebuild Processed Datasets

The original raw dataset files are not tracked in Git. To fully rebuild `data/processed/`, first restore or download the raw data described in:

```text
data/docs/DATASET_MANIFEST.md
```

Expected raw-data location:

```text
data/raw/
```

Then run:

```bash
python scripts/build_processed_datasets.py
python scripts/build_headline_generation_seed.py
```

This rebuilds the clickbait split, MIND headline pool sample, and fixed 100-example headline generation seed set used across the project.

### 6. Retrain Local Critics

Because model weights are excluded from Git, retrain the local critics before rerunning model-based scoring or agentic selection. First prepare the local DistilBERT base checkpoint used by the critic scripts:

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

out = Path("models/base/distilbert-base-uncased-seqcls")
out.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer.save_pretrained(out)
model.save_pretrained(out)
print(f"saved base checkpoint to {out}")
PY
```

This command downloads from Hugging Face the first time it is run. After the base checkpoint exists locally, train the critics.

Clickbait penalty critic:

```bash
python scripts/train_clickbait_penalty_bert.py \
  --data data/processed/clickbait_penalty_splits.csv \
  --out models/clickbait_penalty_distilbert \
  --device mps
```

If MPS is unavailable, use:

```bash
python scripts/train_clickbait_penalty_bert.py \
  --data data/processed/clickbait_penalty_splits.csv \
  --out models/clickbait_penalty_distilbert \
  --device cpu
```

Reward critics:

```bash
python scripts/train_headline_quality_reward_critic.py --device mps
python scripts/train_headline_pairwise_reward_critic.py --device mps
```

Use `--device cpu` instead if MPS is unavailable.

These scripts use the reward and pairwise examples in `data/processed/`.

To train the latest v2 critics used by the current rerun:

```bash
python scripts/build_reward_training_v2.py

python scripts/train_headline_quality_reward_critic.py \
  --data data/processed/headline_quality_reward_model_examples_v2.jsonl \
  --out models/headline_quality_reward_distilbert_v2 \
  --device mps

python scripts/train_headline_pairwise_reward_critic.py \
  --data data/processed/headline_quality_pairwise_preferences_v2.jsonl \
  --out models/headline_pairwise_reward_distilbert_v2 \
  --device mps
```

### 7. Optional API-Based Generation And Judging

The API-based scripts require an OpenAI-compatible API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Zero-shot headline generation:

```bash
python scripts/run_zero_shot_headline_generation.py \
  --input data/processed/headline_generation_eval_seed_100.csv \
  --output data/processed/headline_generation_zero_shot_100.csv \
  --metadata data/processed/headline_generation_zero_shot_100_metadata.json \
  --model gpt-4o-mini \
  --overwrite-existing
```

LLM-as-judge evaluation:

```bash
python scripts/run_llm_judge_headline_quality.py
```

Agentic comparison judging:

```bash
python scripts/run_llm_judge_agentic_comparison.py
```

API calls cost money and can take time. Use `--dry-run` where available before launching a full run.

Reproducibility note: OpenAI generation/judging and local SFT training are stochastic, so exact headline strings and rationales can differ across reruns. Cricli fixes the 100-example evaluation seed set and reports aggregate metrics, which preserve the main conclusions without requiring bit-for-bit identical outputs.

### 8. Reproduce The Current Selection Layer

After processed artifacts and local critic outputs are available, rebuild the objective-selection matrix:

```bash
python scripts/build_multi_agent_objective_matrix.py
```

Run audience/persona voting if API access is configured:

```bash
python scripts/run_audience_persona_voting.py
python scripts/analyze_audience_persona_votes.py
```

The main generated reports are:

```text
data/processed/headline_multi_agent_objective_profile.md
data/processed/headline_audience_persona_votes_profile.md
```

### 9. Expected Reproduction Levels

There are four practical levels of reproduction:

| Level | What To Run | Requires API? | Requires Training? |
| --- | --- | ---: | ---: |
| Read current results | Inspect `docs/` and `data/processed/*.md` | No | No |
| Rebuild local critics | Dataset scripts + critic training | No | Yes |
| Regenerate LLM outputs | Generation, judge, persona scripts | Yes | Optional |
| Full raw-data rebuild | Restore `data/raw/`, rebuild processed data, retrain, regenerate API outputs | Yes | Yes |

For most group review and report writing, Level 1 is enough. For model development, Level 2 is needed. For refreshing judge labels or persona votes, Level 3 is needed. Level 4 is only needed when rebuilding everything from raw external datasets.

## Recommended Reading Order

1. `docs/PROJECT_STRUCTURE.md`
2. `docs/PROJECT_CODE_STRUCTURE.md`
3. `docs/SETUP_NOTES.md`
4. `docs/REPRODUCIBILITY_CHECKLIST.md`
5. `docs/PRODUCT_DEMO_SCENARIOS.md`
6. `docs/DEMO_DELIVERY_SCOPE.md`
7. `docs/SIMPLIFIED_PRODUCT_WORKFLOW.md`
8. `docs/TEAM_WORKPLAN.md`
9. `docs/EVALUATION_REWARD_ANALYSIS.md`
10. `docs/WORK_SUMMARY.md`
11. `data/processed/headline_review_demo_profile.md`
12. `data/processed/headline_multi_agent_objective_profile.md`
13. `data/processed/headline_audience_persona_votes_profile.md`
14. `data/processed/headline_persona_calibrated_objective_profile.md`
15. `data/processed/headline_quality_llm_judge_agentic_v3_specificity_profile.md`
16. `data/processed/bootstrap_significance.md`
17. `data/processed/headline_sft_judge_error_analysis.md`

## Main Scripts

Candidate generation and rewriting:

- `scripts/run_zero_shot_headline_generation.py`
- `scripts/run_critic_guided_rewrite.py`
- `scripts/run_critic_guided_rewrite_round2.py`
- `scripts/run_agentic_headline_optimizer.py`

Critic and reward model training:

- `scripts/train_clickbait_penalty_bert.py`
- `scripts/train_headline_quality_reward_critic.py`
- `scripts/train_headline_pairwise_reward_critic.py`

Evaluation and selection:

- `scripts/run_llm_judge_headline_quality.py`
- `scripts/run_llm_judge_agentic_comparison.py`
- `scripts/build_multi_agent_objective_matrix.py`
- `scripts/build_persona_calibrated_selector.py`
- `scripts/run_product_demo.py`
- `scripts/build_headline_review_demo_cases.py`
- `scripts/build_headline_review_demo_html.py`
- `scripts/check_project_assets.py`
- `scripts/review_single_article.py`
- `scripts/bootstrap_significance.py`
- `scripts/run_audience_persona_voting.py`
- `scripts/analyze_audience_persona_votes.py`

Auxiliary SFT generator experiments:

- `scripts/build_headline_sft_dataset.py`
- `scripts/train_headline_generator_sft.py`
- `scripts/evaluate_sft_generators.py`
- `scripts/run_llm_judge_sft_comparison.py`
- `scripts/analyze_sft_judge_errors.py`

## Demo Direction

The final demo is a lightweight headline review console. The static fallback uses the compact `data/processed/headline_review_demo_cases.csv`, while the Gradio live demo uses the full `data/processed/headline_review_demo_cases_full.csv` when it is available. The current local static version is generated at `demo/headline_review_console.html`:

```text
Select one article summary
-> show 3-4 visible options: Human baseline, GenAI baseline, Low-risk alternative, Recommended
-> show unified scores: Quality, Risk/Safety, Audience Fit, Objective Fit
-> switch objective: trust / growth / editorial / specificity
-> show the persona-calibrated recommendation and why alternatives were not selected
```

The demo should emphasize decision support, not raw generation quality. The full research candidate pool stays hidden behind the simplified demo datasets. Use `scripts/run_product_demo.py --limit-seeds 10` as the main demo rebuild command. The `--limit-seeds 10` setting controls only the compact static HTML fallback; the Gradio case explorer is rebuilt from all 100 fixed article seeds.

For the live AI-style demo, run the Gradio app:

```bash
python demo/gradio_app.py
```

The Gradio demo exposes five presentation scenarios, a 100-seed saved case explorer, candidate source pool summaries, candidate-level clickbait penalty, quality/risk/audience/objective scores, persona votes, and final recommendation explanations.

If port 7860 is busy:

```bash
python demo/gradio_app.py --port 7861
```

For a real single-article use case, run:

```bash
python scripts/review_single_article.py \
  --summary "paste article summary here" \
  --category news \
  --objective editorial \
  --run-name example_article
```

Use `--objective all` to generate trust/safety, growth, editorial, and specificity recommendations from the same candidate set:

```bash
python scripts/review_single_article.py \
  --summary "paste article summary here" \
  --category news \
  --objective all \
  --run-name example_article
```

Without `--run-name`, the script writes the default latest-run files:

```text
data/processed/single_article_review_candidates.csv
demo/single_article_review.html
data/processed/single_article_review_metadata.json
```

With `--run-name`, it writes suffixed files and avoids overwriting previous reviews. Use `--dry-run` or `--force-fallback` to avoid API calls and generate deterministic fallback candidates.

## Testing And Troubleshooting

Run lightweight tests that do not require model weights or API keys:

```bash
python -m unittest discover -s tests
```

The smoke tests validate tracked data schemas, demo fallback scoring, live URL HTML extraction, and the asset-check integration path. They are lightweight enough to run locally and can be wired into CI later if workflow permissions are available.

For setup, API, model, port, and reproducibility issues, see:

```text
docs/TROUBLESHOOTING.md
```

## Project Positioning

Best short description:

> Cricli is an audience-aware headline review and selection system. It uses local critics, LLM-as-judge labels, persona voting, and objective-specific selectors to help content teams choose publishable headlines from GenAI and human-written candidates.

This framing keeps the project aligned with the original proposal while giving it a clearer practical use case.
