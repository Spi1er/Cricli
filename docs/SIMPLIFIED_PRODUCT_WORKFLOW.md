# Simplified Product Workflow

This document defines the simplified product/demo version of Cricli.

The full repository contains many research stages because we tested baselines, critics, reward models, SFT generators, agentic reranking, LLM judges, and persona voting. That complexity is useful for the report, but it should not be exposed directly in the product demo.

The demo should present Cricli as a clear headline review and selection console.

## 1. Two-Layer Architecture

### Research Pipeline

The research pipeline keeps all experimental evidence:

```text
data preparation
-> candidate generation variants
-> clickbait scoring
-> LLM-as-judge labels
-> local reward critics
-> agentic / best-of-N reranking
-> objective selection
-> persona voting
-> persona-calibrated selection
```

This layer is useful for methodology, ablation, and final report evidence.

### Product / Demo Pipeline

The product layer should be much simpler:

```text
Input article
-> Candidate Builder
-> Unified Evaluator
-> Objective / Persona Selector
-> Recommendation + Explanation
```

This is the workflow that should be shown to users.

## 2. Current Overlap And Simplification

### Overlap 1: Multiple Evaluators

Current overlapping evaluators:

- clickbait critic;
- LLM-as-judge;
- local pointwise reward critic;
- local pairwise reward critic;
- persona voting.

Simplified product framing:

```text
Unified Evaluator = quality + risk/safety + audience fit + objective fit
```

The user does not need to see every raw evaluator. The demo can show grouped scores:

| Display Score | Hidden Inputs |
| --- | --- |
| Quality | `pred_overall` blended with summary support |
| Risk / Safety | `clickbait_penalty`, `pred_non_clickbait` |
| Audience Fit | persona `overall`, persona best / consensus flags, or neutral fallback |
| Objective Fit | persona-calibrated objective score for trust, growth, editorial, or specificity |

### Overlap 2: Clickbait Critic And Non-Clickbait Scores

The clickbait critic should remain, but not as the only definition of quality.

Correct framing:

```text
clickbait critic = risk guardrail
quality reward = editorial quality
persona signal = audience fit
objective selector = business preference
```

Do not say:

```text
Clickbait score decides the best headline.
```

Say:

```text
Clickbait risk is one component in a unified scoring system.
```

### Overlap 3: Too Many Candidate Sources

Current research candidate sources:

- original;
- zero-shot;
- round-1 rewrite;
- round-2 rewrite;
- agentic candidates;
- agentic selected;
- generic SFT;
- specificity SFT.

Simplified demo display:

| Visible Option | Meaning |
| --- | --- |
| Human baseline | Original MIND headline. |
| GenAI baseline | Direct GPT zero-shot headline. |
| Low-risk alternative | Best low-clickbait candidate from the hidden candidate pool. |
| Recommended | Persona-calibrated selection for the chosen objective. |

The hidden candidate pool can still contain all research variants. The demo only exposes the candidates that help a user make a decision.

### Overlap 4: Base Objective Selector And Persona-Calibrated Selector

The base objective selector is useful as a baseline. The persona-calibrated selector should be the product-facing selector.

Simplified framing:

```text
base objective selector = internal baseline
persona-calibrated selector = current product recommendation
```

## 3. Final Demo Workflow

The demo should use this flow:

```text
1. User selects an article.
2. User selects an objective: trust, growth, editorial, or specificity.
3. Cricli shows 3-4 candidate headlines.
4. Cricli shows a unified scorecard: Quality, Risk/Safety, Audience Fit, Objective Fit.
5. Cricli recommends one headline and explains why it wins.
6. Cricli explains why the visible alternatives were not selected.
```

The user should not need to understand round-1 rewrite, round-2 rewrite, agentic v1/v2/v3, SFT variants, or all internal reward columns.

## 4. Recommended Demo Candidate Policy

For each article and objective, show at most four options:

1. `Human baseline`
   - Original human title.
   - Useful as a reference.

2. `GenAI baseline`
   - Zero-shot GPT title.
   - Important because direct GenAI is the strongest baseline.

3. `Low-risk alternative`
   - Lowest-risk non-duplicate option from the hidden candidate pool.
   - Selected primarily by low clickbait penalty, then by predicted quality.

4. `Recommended`
   - Persona-calibrated objective selection.
   - This is the final product recommendation.

If the same headline fills multiple roles, merge the roles into one row instead of showing duplicates.

## 5. Unified Evaluator Schema

The product-facing score schema should be:

| Score | Range | Meaning |
| --- | --- | --- |
| `quality_score` | 0-1 | Local predicted editorial quality blended with summary support. |
| `risk_score` | 0-1 | Inverse of clickbait risk; higher is safer. |
| `audience_score` | 0-1 | Persona mean overall score, with neutral fallback when unavailable. |
| `objective_fit_score` | 0-1 | Normalized persona-calibrated score for the selected objective. |
| `unified_decision_score` | 0-1 | Product-facing weighted score shown in the demo. |
| `decision_explanation` | text | Why this title was selected or why it lost to the recommendation. |

This hides the complexity of raw reward dimensions while preserving the key decision signals.

## 6. Why This Simplification Helps

The simplified pipeline makes the project easier to understand:

```text
Candidate Builder: creates and filters headline options.
Unified Evaluator: scores quality, risk/safety, audience fit, and objective fit.
Selector: chooses the best headline for the current objective.
Explanation: tells the user why this title was recommended and why alternatives lost.
```

It also makes the business value clearer:

```text
GenAI can generate headlines, but Cricli helps decide which headline should be published.
```

## 7. Implementation Artifact

The simplified sample-review dataset should be generated by:

```bash
python scripts/build_headline_review_demo_cases.py
```

Expected outputs:

- `data/processed/headline_review_demo_cases.csv`
- `data/processed/headline_review_demo_profile.md`
- `data/processed/headline_review_demo_metadata.json`

The Gradio live demo can also use the full 100-seed case explorer produced by `scripts/run_product_demo.py`:

- `data/processed/headline_review_demo_cases_full.csv`
- `data/processed/headline_review_demo_cases_full_profile.md`
- `data/processed/headline_review_demo_cases_full_metadata.json`

For a real single-article review flow, run:

```bash
python scripts/review_single_article.py \
  --summary "paste article summary here" \
  --category news \
  --objective editorial
```

Expected outputs:

- `data/processed/single_article_review_candidates.csv`
- `demo/single_article_review.html`
- `data/processed/single_article_review_metadata.json`

These files are the product-facing demo inputs. The fixed sample viewer is useful for comparing historical experiments; the single-article script is the more realistic business workflow.

## 8. What To Keep Hidden In The Demo

Hide these details unless the user opens an advanced/research view:

- round-1 and round-2 rewrite labels;
- agentic v1/v2/v3 names;
- all raw pointwise reward dimensions;
- pairwise reward internals;
- SFT generator failure details;
- every generated candidate.

The report can still discuss these details. The demo should not.
