# Work Summary

Project: Audience-Aware Multi-Agent Headline Optimization with Fine-Tuned Critics

Current focus: reward modeling, critic distillation, and critic-guided headline optimization.

## 1. Proposal Alignment

This work supports the proposal's core idea: turn headline generation from a one-shot generation task into a closed-loop optimization system with generation, scoring, rewriting, and learned critics.

Covered proposal components:

- Public data collection and preprocessing.
- Zero-shot API baseline generation.
- Fine-tuned small critic model for low-cost scoring.
- Critic-guided iterative rewriting.
- LLM-as-judge scoring and distillation data.
- Local reward critics for multi-dimensional scoring and pairwise preference ranking.
- Latency benchmark for local critics.

Still not covered or only partially covered:

- Few-shot baseline.
- Audience persona ensemble and voting.
- Human evaluation.
- Full UI/demo integration.
- Larger-scale ablation over optimizer, learning rate, clipping, or LoRA/QLoRA settings.

## 2. Data Prepared

Raw datasets:

- MIND small derivative from Hugging Face.
- Clickbait title classification dataset from Hugging Face.

Processed datasets:

- `projects/data/processed/mind_headline_pool_sample.csv`
  - 10,000 cleaned MIND news examples.
  - Used for headline generation and scoring.

- `projects/data/processed/clickbait_penalty_splits.csv`
  - 31,986 clickbait/non-clickbait examples.
  - Split into train/val/test.

- `projects/data/processed/headline_generation_eval_seed_100.csv`
  - 100 stratified examples across news, sports, finance, lifestyle, health, travel, food, autos, and weather.
  - Used as the fixed evaluation seed set.

- `projects/data/processed/headline_quality_pairwise_preferences.jsonl`
  - 167 LLM-judge pairwise preference examples.
  - Used for pairwise reward modeling.

- `projects/data/processed/headline_quality_reward_model_examples.jsonl`
  - 300 pointwise LLM-judge reward examples.
  - Used for multi-dimensional reward critic training.

## 3. Clickbait Penalty Critic

Two clickbait penalty models were trained:

1. Lightweight hashed n-gram logistic regression baseline.
2. Fine-tuned DistilBERT classifier.

Main model:

- Path: `projects/models/clickbait_penalty_distilbert/`
- Input: headline text.
- Output: `P(clickbait)`, used as `clickbait_penalty`.

DistilBERT test metrics:

- Accuracy: 0.9891
- Precision: 0.9912
- Recall: 0.9869
- F1: 0.9890
- ROC-AUC: 0.9988

This model was applied to the 10,000-title MIND headline pool.

MIND title scoring result:

- Mean clickbait penalty: 0.2328
- Median clickbait penalty: 0.0002
- Predicted clickbait rate: 23.15%

## 4. API Zero-Shot Baseline

API model used:

- `gpt-4o-mini`

Input:

- 100 MIND summaries from `headline_generation_eval_seed_100.csv`.

Output:

- `projects/data/processed/headline_generation_zero_shot_100.csv`

Zero-shot prompt rules:

- 6 to 14 words.
- Faithful to the summary.
- Non-clickbait.
- No question headlines.
- No unsupported facts.

Clickbait comparison:

| Stage | Mean clickbait penalty | Predicted clickbait rate |
| --- | ---: | ---: |
| Original MIND titles | 0.2688 | 27.00% |
| Zero-shot API titles | 0.0879 | 9.00% |

## 5. Critic-Guided Rewrite Loop

A local clickbait critic was used to identify generated headlines with high clickbait penalty.

Workflow:

1. Generate zero-shot headline with API.
2. Score with DistilBERT clickbait critic.
3. Rewrite high-penalty examples using critic feedback.
4. Re-score rewritten headlines.
5. Run a second stricter rewrite round for remaining high-penalty cases.

Overall result:

| Stage | Mean clickbait penalty | Predicted clickbait rate |
| --- | ---: | ---: |
| Original | 0.2688 | 27.00% |
| Zero-shot | 0.0879 | 9.00% |
| Round-1 critic-guided rewrite | 0.0755 | 7.00% |
| Round-2 critic-guided rewrite | 0.0656 | 6.00% |

Interpretation:

- Zero-shot generation already strongly reduced clickbait style.
- Critic-guided rewriting provided additional reductions.
- Later rewrite rounds showed diminishing returns.
- Some lifestyle/food titles remained high penalty, suggesting critic genre bias and the need for multi-dimensional evaluation.

## 6. LLM Judge Evaluation

An LLM judge scored three variants per seed:

- `original`
- `zero_shot`
- `optimized` / round-2 final

Dimensions:

- faithfulness
- clarity
- specificity
- attractiveness
- non_clickbait
- overall

Mean scores:

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 4.090 | 4.420 | 3.610 | 3.530 | 4.480 | 3.830 | 0.2688 |
| zero_shot | 4.880 | 4.950 | 4.610 | 4.120 | 4.960 | 4.850 | 0.0879 |
| optimized | 4.860 | 4.940 | 4.560 | 4.070 | 4.960 | 4.820 | 0.0656 |

Winner counts:

- zero_shot: 71
- original: 24
- optimized: 5

Key finding:

- The optimized version had the lowest clickbait penalty, but the zero-shot version had slightly higher LLM-judge overall quality.
- This shows a trade-off between style safety and headline quality/specificity.

## 7. Multi-Dimensional Reward Critic

Model:

- Path: `projects/models/headline_quality_reward_distilbert/`
- DistilBERT multi-output regression critic.

Input:

- category
- summary
- headline

Outputs:

- faithfulness
- clarity
- specificity
- attractiveness
- non_clickbait
- overall

Training data:

- 300 pointwise LLM-judge examples.
- Split by `seed_id` to avoid leakage.
- Train/val/test rows: 210/45/45.

Test metrics on 1-5 score scale:

| Dimension | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| faithfulness | 0.524 | 0.621 | -0.046 |
| clarity | 0.391 | 0.460 | -0.222 |
| specificity | 0.715 | 0.850 | 0.021 |
| attractiveness | 0.404 | 0.537 | -0.309 |
| non_clickbait | 0.374 | 0.452 | -0.277 |
| overall | 0.544 | 0.671 | 0.046 |

Macro MAE:

- 0.492

Interpretation:

- This is a proof-of-concept local reward critic distilled from LLM judge labels.
- Dataset size is small, so the model should not be treated as production-ready.
- The result supports the feasibility of LLM-judge-to-small-critic distillation.

## 8. Pairwise Reward Critic

Model:

- Path: `projects/models/headline_pairwise_reward_distilbert/`
- DistilBERT pairwise reward scorer.

Training objective:

```text
loss = -log sigmoid(score(chosen) - score(rejected))
```

Training data:

- 167 LLM-judge pairwise preference examples.
- Train/val/test pairs: 117/24/26.

Test metrics:

- Accuracy: 0.846
- Symmetric AUC: 0.846
- Mean margin: 0.020
- Median margin: 0.015

Interpretation:

- The model learned a weak but meaningful preference signal.
- More preference pairs are needed for stronger margins and robust generalization.
- This is the most directly RLHF/RLAIF-like component so far.

## 9. Local Critic Latency Benchmark

Benchmark output:

- `projects/data/processed/critic_latency_benchmark.md`
- `projects/data/processed/critic_latency_benchmark.json`

CPU benchmark:

| Critic | Examples/sec | ms/example |
| --- | ---: | ---: |
| clickbait penalty critic | 360.36 | 2.78 |
| multi-dimensional quality critic | 67.17 | 14.89 |
| pairwise reward critic | 33.13 | 30.18 |

Interpretation:

- Local critics can score examples with millisecond-level latency and no API calls.
- This directly supports the proposal's low-cost/low-latency critic motivation.

## 10. Current Status

Completed:

- Data collection and processing.
- Clickbait critic.
- Zero-shot API baseline.
- Critic-guided rewrite loop.
- LLM judge scoring.
- Reward model training data export.
- Multi-dimensional reward critic.
- Pairwise reward critic.
- Local latency benchmark.

Recommended next steps:

1. Candidate generation and reranking:
   - API generates 5 headlines per summary.
   - Local critics score each candidate.
   - Select highest reward candidate.

2. Few-shot baseline:
   - Compare against zero-shot and critic-guided methods.

3. Persona ensemble:
   - Add audience-specific judge/persona voting.

4. Human evaluation:
   - Small pairwise evaluation on 20-30 examples.

5. Demo:
   - Streamlit or Gradio interface.

