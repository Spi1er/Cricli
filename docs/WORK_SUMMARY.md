# Work Summary

Project: Audience-Aware Headline Optimization with Local Critics and Reward-Model Iteration

Current status: the project is re-centered around the original proposal direction: audience-aware headline optimization with local critics, LLM-as-judge labels, reward-model distillation, persona/audience evaluation, and objective-specific headline selection.

The project should not be framed as a full autonomous agent or online RL system. It is best described as a proposal-aligned, RLHF-inspired workflow:

```text
candidate headline generation
-> multi-dimensional evaluation
-> audience/persona preference estimation
-> small critic / reward model distillation
-> objective-specific selection
-> reward and evaluator misalignment analysis
```

The FLAN-T5 SFT generator experiments are retained as auxiliary candidate-source experiments. They are not the main project objective.

## 1. Project Positioning

This is not an industrial-scale deployment. The reward-model data is still small, so the results should be interpreted as a proof of concept rather than a production-ready optimizer.

The main value is that the workflow is realistic:

```text
policy output
-> LLM judge labels
-> local reward critic training
-> best-of-N / rejection-sampling style reranking
-> evaluation
-> reward-model update
-> improved candidate generation
-> error analysis
```

In post-training language, this project demonstrates:

- supervised critic training;
- LLM-as-judge reward labeling;
- pointwise reward modeling;
- pairwise preference modeling;
- reward-guided inference-time optimization;
- iterative reward-model improvement;
- reward misalignment analysis.

## 2. Main Data Assets

Raw data:

- MIND small derivative from Hugging Face.
- Clickbait title classification dataset from Hugging Face.

Processed data:

- `data/processed/clickbait_penalty_splits.csv`
  - 31,986 clickbait/non-clickbait examples.
  - Used to train the clickbait penalty critic.

- `data/processed/mind_headline_pool_sample.csv`
  - 10,000 cleaned MIND headline examples.
  - Used for scoring and seed sampling.

- `data/processed/headline_generation_eval_seed_100.csv`
  - 100 stratified MIND examples.
  - Used as the fixed evaluation set.

- `data/processed/headline_quality_reward_model_examples.jsonl`
  - 300 original 3-way LLM-judge pointwise examples.

- `data/processed/headline_quality_pairwise_preferences.jsonl`
  - 167 original 3-way LLM-judge pairwise examples.

- `data/processed/headline_quality_reward_model_examples_v2.jsonl`
  - 700 merged pointwise reward examples.
  - Combines original 3-way judge data and agentic 4-way judge data.

- `data/processed/headline_quality_pairwise_preferences_v2.jsonl`
  - 503 merged pairwise preference examples.

## 3. Clickbait Penalty Critic

Main model:

- `models/clickbait_penalty_distilbert`
- Architecture: DistilBERT sequence classifier.
- Input: headline.
- Output: clickbait probability, used as `clickbait_penalty`.

Test metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9891 |
| Precision | 0.9900 |
| Recall | 0.9881 |
| F1 | 0.9890 |
| ROC-AUC | 0.9988 |

MIND headline pool scoring:

| Metric | Value |
| --- | ---: |
| Mean clickbait penalty | 0.2328 |
| Median clickbait penalty | 0.0002 |
| Predicted clickbait rate | 23.15% |

This critic is the most mature model in the project because it was trained on roughly 32k labeled examples.

## 4. Zero-Shot Baseline

Generator:

- `gpt-4o-mini`

Workflow:

```text
summary -> gpt-4o-mini -> one headline
```

Prompt requirements:

- concise;
- faithful to the summary;
- non-clickbait;
- no question headlines;
- no unsupported facts.

Result:

| Variant | Clickbait penalty | Clickbait rate |
| --- | ---: | ---: |
| Original MIND title | 0.2688 | 27% |
| Zero-shot API title | 0.0891 | 9% |

The zero-shot baseline is very strong. It remains the hardest system to beat in average LLM-judge overall quality.

## 5. Clickbait Critic-Guided Rewrite

Generator / rewriter:

- `gpt-4o-mini`

Critic:

- `models/clickbait_penalty_distilbert`

Workflow:

```text
zero-shot headline
-> score with clickbait critic
-> rewrite high-penalty headlines
-> score again
-> round-2 rewrite for remaining high-penalty cases
```

Result:

| Stage | Mean clickbait penalty | Clickbait rate |
| --- | ---: | ---: |
| Original | 0.2688 | 27% |
| Zero-shot | 0.0891 | 9% |
| Round-1 final | 0.0814 | 8% |
| Round-2 final / optimized | 0.0585 | 5% |

Interpretation:

- Critic-guided rewriting lowered clickbait beyond zero-shot.
- However, LLM judge later showed that lower clickbait did not always mean better overall headline quality.

## 6. Final LLM Judge Evaluation

Judge:

- `gpt-4o-mini`

Compared variants in the final v3 judge run:

- `original`
- `zero_shot`
- `optimized` / round-2 final
- `agentic_v3`

Judge dimensions:

- faithfulness;
- clarity;
- specificity;
- attractiveness;
- non_clickbait;
- overall.

Mean LLM-judge scores:

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 4.08 | 4.39 | 3.64 | 3.57 | 4.43 | 3.81 | 0.281 |
| zero_shot | 4.87 | 4.94 | 4.57 | 4.02 | 4.99 | 4.77 | 0.087 |
| optimized | 4.84 | 4.93 | 4.53 | 3.98 | 4.98 | 4.73 | 0.060 |
| agentic_v3 | 4.73 | 4.75 | 4.40 | 4.02 | 4.94 | 4.50 | 0.063 |

Winner counts:

| Variant | Best count |
| --- | ---: |
| zero_shot | 60 |
| original | 26 |
| agentic_v3 | 13 |
| optimized | 1 |

Key finding:

The optimized rewrite had the lowest clickbait penalty, but zero-shot had slightly better overall quality. This introduced the main project tension: reducing clickbait alone is not enough.

## 7. Reward Model v1

Two local reward critics were trained from the original LLM-judge labels.

### Pointwise Reward Critic v1

Model:

- `models/headline_quality_reward_distilbert`
- DistilBERT multi-output regressor.

Input:

```text
category + summary + headline
```

Output:

```text
faithfulness, clarity, specificity, attractiveness, non_clickbait, overall
```

Training data:

- 300 pointwise LLM-judge examples.

Test metrics:

| Metric | Value |
| --- | ---: |
| Test macro MAE | 0.4686 |
| Test overall MAE | 0.5188 |

### Pairwise Reward Critic v1

Model:

- `models/headline_pairwise_reward_distilbert`
- DistilBERT pairwise reward scorer.

Objective:

```text
loss = -log sigmoid(score(chosen) - score(rejected))
```

Training data:

- 167 pairwise preference examples.

Test metrics:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.6923 |
| Symmetric AUC | 0.8107 |

## 8. Agentic Selection v1

Models:

- Generator: `gpt-4o-mini`
- Clickbait critic: `models/clickbait_penalty_distilbert`
- Quality reward critic: `models/headline_quality_reward_distilbert`
- Pairwise reward critic: `models/headline_pairwise_reward_distilbert`

Workflow:

```text
summary
-> generate K candidates with gpt-4o-mini
-> score candidates with local critics
-> choose highest reward candidate
```

Candidate count:

- 100 seed examples.
- 298 generated candidate rows.
- 100 selected headlines.

Selection formula:

```text
final_score =
  1.0 * quality_reward
+ 0.25 * pairwise_reward
- 1.0 * clickbait_penalty
```

LLM judge result:

| Metric | Agentic v1 |
| --- | ---: |
| Overall | 4.33 |
| Best count | 2 |
| Delta vs original | +0.50 |
| Delta vs zero-shot | -0.43 |
| Delta vs optimized | -0.38 |
| Clickbait penalty | 0.068 |

Interpretation:

Agentic v1 improved over original, but underperformed zero-shot and optimized. The main issue was not clickbait; it was faithfulness, clarity, and specificity.

## 9. Reward Model v2

Agentic v1 outputs were judged in a new 4-way comparison:

- `original`
- `zero_shot`
- `optimized`
- `agentic_selected`

The new judge data was merged with the old judge data.

Merged v2 training data:

| Dataset | Rows |
| --- | ---: |
| Pointwise reward examples | 700 |
| Pairwise preference examples | 503 |

### Pointwise Reward Critic v2

Model:

- `models/headline_quality_reward_distilbert_v2`

Test metrics:

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Test macro MAE | 0.4686 | 0.4546 |
| Faithfulness MAE | 0.5205 | 0.4466 |
| Clarity MAE | 0.3732 | 0.3379 |
| Non-clickbait MAE | 0.3533 | 0.2611 |

The pointwise v2 critic improved overall alignment to judge labels.

### Pairwise Reward Critic v2

Model:

- `models/headline_pairwise_reward_distilbert_v2`

Test metrics:

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Test accuracy | 0.6923 | 0.8193 |
| Symmetric AUC | 0.8107 | 0.8379 |

The pairwise v2 critic improved in the latest local rerun after adding more agentic-vs-baseline comparisons.

## 10. Agentic Selection v2

No new API generation was used.

Workflow:

```text
reuse v1 candidates
-> rescore with v2 reward critics
-> rerank
```

LLM judge result:

| Metric | Agentic v1 | Agentic v2 |
| --- | ---: | ---: |
| Overall | 4.33 | 4.45 |
| Best count | 2 | 8 |
| Delta vs original | +0.50 | +0.70 |
| Delta vs zero-shot | -0.43 | -0.35 |
| Delta vs optimized | -0.38 | -0.32 |

Interpretation:

Reward-model updating helped. It narrowed the gap to zero-shot and optimized, but did not beat them.

## 11. Agentic Selection v3

Agentic v3 targeted the main failure mode from v1/v2: weak specificity, clarity, and faithfulness.

Changes:

- New candidate generation prompt: `--prompt-style specificity`
- New reward preset: `--reward-preset faithfulness_specificity`
- New weights:

```text
quality_weight = 1.3
pairwise_weight = 0.4
clickbait_weight = 0.5
```

The v3 prompt asked the API generator to produce candidates with different editorial emphases:

1. most faithful;
2. most specific;
3. clearest general-news style;
4. most engaging without clickbait;
5. best balanced editor headline.

Local critic result:

| Variant | Clickbait penalty | Clickbait rate | Quality reward | Pairwise reward | Final score |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero_shot | 0.0872 | 8% | 4.6279 | 0.8133 | 6.3643 |
| optimized | 0.0603 | 5% | 4.6156 | 0.8045 | 6.3573 |
| agentic_v3 | 0.0626 | 6% | 4.6185 | 0.8033 | 6.3582 |

Local evaluation treats zero-shot, optimized, and agentic v3 as very close. Zero-shot has the highest mean local final score in the latest rerun, while agentic v3 remains a controllable low-risk selected candidate source.

LLM judge result:

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 4.08 | 4.39 | 3.64 | 3.57 | 4.43 | 3.81 | 0.281 |
| zero_shot | 4.87 | 4.94 | 4.57 | 4.02 | 4.99 | 4.77 | 0.087 |
| optimized | 4.84 | 4.93 | 4.53 | 3.98 | 4.98 | 4.73 | 0.060 |
| agentic_v3 | 4.73 | 4.75 | 4.40 | 4.02 | 4.94 | 4.50 | 0.063 |

LLM judge progression:

| Metric | Agentic v1 | Agentic v2 | Agentic v3 |
| --- | ---: | ---: | ---: |
| Overall | 4.33 | 4.45 | 4.50 |
| Best count | 2 | 8 | 13 |
| Delta vs zero-shot | -0.43 | -0.35 | -0.27 |
| Delta vs optimized | -0.38 | -0.32 | -0.23 |

Interpretation:

Agentic v3 still does not beat zero-shot on average LLM-judge overall quality, but it remains stronger than the earlier agentic runs and provides an explicit reward-selected candidate. The current rerun shows the tradeoff clearly: agentic selection is controllable, but direct GenAI remains the strongest average generator.

## 12. Error Analysis

Error analysis compared agentic v3 against zero-shot using both local reward and LLM judge results.

Output:

- `data/processed/headline_agentic_v3_error_analysis.csv`
- `data/processed/headline_agentic_v3_error_analysis.md`

Case counts:

| Case type | Count |
| --- | ---: |
| tie_or_mixed | 53 |
| zero_shot_beats_agentic | 20 |
| local_reward_overestimates_agentic | 16 |
| agentic_beats_zero_shot | 9 |
| local_reward_underestimates_agentic | 2 |

Mean deltas:

| Metric | Value |
| --- | ---: |
| LLM judge delta, agentic - zero-shot | -0.27 |
| Local reward delta, agentic - zero-shot | -0.006 |

This shows reward misalignment in magnitude: the local reward model sees agentic v3 and zero-shot as nearly tied, but the LLM judge still prefers zero-shot on average.

Main dimensions where agentic loses to zero-shot:

| Dimension | Loss count |
| --- | ---: |
| specificity | 28 |
| clarity | 21 |
| attractiveness | 20 |
| faithfulness | 20 |
| non_clickbait | 5 |

Key error pattern:

The local reward model sometimes overvalues concrete named entities while undervaluing broader context completeness.

Example:

```text
zero_shot:
Celtics' Javonte Green steps up during close game after Hayward's injury

agentic:
Brad Stevens Surprises with Javonte Green's Minutes Against Mavericks
```

The agentic headline includes specific names, but loses the more important causal context: Hayward's injury and Green stepping up.

Another example:

```text
zero_shot:
Fosun Acquires Thomas Cook Brand for $14.2 Million Following Bankruptcy

agentic:
Fosun Acquires Thomas Cook Brand for $14.2 Million
```

The agentic headline is clean, but drops the bankruptcy context.

## 13. Current Main Conclusions

1. Zero-shot GPT is a very strong baseline.

2. The clickbait critic is effective and reliable.

3. Clickbait reduction alone does not guarantee better headline quality.

4. Local reward critics can guide best-of-N selection, but are sensitive to reward misalignment.

5. Reward-model iteration helps:

```text
Agentic overall: 4.33 -> 4.45 -> 4.50
Best count:      2 -> 8 -> 13
Gap to zero-shot: -0.43 -> -0.35 -> -0.27
```

6. Candidate generation quality matters as much as reward modeling. V3 improved mainly because the candidate prompt produced more specific and useful options.

7. The remaining bottleneck is source-grounded specificity: specific details help only when they preserve the central factual context.

## 14. Industrial-Scale Gaps

Compared with an industrial post-training or RLHF pipeline, this project is missing or only lightly covers:

- larger-scale prompt/output sampling;
- human preference labels;
- judge calibration and inter-annotator agreement;
- reward model calibration;
- reward hacking detection beyond the current error analysis;
- generator fine-tuning with SFT/DPO/RL;
- online feedback such as CTR or dwell time;
- A/B testing;
- production monitoring and drift detection;
- factuality verification beyond LLM judge faithfulness scores;
- full audience persona modeling.

The current system is best framed as:

```text
Small data, real workflow.
```

## 15. Reproducibility Status

The latest local rerun completed both implementation lines:

1. Reward / post-training line:
   - OpenAI candidate generation and judging.
   - DistilBERT clickbait, quality reward, and pairwise reward critics.
   - v2 reward data and v2 critic retraining.
   - Agentic v3 specificity selection.
   - Persona voting and persona-calibrated objective selection.

2. SFT generator line:
   - Generic FLAN-T5-small SFT generator.
   - Specificity-aware FLAN-T5-small SFT generator.
   - Local critic evaluation.
   - LLM judge comparison and error analysis.

Exact generated headline strings are not expected to be bit-for-bit identical across reruns because OpenAI generation/judging and local SFT training are stochastic. The reproducibility target is the full workflow, tracked artifacts, fixed 100-example seed set, and aggregate metrics.

## 16. Recommended Next Steps

Highest-value next steps:

1. Write the final report and presentation around the v1 -> v2 -> v3 progression.

2. Add a concise experiment table comparing:

```text
original
zero_shot
optimized / round2
agentic v1
agentic v2
agentic v3
```

3. Add error-analysis examples to explain why v3 still trails zero-shot.

4. If more experiments are needed, improve source-grounded candidate generation:

```text
Generate specific headlines only using details explicitly present in the summary.
Before outputting each headline, silently verify that every named entity, number, and event appears in the summary.
```

5. For a more advanced extension, train a small generator with SFT or DPO on preferred headlines. This would move the project from inference-time reranking toward actual policy optimization.

## 17. Files Most Relevant For Report

Core scripts:

- `scripts/run_zero_shot_headline_generation.py`
- `scripts/train_clickbait_penalty_bert.py`
- `scripts/score_headline_clickbait_penalty.py`
- `scripts/run_critic_guided_rewrite.py`
- `scripts/run_critic_guided_rewrite_round2.py`
- `scripts/run_llm_judge_headline_quality.py`
- `scripts/train_headline_quality_reward_critic.py`
- `scripts/train_headline_pairwise_reward_critic.py`
- `scripts/run_agentic_headline_optimizer.py`
- `scripts/build_reward_training_v2.py`
- `scripts/rerank_agentic_candidates.py`
- `scripts/run_llm_judge_agentic_comparison.py`
- `scripts/evaluate_agentic_vs_baselines.py`
- `scripts/analyze_agentic_v3_errors.py`

Core outputs:

- `data/processed/headline_generation_zero_shot_scored_100.csv`
- `data/processed/headline_generation_rewrite_round2_critic_guided_scored_100.csv`
- `data/processed/headline_quality_llm_judge_scores.csv`
- `data/processed/headline_quality_llm_judge_agentic_scores.csv`
- `data/processed/headline_quality_llm_judge_agentic_v2_scores.csv`
- `data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv`
- `data/processed/headline_agentic_v3_error_analysis.md`

Core models:

- `models/clickbait_penalty_distilbert`
- `models/headline_quality_reward_distilbert`
- `models/headline_pairwise_reward_distilbert`
- `models/headline_quality_reward_distilbert_v2`
- `models/headline_pairwise_reward_distilbert_v2`

Model weights are intentionally not tracked in GitHub because they are large.
