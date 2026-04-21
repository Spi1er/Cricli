# LLM Judge Agentic Comparison

- Scores: `data/processed/headline_quality_llm_judge_agentic_v3_specificity_scores.csv`
- Scored rows: 400
- Pairwise preference examples: 348
- Pointwise reward examples: 400

## Mean Scores By Variant

| variant | faithfulness | clarity | specificity | attractiveness | non_clickbait | overall | clickbait_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3.96 | 4.34 | 3.63 | 3.53 | 4.38 | 3.74 | 0.273 |
| zero_shot | 4.82 | 4.91 | 4.6 | 4.0 | 4.98 | 4.71 | 0.088 |
| optimized | 4.78 | 4.89 | 4.56 | 3.98 | 4.97 | 4.68 | 0.066 |
| agentic_selected | 4.73 | 4.71 | 4.52 | 4.03 | 4.9 | 4.49 | 0.053 |

## Judge Winners

| variant | best_count | worst_count |
| --- | --- | --- |
| original | 25 | 51 |
| zero_shot | 56 | 3 |
| optimized | 2 | 23 |
| agentic_selected | 17 | 23 |

## Agentic Overall Deltas

| comparison | mean_overall_delta | median_overall_delta | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.750 | 1.000 | 0.540 |
| agentic_selected - zero_shot | -0.220 | 0.000 | 0.190 |
| agentic_selected - optimized | -0.190 | 0.000 | 0.210 |

## Interpretation

This is the final LLM-judge check for whether the local critic selected headlines are actually preferred when compared directly against the original, zero-shot, and round-2 optimized baselines.

## Training Use

- `headline_quality_agentic_pairwise_preferences.jsonl` can extend the pairwise reward dataset with agentic-vs-baseline preferences.
- `headline_quality_agentic_reward_model_examples.jsonl` can extend the pointwise reward critic training set with a fourth policy-output variant.
- These labels are suitable for later best-of-N reranking, reward-model retraining, or policy optimization experiments.
