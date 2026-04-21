# LLM Judge Agentic Comparison

- Scores: `data/processed/headline_quality_llm_judge_agentic_v2_scores.csv`
- Scored rows: 400
- Pairwise preference examples: 334
- Pointwise reward examples: 400

## Mean Scores By Variant

| variant | faithfulness | clarity | specificity | attractiveness | non_clickbait | overall | clickbait_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3.95 | 4.33 | 3.55 | 3.51 | 4.44 | 3.75 | 0.273 |
| zero_shot | 4.89 | 4.92 | 4.65 | 4.05 | 4.98 | 4.8 | 0.088 |
| optimized | 4.86 | 4.9 | 4.62 | 4.01 | 4.98 | 4.77 | 0.066 |
| agentic_selected | 4.67 | 4.67 | 4.33 | 3.93 | 4.85 | 4.45 | 0.069 |

## Judge Winners

| variant | best_count | worst_count |
| --- | --- | --- |
| original | 25 | 59 |
| zero_shot | 62 | 1 |
| optimized | 5 | 14 |
| agentic_selected | 8 | 26 |

## Agentic Overall Deltas

| comparison | mean_overall_delta | median_overall_delta | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.700 | 1.000 | 0.540 |
| agentic_selected - zero_shot | -0.350 | 0.000 | 0.100 |
| agentic_selected - optimized | -0.320 | 0.000 | 0.120 |

## Interpretation

This is the final LLM-judge check for whether the local critic selected headlines are actually preferred when compared directly against the original, zero-shot, and round-2 optimized baselines.

## Training Use

- `headline_quality_agentic_pairwise_preferences.jsonl` can extend the pairwise reward dataset with agentic-vs-baseline preferences.
- `headline_quality_agentic_reward_model_examples.jsonl` can extend the pointwise reward critic training set with a fourth policy-output variant.
- These labels are suitable for later best-of-N reranking, reward-model retraining, or policy optimization experiments.
