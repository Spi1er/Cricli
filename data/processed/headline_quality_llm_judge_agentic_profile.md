# LLM Judge Agentic Comparison

- Scores: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_quality_llm_judge_agentic_scores.csv`
- Scored rows: 400
- Pairwise preference examples: 336
- Pointwise reward examples: 400

## Mean Scores By Variant

| variant | faithfulness | clarity | specificity | attractiveness | non_clickbait | overall | clickbait_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 4.07 | 4.36 | 3.66 | 3.55 | 4.49 | 3.83 | 0.273 |
| zero_shot | 4.88 | 4.94 | 4.64 | 4.06 | 4.96 | 4.76 | 0.088 |
| optimized | 4.86 | 4.93 | 4.58 | 4.01 | 4.96 | 4.71 | 0.066 |
| agentic_selected | 4.68 | 4.67 | 4.28 | 3.98 | 4.88 | 4.33 | 0.068 |

## Judge Winners

| variant | best_count | worst_count |
| --- | --- | --- |
| original | 28 | 54 |
| zero_shot | 70 | 2 |
| optimized | 0 | 12 |
| agentic_selected | 2 | 32 |

## Agentic Overall Deltas

| comparison | mean_overall_delta | median_overall_delta | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.500 | 0.000 | 0.470 |
| agentic_selected - zero_shot | -0.430 | 0.000 | 0.050 |
| agentic_selected - optimized | -0.380 | 0.000 | 0.070 |

## Interpretation

This is the final LLM-judge check for whether the local critic selected headlines are actually preferred when compared directly against the original, zero-shot, and round-2 optimized baselines.

## Training Use

- `headline_quality_agentic_pairwise_preferences.jsonl` can extend the pairwise reward dataset with agentic-vs-baseline preferences.
- `headline_quality_agentic_reward_model_examples.jsonl` can extend the pointwise reward critic training set with a fourth policy-output variant.
- These labels are suitable for later best-of-N reranking, reward-model retraining, or policy optimization experiments.
