# LLM Judge Headline Quality Profile

- Scores: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/headline_quality_llm_judge_scores.csv`
- Scored rows: 300
- Pairwise preference examples: 167
- Pointwise reward examples: 300

## Mean Scores By Variant

| Variant | Faithfulness | Clarity | Specificity | Attractiveness | Non-clickbait | Overall | Clickbait penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 4.090 | 4.420 | 3.610 | 3.530 | 4.480 | 3.830 | 0.2688 |
| zero_shot | 4.880 | 4.950 | 4.610 | 4.120 | 4.960 | 4.850 | 0.0879 |
| optimized | 4.860 | 4.940 | 4.560 | 4.070 | 4.960 | 4.820 | 0.0656 |

## Judge Winners

| Variant | Best count | Worst count |
| --- | ---: | ---: |
| original | 24 | 75 |
| zero_shot | 71 | 11 |
| optimized | 5 | 14 |

## Overall Score Deltas

- Optimized vs zero-shot mean overall delta: -0.030
- Optimized vs original mean overall delta: 0.990
- Zero-shot vs original mean overall delta: 1.020

## Training Use

- `headline_quality_pairwise_preferences.jsonl` can train a pairwise reward model with a RankNet/DPO-style loss.
- `headline_quality_reward_model_examples.jsonl` can train a pointwise multi-head critic for faithfulness, clarity, specificity, attractiveness, non-clickbait, and overall quality.
- Combine these labels with the DistilBERT clickbait penalty to form a multi-objective reward.
