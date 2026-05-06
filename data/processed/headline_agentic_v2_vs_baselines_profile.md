# Agentic vs Baselines Local Evaluation

This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.

## Configuration

- Device: `mps`
- Clickbait weight: 1.0
- Quality weight: 1.0
- Pairwise weight: 0.25
- Output: `data/processed/headline_agentic_v2_vs_baselines_eval.csv`

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2733 | 0.2700 | 4.0225 | 0.3817 | 3.8446 | 3.9102 |
| zero_shot | 100 | 0.0879 | 0.0900 | 4.6073 | 0.8133 | 4.7228 | 4.6556 |
| round1_final | 100 | 0.0755 | 0.0700 | 4.6009 | 0.8067 | 4.7271 | 4.6466 |
| round2_final | 100 | 0.0656 | 0.0600 | 4.5908 | 0.8011 | 4.7255 | 4.6331 |
| agentic_selected | 100 | 0.0688 | 0.0700 | 4.5810 | 0.7635 | 4.7031 | 4.6167 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 0.8585 | 0.5655 | 0.9000 |
| agentic_selected - zero_shot | -0.0197 | -0.0125 | 0.3500 |
| agentic_selected - round1_final | -0.0240 | -0.0115 | 0.3800 |
| agentic_selected - round2_final | -0.0224 | -0.0121 | 0.3600 |

## Best Variant by Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| zero_shot | 58 | 0.5800 |
| agentic_selected | 32 | 0.3200 |
| original | 6 | 0.0600 |
| round1_final | 2 | 0.0200 |
| round2_final | 2 | 0.0200 |

## Top Agentic Selected Examples

| seed_id | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | lifestyle | Adam's Corner and Fisher House Support Military Families with Resources | 0.0001 | 4.7061 | 1.1074 | 4.9828 |
| 31 | autos | Duane Roots' 1,500hp Charger Hellcat Features E90 and Nitrous | 0.0002 | 4.6996 | 1.1271 | 4.9812 |
| 65 | travel | Brevard County TRIAD's Senior Santa Brings Cheer to Seniors | 0.0002 | 4.7070 | 1.0658 | 4.9733 |
| 26 | news | Sacramento Man Hosts Healdsburg Couple with 18 Dogs and Puppies | 0.0001 | 4.6991 | 1.0711 | 4.9668 |
| 39 | sports | Hawaii Man Catches 100-Pound Ulua, Fulfilling Fishermen's Dream | 0.0001 | 4.7121 | 0.9968 | 4.9612 |
| 50 | foodanddrink | Vice Officers Raid Building Linked to Design District Massage Parlor | 0.0001 | 4.6944 | 1.0558 | 4.9583 |
| 13 | travel | Game Wardens Investigate Fatal ATV Crash Near Belfast | 0.0001 | 4.6990 | 1.0005 | 4.9491 |
| 88 | sports | Finley's Performance Highlights Bengals' Search for Future Quarterback | 0.0001 | 4.6974 | 0.9971 | 4.9466 |
| 36 | health | Scientists Intentionally Infect Healthy Individuals to Study Virus Progression | 0.0003 | 4.6840 | 1.0491 | 4.9460 |
| 6 | health | Local Airman Welcomed Home by Family and Friends in Miami Valley | 0.0002 | 4.7054 | 0.9569 | 4.9444 |
| 98 | lifestyle | Wall of Faces Campaign Seeks Photos of 11 Missing Detroit Veterans | 0.0001 | 4.7003 | 0.9564 | 4.9394 |
| 42 | weather | Free Event at Lincoln Center Celebrates Climate Action Anniversary | 0.0001 | 4.6964 | 0.9713 | 4.9391 |

## Interpretation

Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.
