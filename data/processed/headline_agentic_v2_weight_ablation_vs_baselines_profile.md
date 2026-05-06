# Agentic vs Baselines Local Evaluation

This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.

## Configuration

- Device: `mps`
- Clickbait weight: 0.5
- Quality weight: 1.3
- Pairwise weight: 0.4
- Reward preset: `faithfulness_specificity`
- Output: `data/processed/headline_agentic_v2_weight_ablation_vs_baselines_eval.csv`

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2733 | 0.2700 | 4.0225 | 0.3817 | 5.2464 | 3.9102 |
| zero_shot | 100 | 0.0879 | 0.0900 | 4.6073 | 0.8133 | 6.3344 | 4.6556 |
| round1_final | 100 | 0.0755 | 0.0700 | 4.6009 | 0.8067 | 6.3290 | 4.6466 |
| round2_final | 100 | 0.0656 | 0.0600 | 4.5908 | 0.8011 | 6.3176 | 4.6331 |
| agentic_selected | 100 | 0.0786 | 0.0800 | 4.5840 | 0.7673 | 6.2874 | 4.6196 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 1.0410 | 0.7142 | 0.8900 |
| agentic_selected - zero_shot | -0.0470 | -0.0196 | 0.3500 |
| agentic_selected - round1_final | -0.0416 | -0.0173 | 0.3700 |
| agentic_selected - round2_final | -0.0302 | -0.0196 | 0.3500 |

## Best Variant by Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| zero_shot | 60 | 0.6000 |
| agentic_selected | 31 | 0.3100 |
| original | 5 | 0.0500 |
| round1_final | 2 | 0.0200 |
| round2_final | 2 | 0.0200 |

## Top Agentic Selected Examples

| seed_id | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| 31 | autos | Duane Roots' 1,500hp Charger Hellcat Features E90 and Nitrous | 0.0002 | 4.6996 | 1.1271 | 6.6284 |
| 25 | lifestyle | Adam's Corner and Fisher House Support Military Families with Resources | 0.0001 | 4.7061 | 1.1074 | 6.6257 |
| 65 | travel | Brevard County TRIAD's Senior Santa Brings Cheer to Seniors | 0.0002 | 4.7070 | 1.0658 | 6.6130 |
| 26 | news | Sacramento Man Hosts Healdsburg Couple with 18 Dogs and Puppies | 0.0001 | 4.6991 | 1.0711 | 6.6120 |
| 50 | foodanddrink | Vice Officers Raid Building Linked to Design District Massage Parlor | 0.0001 | 4.6944 | 1.0558 | 6.6006 |
| 39 | sports | Hawaii Man Catches 100-Pound Ulua, Fulfilling Fishermen's Dream | 0.0001 | 4.7121 | 0.9968 | 6.5921 |
| 13 | travel | Game Wardens Investigate Fatal ATV Crash Near Belfast | 0.0001 | 4.6990 | 1.0005 | 6.5803 |
| 88 | sports | Finley's Performance Highlights Bengals' Search for Future Quarterback | 0.0001 | 4.6974 | 0.9971 | 6.5795 |
| 36 | health | Scientists Intentionally Infect Healthy Individuals to Study Virus Progression | 0.0003 | 4.6840 | 1.0491 | 6.5732 |
| 42 | weather | Free Event at Lincoln Center Celebrates Climate Action Anniversary | 0.0001 | 4.6964 | 0.9713 | 6.5692 |
| 6 | health | Local Airman Welcomed Home by Family and Friends in Miami Valley | 0.0002 | 4.7054 | 0.9569 | 6.5689 |
| 98 | lifestyle | Wall of Faces Campaign Seeks Photos of 11 Missing Detroit Veterans | 0.0001 | 4.7003 | 0.9564 | 6.5659 |

## Interpretation

Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.
