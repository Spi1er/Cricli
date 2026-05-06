# Agentic vs Baselines Local Evaluation

This report re-scores all variants with the same local critics: clickbait penalty, multi-dimensional quality reward, and pairwise reward.

## Configuration

- Device: `mps`
- Clickbait weight: 0.5
- Quality weight: 1.3
- Pairwise weight: 0.4
- Reward preset: `faithfulness_specificity`
- Output: `data/processed/headline_agentic_v3_specificity_vs_baselines_eval.csv`

## Variant Summary

| variant | rows | mean_clickbait_penalty | clickbait_rate | mean_quality_reward | mean_pairwise_reward | mean_final_score | mean_pred_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.2733 | 0.2700 | 4.0225 | 0.3817 | 5.2464 | 3.9102 |
| zero_shot | 100 | 0.0879 | 0.0900 | 4.6073 | 0.8133 | 6.3344 | 4.6556 |
| round1_final | 100 | 0.0755 | 0.0700 | 4.6009 | 0.8067 | 6.3290 | 4.6466 |
| round2_final | 100 | 0.0656 | 0.0600 | 4.5908 | 0.8011 | 6.3176 | 4.6331 |
| agentic_selected | 100 | 0.0535 | 0.0500 | 4.6087 | 0.8191 | 6.3552 | 4.6572 |

## Paired Final-Score Deltas

| comparison | mean_delta_final_score | median_delta_final_score | agentic_win_rate |
| --- | --- | --- | --- |
| agentic_selected - original | 1.1088 | 0.8386 | 0.9400 |
| agentic_selected - zero_shot | 0.0208 | 0.0025 | 0.5100 |
| agentic_selected - round1_final | 0.0262 | 0.0065 | 0.5200 |
| agentic_selected - round2_final | 0.0376 | 0.0002 | 0.5000 |

## Best Variant by Local Final Score

| variant | best_count | best_rate |
| --- | --- | --- |
| zero_shot | 48 | 0.4800 |
| agentic_selected | 43 | 0.4300 |
| original | 5 | 0.0500 |
| round1_final | 2 | 0.0200 |
| round2_final | 2 | 0.0200 |

## Top Agentic Selected Examples

| seed_id | category | headline | clickbait_penalty | quality_reward | pairwise_reward | final_score |
| --- | --- | --- | --- | --- | --- | --- |
| 65 | travel | Brevard County TRIAD's Senior Santa Program Delivers Holiday Cheer to Seniors | 0.0002 | 4.7085 | 1.1395 | 6.6454 |
| 25 | lifestyle | Adam's Corner and Fisher House Support Military Families with Housing and Child Care | 0.0002 | 4.7055 | 1.1241 | 6.6342 |
| 31 | autos | Duane Roots' 1,500hp Charger Hellcat Features E90 and Nitrous Setup | 0.0002 | 4.6961 | 1.1499 | 6.6336 |
| 60 | news | Sky Valley Education Center Teachers Evacuate Students After Light Fire | 0.0001 | 4.6940 | 1.1075 | 6.6168 |
| 62 | news | Dr. Ulrich Klopfer's Family Discovers 2,246 Preserved Fetuses in Illinois Home | 0.0002 | 4.6999 | 1.0468 | 6.6036 |
| 6 | health | Local Airman Returns Home to Miami Valley on Friday | 0.0001 | 4.7090 | 1.0259 | 6.5994 |
| 42 | weather | Lincoln Center Hosts Free Event for Fort Collins' Climate Action Anniversary | 0.0001 | 4.7024 | 1.0274 | 6.5978 |
| 13 | travel | Game Wardens Investigate Fatal ATV Crash in Montville, West of Belfast | 0.0001 | 4.7032 | 1.0281 | 6.5967 |
| 76 | news | Steny Hoyer Discusses Impeachment Decision After Hearing with Diplomats | 0.0001 | 4.6801 | 1.0724 | 6.5943 |
| 50 | foodanddrink | Vice Officers Storm Building Linked to Design District Massage Parlor Raid | 0.0001 | 4.6913 | 1.0466 | 6.5940 |
| 39 | sports | Hawaii Man Catches 100-Pound Ulua, Fulfilling Fishermen's Dream | 0.0001 | 4.7121 | 0.9968 | 6.5921 |
| 99 | news | Missouri Auditor Nicole Galloway Issues Subpoena to Clay County Officials | 0.0001 | 4.6901 | 1.0436 | 6.5920 |

## Interpretation

Use this as a local reward-model evaluation, not as the final human-quality verdict. The next step is to run the LLM judge on `agentic_selected` and compare those judge scores against the earlier original / zero-shot / optimized variants.
