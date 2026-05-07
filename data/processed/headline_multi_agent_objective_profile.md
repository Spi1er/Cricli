# Multi-Agent Objective Headline Matrix

This report reframes the project as an offline agentic RL-style selection system.

## Agentic Framing

- State: article summary, category, and context.
- Action: choose a candidate headline from multiple generator agents.
- Reward vector: local critic scores, clickbait penalty, pairwise reward, style heuristics, and retrospective LLM judge labels when available.
- Policy: objective-specific selector that scalarizes the reward vector differently for trust, growth, editorial, or specificity goals.

## Files

- Candidate matrix: `data/processed/headline_multi_agent_candidate_matrix.csv`
- Objective selections: `data/processed/headline_multi_agent_objective_selection.csv`

## Candidate Sources

| variant | candidate_source | rows | mean_clickbait_penalty | mean_faithfulness | mean_specificity | mean_attractiveness | mean_local_final | mean_llm_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agentic_candidate | agentic_generator_v3_candidate | 300 | 0.095 | 4.693 | 4.377 | 3.968 | 6.168 |  |
| agentic_selected | agentic_selector_v3 | 100 | 0.063 | 4.790 | 4.548 | 4.073 | 6.358 | 4.500 |
| generic_sft | generic_sft_generator | 100 | 0.226 | 4.755 | 4.518 | 4.055 | 4.680 | 3.780 |
| original | human_editor | 100 | 0.281 | 4.126 | 3.699 | 3.599 | 5.255 | 3.810 |
| round1_final | critic_guided_rewriter_round1 | 100 | 0.071 | 4.791 | 4.562 | 4.074 | 6.359 |  |
| round2_final | critic_guided_rewriter_round2 | 100 | 0.060 | 4.786 | 4.554 | 4.069 | 6.357 | 4.730 |
| specificity_sft | specificity_sft_generator | 100 | 0.211 | 4.785 | 4.549 | 4.073 | 4.713 | 3.860 |
| zero_shot | api_zero_shot_generator | 100 | 0.087 | 4.797 | 4.575 | 4.083 | 6.364 | 4.770 |

## Objective Selection Counts

| objective | selected_variant | selected_count |
| --- | --- | --- |
| editorial | zero_shot | 39 |
| editorial | agentic_selected | 24 |
| editorial | agentic_candidate | 18 |
| editorial | generic_sft | 9 |
| editorial | specificity_sft | 5 |
| editorial | original | 2 |
| editorial | round1_final | 2 |
| editorial | round2_final | 1 |
| growth | zero_shot | 36 |
| growth | agentic_selected | 19 |
| growth | agentic_candidate | 18 |
| growth | generic_sft | 16 |
| growth | specificity_sft | 7 |
| growth | original | 2 |
| growth | round1_final | 1 |
| growth | round2_final | 1 |
| specificity | generic_sft | 30 |
| specificity | zero_shot | 24 |
| specificity | agentic_candidate | 21 |
| specificity | agentic_selected | 12 |
| specificity | specificity_sft | 7 |
| specificity | original | 4 |
| specificity | round1_final | 1 |
| specificity | round2_final | 1 |
| trust_safety | zero_shot | 41 |
| trust_safety | agentic_candidate | 23 |
| trust_safety | agentic_selected | 22 |
| trust_safety | generic_sft | 7 |
| trust_safety | specificity_sft | 4 |
| trust_safety | original | 1 |
| trust_safety | round1_final | 1 |
| trust_safety | round2_final | 1 |

## Objective Mean Selected Scores

| objective | mean_clickbait_penalty | mean_faithfulness | mean_specificity | mean_attractiveness | mean_summary_support | mean_llm_overall |
| --- | --- | --- | --- | --- | --- | --- |
| editorial | 0.043 | 4.833 | 4.641 | 4.128 | 0.763 | 4.575 |
| growth | 0.054 | 4.837 | 4.651 | 4.132 | 0.785 | 4.543 |
| specificity | 0.054 | 4.834 | 4.645 | 4.126 | 0.860 | 4.385 |
| trust_safety | 0.023 | 4.817 | 4.616 | 4.112 | 0.744 | 4.539 |

## Objective Presets

### trust_safety

Prefer faithful, clear, non-clickbait titles for trust-sensitive surfaces.

Weights:

| signal | weight |
| --- | --- |
| pred_faithfulness | 0.400 |
| pred_clarity | 0.200 |
| pred_specificity | 0.100 |
| pred_attractiveness | 0.000 |
| pred_non_clickbait | 0.300 |
| pairwise_reward | 0.100 |
| clickbait_penalty | -1.000 |
| length_style_score | 0.200 |

### growth

Prefer attractive, clear, specific titles while keeping clickbait risk bounded.

Weights:

| signal | weight |
| --- | --- |
| pred_faithfulness | 0.150 |
| pred_clarity | 0.200 |
| pred_specificity | 0.200 |
| pred_attractiveness | 0.350 |
| pred_non_clickbait | 0.100 |
| pairwise_reward | 0.200 |
| clickbait_penalty | -0.350 |
| length_style_score | 0.150 |

### editorial

Prefer balanced titles that look like compact human-edited news headlines.

Weights:

| signal | weight |
| --- | --- |
| pred_faithfulness | 0.250 |
| pred_clarity | 0.200 |
| pred_specificity | 0.200 |
| pred_attractiveness | 0.200 |
| pred_non_clickbait | 0.150 |
| pairwise_reward | 0.150 |
| clickbait_penalty | -0.500 |
| length_style_score | 0.350 |

### specificity

Prefer concrete, supported details without sacrificing faithfulness.

Weights:

| signal | weight |
| --- | --- |
| pred_faithfulness | 0.300 |
| pred_clarity | 0.100 |
| pred_specificity | 0.400 |
| pred_attractiveness | 0.050 |
| pred_non_clickbait | 0.150 |
| pairwise_reward | 0.100 |
| clickbait_penalty | -0.450 |
| length_style_score | 0.100 |
| summary_support_rate | 0.200 |

## Selection Examples

| objective | seed_id | category | selected_variant | selected_source | selected_headline | runner_up_variant | score_margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| editorial | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | zero_shot | 0.355 |
| editorial | 89 | lifestyle | round2_final | critic_guided_rewriter_round2 | Michelle Mero Riedel Cultivates a Garden in Oakdale | original | 0.149 |
| editorial | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.058 |
| editorial | 84 | sports | generic_sft | generic_sft_generator | Head coach may have doomed his team - and his kicker - with questionable call | agentic_candidate | 0.029 |
| editorial | 24 | news | original | human_editor | Tractor-trailer wedges itself under Waterbury bridge Monday morning | agentic_selected | 0.029 |
| growth | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | zero_shot | 0.249 |
| growth | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.237 |
| growth | 89 | lifestyle | round2_final | critic_guided_rewriter_round2 | Michelle Mero Riedel Cultivates a Garden in Oakdale | original | 0.186 |
| growth | 96 | travel | generic_sft | generic_sft_generator | You'll be able to see it from your own moped driven by you next month | zero_shot | 0.104 |
| growth | 43 | news | specificity_sft | specificity_sft_generator | Trump equates case of hate crime allegation made by actor Jussie Smollett to current impeachment process | zero_shot | 0.067 |
| specificity | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | round1_final | 0.386 |
| specificity | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.313 |
| specificity | 89 | lifestyle | round2_final | critic_guided_rewriter_round2 | Michelle Mero Riedel Cultivates a Garden in Oakdale | original | 0.239 |
| specificity | 96 | travel | generic_sft | generic_sft_generator | You'll be able to see it from your own moped driven by you next month | agentic_selected | 0.184 |
| specificity | 28 | sports | specificity_sft | specificity_sft_generator | Lakers' Frank Vogel is aware of the great situation he's in with the Lakers, and wants to make the best of it | zero_shot | 0.163 |
| trust_safety | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | zero_shot | 0.772 |
| trust_safety | 89 | lifestyle | round2_final | critic_guided_rewriter_round2 | Michelle Mero Riedel Cultivates a Garden in Oakdale | original | 0.065 |
| trust_safety | 69 | lifestyle | specificity_sft | specificity_sft_generator | Manatees: What's the sweetest sea creature? | round2_final | 0.057 |
| trust_safety | 92 | sports | generic_sft | generic_sft_generator | Michigan coach Juwan Howard asks questions about coaching | agentic_candidate | 0.021 |
| trust_safety | 24 | news | original | human_editor | Tractor-trailer wedges itself under Waterbury bridge Monday morning | agentic_candidate | 0.020 |

## Next Step

Use this matrix as the control plane for multi-agent work: add audience persona agents as new reward columns, then compare how their votes change objective-specific selection.
