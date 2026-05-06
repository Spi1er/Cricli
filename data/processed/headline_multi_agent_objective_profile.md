# Multi-Agent Objective Headline Matrix

This report reframes the project as an offline agentic RL-style selection system.

## Agentic Framing

- State: article summary, category, and context.
- Action: choose a candidate headline from multiple generator agents.
- Reward vector: local critic scores, clickbait penalty, pairwise reward, style heuristics, and retrospective LLM judge labels when available.
- Policy: objective-specific selector that scalarizes the reward vector differently for trust, growth, editorial, or specificity goals.

## Files

- Candidate matrix: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_multi_agent_candidate_matrix.csv`
- Objective selections: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_multi_agent_objective_selection.csv`

## Candidate Sources

| variant | candidate_source | rows | mean_clickbait_penalty | mean_faithfulness | mean_specificity | mean_attractiveness | mean_local_final | mean_llm_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agentic_candidate | agentic_generator_v3_candidate | 500 | 0.109 | 4.664 | 4.321 | 3.936 | 6.114 |  |
| agentic_selected | agentic_selector_v3 | 100 | 0.053 | 4.779 | 4.533 | 4.065 | 6.355 | 4.490 |
| generic_sft | generic_sft_generator | 100 | 0.226 | 4.755 | 4.518 | 4.055 | 4.680 | 3.780 |
| original | human_editor | 100 | 0.273 | 4.117 | 3.681 | 3.591 | 5.246 | 3.740 |
| round1_final | critic_guided_rewriter_round1 | 100 | 0.076 | 4.772 | 4.523 | 4.055 | 6.329 |  |
| round2_final | critic_guided_rewriter_round2 | 100 | 0.066 | 4.761 | 4.508 | 4.046 | 6.318 | 4.680 |
| specificity_sft | specificity_sft_generator | 100 | 0.211 | 4.785 | 4.549 | 4.073 | 4.713 | 3.860 |
| zero_shot | api_zero_shot_generator | 100 | 0.088 | 4.777 | 4.535 | 4.063 | 6.334 | 4.710 |

## Objective Selection Counts

| objective | selected_variant | selected_count |
| --- | --- | --- |
| editorial | zero_shot | 34 |
| editorial | agentic_selected | 21 |
| editorial | agentic_candidate | 16 |
| editorial | generic_sft | 13 |
| editorial | specificity_sft | 8 |
| editorial | original | 4 |
| editorial | round2_final | 3 |
| editorial | round1_final | 1 |
| growth | zero_shot | 31 |
| growth | generic_sft | 19 |
| growth | agentic_selected | 18 |
| growth | agentic_candidate | 16 |
| growth | specificity_sft | 9 |
| growth | round2_final | 3 |
| growth | original | 2 |
| growth | round1_final | 2 |
| specificity | generic_sft | 38 |
| specificity | agentic_candidate | 18 |
| specificity | zero_shot | 18 |
| specificity | agentic_selected | 10 |
| specificity | specificity_sft | 9 |
| specificity | original | 4 |
| specificity | round2_final | 2 |
| specificity | round1_final | 1 |
| trust_safety | zero_shot | 34 |
| trust_safety | agentic_selected | 22 |
| trust_safety | agentic_candidate | 14 |
| trust_safety | generic_sft | 14 |
| trust_safety | specificity_sft | 7 |
| trust_safety | original | 5 |
| trust_safety | round2_final | 3 |
| trust_safety | round1_final | 1 |

## Objective Mean Selected Scores

| objective | mean_clickbait_penalty | mean_faithfulness | mean_specificity | mean_attractiveness | mean_summary_support | mean_llm_overall |
| --- | --- | --- | --- | --- | --- | --- |
| editorial | 0.055 | 4.818 | 4.610 | 4.111 | 0.771 | 4.554 |
| growth | 0.069 | 4.824 | 4.625 | 4.118 | 0.792 | 4.476 |
| specificity | 0.065 | 4.822 | 4.620 | 4.113 | 0.863 | 4.333 |
| trust_safety | 0.034 | 4.801 | 4.585 | 4.096 | 0.763 | 4.494 |

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
| editorial | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | agentic_candidate | 0.300 |
| editorial | 41 | finance | original | human_editor | Online petition to keep divisive Braves tomahawk chop nears 60,000 | agentic_candidate | 0.134 |
| editorial | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.076 |
| editorial | 84 | sports | generic_sft | generic_sft_generator | Head coach may have doomed his team - and his kicker - with questionable call | agentic_candidate | 0.044 |
| editorial | 77 | sports | specificity_sft | specificity_sft_generator | Garoppolo and 49ers offense could operate at less than full strength Sunday against Arizona | generic_sft | 0.028 |
| growth | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | round1_final | 0.264 |
| growth | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.255 |
| growth | 96 | travel | generic_sft | generic_sft_generator | You'll be able to see it from your own moped driven by you next month | agentic_selected | 0.165 |
| growth | 41 | finance | original | human_editor | Online petition to keep divisive Braves tomahawk chop nears 60,000 | agentic_candidate | 0.124 |
| growth | 28 | sports | specificity_sft | specificity_sft_generator | Lakers' Frank Vogel is aware of the great situation he's in with the Lakers, and wants to make the best of it | zero_shot | 0.074 |
| specificity | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | round1_final | 0.399 |
| specificity | 59 | health | specificity_sft | specificity_sft_generator | Patient's name has not been released, but there's a breakdown of what we do know | original | 0.329 |
| specificity | 28 | sports | specificity_sft | specificity_sft_generator | Lakers' Frank Vogel is aware of the great situation he's in with the Lakers, and wants to make the best of it | zero_shot | 0.191 |
| specificity | 96 | travel | generic_sft | generic_sft_generator | You'll be able to see it from your own moped driven by you next month | specificity_sft | 0.188 |
| specificity | 84 | sports | generic_sft | generic_sft_generator | Head coach may have doomed his team - and his kicker - with questionable call | specificity_sft | 0.083 |
| trust_safety | 41 | finance | original | human_editor | Online petition to keep divisive Braves tomahawk chop nears 60,000 | agentic_candidate | 0.120 |
| trust_safety | 69 | lifestyle | specificity_sft | specificity_sft_generator | Manatees: What's the sweetest sea creature? | zero_shot | 0.117 |
| trust_safety | 59 | health | original | human_editor | Patient dies in ER at WellSpan York Hospital what we know now | agentic_selected | 0.114 |
| trust_safety | 37 | lifestyle | specificity_sft | specificity_sft_generator | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | agentic_candidate | 0.110 |
| trust_safety | 84 | sports | generic_sft | generic_sft_generator | Head coach may have doomed his team - and his kicker - with questionable call | agentic_candidate | 0.026 |

## Next Step

Use this matrix as the control plane for multi-agent work: add audience persona agents as new reward columns, then compare how their votes change objective-specific selection.
