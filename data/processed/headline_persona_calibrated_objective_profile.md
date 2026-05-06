# Persona-Calibrated Objective Selection

This report adds audience/persona preference signals to the existing multi-agent objective selector.

## Functional Change

The previous selector used local critic scores, clickbait penalty, pairwise reward, style heuristics, and support heuristics. This extension keeps those base objective scores, then adds persona-specific calibration terms. The default calibration strength is intentionally moderate so persona signals adjust rather than replace local critic scores.

```text
base objective score
+ persona target preference adjustment
+ consensus / persona-best bonus
= persona-calibrated objective score
```

## Files

- Input matrix: `data/processed/headline_multi_agent_candidate_matrix.csv`
- Persona votes: `data/processed/headline_audience_persona_votes.csv`
- Output matrix: `data/processed/headline_persona_calibrated_candidate_matrix.csv`
- Output selection: `data/processed/headline_persona_calibrated_objective_selection.csv`
- Calibration strength: `0.50`

## Persona Signal Coverage

| candidate_rows | seed_count | rows_with_persona_signal | seeds_with_persona_signal |
| --- | --- | --- | --- |
| 1200 | 100 | 419 | 90 |

## Calibration Terms

### trust_safety

Trust-sensitive selector: preserve the original safety objective, then boost candidates preferred by trust-sensitive readers.

| signal | weight |
| --- | --- |
| trust_sensitive_reader_overall | 0.350 |
| trust_sensitive_reader_trust | 0.250 |
| persona_consensus_best_rate | 0.200 |
| persona_best_rate | 0.100 |

### growth

Growth selector: preserve the original growth objective, then boost candidates that growth-oriented readers find engaging.

| signal | weight |
| --- | --- |
| growth_reader_overall | 0.300 |
| growth_reader_engagement | 0.350 |
| growth_reader_audience_fit | 0.150 |
| persona_consensus_best_rate | 0.100 |

### editorial

Editorial selector: preserve balanced editorial scoring, then boost candidates preferred by editorial reviewers and busy readers.

| signal | weight |
| --- | --- |
| editorial_reviewer_overall | 0.350 |
| editorial_reviewer_clarity | 0.200 |
| busy_news_reader_overall | 0.100 |
| persona_consensus_best_rate | 0.150 |

### specificity

Specificity selector: preserve source-supported detail scoring, then boost candidates that remain clear and audience-fit.

| signal | weight |
| --- | --- |
| editorial_reviewer_clarity | 0.200 |
| editorial_reviewer_audience_fit | 0.150 |
| busy_news_reader_clarity | 0.100 |
| persona_mean_audience_fit | 0.150 |

## Selection Change Summary

| objective | seeds | changed_count | mean_selected_score | mean_persona_adjustment | mean_llm_overall | mean_target_persona_overall | mean_clickbait_penalty | changed_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| editorial | 100 | 56 | 5.622 | 0.559 | 4.684 | 4.678 | 0.052 | 0.560 |
| growth | 100 | 60 | 5.420 | 0.626 | 4.695 | 4.663 | 0.080 | 0.600 |
| specificity | 100 | 62 | 5.464 | 0.460 | 4.568 | 4.511 | 0.072 | 0.620 |
| trust_safety | 100 | 51 | 5.592 | 0.559 | 4.636 | 4.556 | 0.045 | 0.510 |

## Calibrated Selected Variant Counts

| objective | selected_variant | selected_count |
| --- | --- | --- |
| editorial | zero_shot | 60 |
| editorial | original | 24 |
| editorial | agentic_selected | 8 |
| editorial | generic_sft | 3 |
| editorial | agentic_candidate | 2 |
| editorial | round2_final | 2 |
| editorial | specificity_sft | 1 |
| growth | zero_shot | 69 |
| growth | original | 14 |
| growth | agentic_selected | 8 |
| growth | agentic_candidate | 5 |
| growth | generic_sft | 3 |
| growth | round2_final | 1 |
| specificity | zero_shot | 51 |
| specificity | agentic_selected | 14 |
| specificity | original | 13 |
| specificity | generic_sft | 12 |
| specificity | agentic_candidate | 5 |
| specificity | specificity_sft | 3 |
| specificity | round2_final | 2 |
| trust_safety | zero_shot | 56 |
| trust_safety | original | 26 |
| trust_safety | agentic_selected | 11 |
| trust_safety | generic_sft | 3 |
| trust_safety | specificity_sft | 2 |
| trust_safety | agentic_candidate | 1 |
| trust_safety | round2_final | 1 |

## Persona Coverage By Variant

| variant | rows | rows_with_persona_signal | mean_persona_overall | mean_persona_best_rate | mean_persona_overall_gap | persona_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- |
| agentic_candidate | 500 | 10 | 3.875 | 0.001 | 0.750 | 0.020 |
| agentic_selected | 100 | 89 | 3.787 | 0.005 | 0.870 | 0.890 |
| generic_sft | 100 | 88 | 3.014 | 0.010 | 1.615 | 0.880 |
| original | 100 | 90 | 4.061 | 0.395 | -0.192 | 0.900 |
| round1_final | 100 | 0 |  | 0.000 |  | 0.000 |
| round2_final | 100 | 9 | 3.500 | 0.000 | 1.016 | 0.090 |
| specificity_sft | 100 | 43 | 2.558 | 0.000 | 2.043 | 0.430 |
| zero_shot | 100 | 90 | 4.469 | 0.487 | 0.194 | 0.900 |

## Examples Where Persona Calibration Changed Selection

| objective | seed_id | category | base_selected_variant | base_selected_headline | selected_variant | selected_headline | selected_persona_adjustment | target_persona_overall | score_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| editorial | 55 | health | generic_sft | A dentist is a long, costly, and demanding endeavor | zero_shot | Becoming a dentist is a long and costly process in health care | 0.725 | 5.000 | 0.627 |
| editorial | 48 | travel | agentic_candidate | Texas Parks and Wildlife Highlights Iconic Bobwhite Quail Hunting | zero_shot | Hunting bobwhite quail remains a cherished tradition in Texas | 0.725 | 5.000 | 0.592 |
| editorial | 8 | sports | agentic_selected | Unrestricted Free Agent Domingue Looks for Playing Opportunities This Summer | zero_shot | Domingue Seeks Regular Playing Opportunity Ahead of Free Agency | 0.725 | 5.000 | 0.503 |
| editorial | 59 | health | specificity_sft | Patient's name has not been released, but there's a breakdown of what we do know | original | Patient dies in ER at WellSpan York Hospital what we know now | 0.725 | 5.000 | 0.499 |
| editorial | 62 | news | agentic_candidate | Dr. Ulrich Klopfer's Family Discovers 2,246 Preserved Fetuses in Illinois Home | zero_shot | Authorities discover additional preserved fetuses at Dr. Ulrich Klopfer's home | 0.725 | 5.000 | 0.448 |
| growth | 48 | travel | agentic_selected | Texas Parks and Wildlife Highlights Iconic Bobwhite Quail Hunting | zero_shot | Hunting bobwhite quail remains a cherished tradition in Texas | 0.850 | 5.000 | 0.701 |
| growth | 8 | sports | agentic_selected | Unrestricted Free Agent Domingue Looks for Playing Opportunities This Summer | zero_shot | Domingue Seeks Regular Playing Opportunity Ahead of Free Agency | 0.850 | 5.000 | 0.646 |
| growth | 9 | news | agentic_selected | Driver Fatally Shot in St. Paul on Sunday Night | zero_shot | Driver Fatally Shot in St. Paul, Police Investigate Incident | 0.850 | 5.000 | 0.625 |
| growth | 62 | news | agentic_candidate | Dr. Ulrich Klopfer's Family Discovers 2,246 Preserved Fetuses in Illinois Home | zero_shot | Authorities discover additional preserved fetuses at Dr. Ulrich Klopfer's home | 0.850 | 5.000 | 0.448 |
| growth | 14 | finance | generic_sft | Critics say Procter & Gamble contributes to deforestation and should use recycled materials for paper products | zero_shot | Critics urge Procter & Gamble to address deforestation and use recycled materials | 0.850 | 5.000 | 0.436 |
| specificity | 88 | sports | generic_sft | Finley: Bengals still looking for quarterback of the future | zero_shot | Bengals Continue Search for Future Quarterback After Finley's Performance | 0.600 | 5.000 | 0.334 |
| specificity | 62 | news | agentic_candidate | Dr. Ulrich Klopfer's Family Discovers 2,246 Preserved Fetuses in Illinois Home | zero_shot | Authorities discover additional preserved fetuses at Dr. Ulrich Klopfer's home | 0.600 | 5.000 | 0.319 |
| specificity | 13 | travel | generic_sft | Game wardens investigating fatal crash in small town west of Belfast | zero_shot | Game wardens investigate fatal ATV crash in Montville, killing 30-year-old Rachel Curtis | 0.600 | 5.000 | 0.312 |
| specificity | 60 | news | generic_sft | Students evacuated at Sky Valley Education Center after noxious fumes caught fire | zero_shot | Sky Valley Education Center teachers evacuate students after fluorescent lights catch fire | 0.600 | 5.000 | 0.246 |
| specificity | 16 | travel | agentic_candidate | Nightmare on Franklin Street Features Live Podcast and Classic Film | zero_shot | Tampa Theatre Hosts Halloween Film Festival with Live Shows and Talks | 0.600 | 5.000 | 0.147 |
| trust_safety | 89 | lifestyle | round2_final | Michelle Mero Riedel Cultivates Garden in Oakdale for Photography | original | Photographer tends a picture-perfect garden in Oakdale | 0.750 | 5.000 | 0.520 |
| trust_safety | 16 | travel | agentic_candidate | Nightmare on Franklin Street Features Live Podcast and Classic Film | zero_shot | Tampa Theatre Hosts Halloween Film Festival with Live Shows and Talks | 0.750 | 5.000 | 0.449 |
| trust_safety | 17 | news | generic_sft | Texas Gov. Greg Abbott targets left-leaning homelessness policies | zero_shot | Texas Officials, Including Gov. Abbott, Criticize Austin's Homelessness Policies | 0.750 | 5.000 | 0.441 |
| trust_safety | 69 | lifestyle | specificity_sft | Manatees: What's the sweetest sea creature? | original | 13 Things You Never Knew About Manatees | 0.750 | 5.000 | 0.438 |
| trust_safety | 85 | weather | generic_sft | Brachiosaurus creates video showing snow in Point Place, Ohio | zero_shot | Video Captures Snow Progression in Point Place, Ohio, Amid Frozen Clock | 0.750 | 5.000 | 0.429 |

## Potential Local Reward Overestimation Examples

These examples have local `pred_overall` at least 0.75 points higher than the mean persona `overall` score.

| seed_id | category | variant | headline | pred_overall | persona_mean_overall | persona_overall_gap | llm_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | health | generic_sft | Aaron Archer loses 230 pounds and nearly 25 percent body fat | 4.750 | 1.000 | 3.750 | 1.000 |
| 11 | lifestyle | generic_sft | The King of Pop is a decade where you'll meet multiple princesses, a new Queen, The King, Prince and the future King | 4.729 | 1.000 | 3.729 | 2.000 |
| 84 | sports | specificity_sft | Head coach may have doneomed his team - and his kicker - with questionable call | 4.712 | 1.000 | 3.712 | 3.000 |
| 41 | finance | specificity_sft | Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan | 4.689 | 1.000 | 3.689 | 1.000 |
| 41 | finance | generic_sft | Kevin Mooneyhan's tomahawk chop would be too much for Kevin Mooneyhan | 4.765 | 1.250 | 3.515 | 1.000 |
| 35 | sports | specificity_sft | Astros' James Paxton gets hit hard, allowing five earned runs on eight hits in just four innings | 4.781 | 1.500 | 3.281 | 4.000 |
| 73 | lifestyle | specificity_sft | 'Spooky wedding shoot': 'It's a great way to get a little dark inspiration for your big day' | 4.654 | 1.500 | 3.154 | 3.000 |
| 75 | travel | specificity_sft | Winter is not stopping Milwaukee's pilot scooter program even if snow does for a day | 4.585 | 1.500 | 3.085 | 4.000 |
| 37 | lifestyle | generic_sft | You won't believe these wedding gowns are made from only quilted Northern toilet paper, tape, glue, and a needle and thread | 4.779 | 1.750 | 3.029 | 4.000 |
| 38 | sports | specificity_sft | Grand Rapids' top player votes for his favorite | 4.250 | 1.250 | 3.000 | 2.000 |
| 12 | sports | generic_sft | Celtics are doing just the opposite, but they're doing just the opposite | 3.977 | 1.000 | 2.977 | 1.000 |
| 61 | foodanddrink | specificity_sft | San Francisco chefs love white bread, but they love it | 3.921 | 1.000 | 2.921 | 2.000 |

## Interpretation

Persona calibration turns audience votes into an operational selection signal. It does not replace local critics; it adjusts them when a candidate appears better aligned with the target audience for a specific objective.

This makes the system closer to the intended product: a headline review console where a user can switch between trust, growth, editorial, and specificity goals and see different recommended headlines.

## Caveats

- Persona signals are available for only the completed persona-vote subset.
- Persona votes are simulated with an LLM, not collected from real users.
- The calibrated selector should be treated as a demo/control-layer feature, not a production ranking model.
- Missing persona signals are treated as neutral, so unvoted candidates are not directly penalized but receive no persona boost.
