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
| 1000 | 100 | 454 | 100 |

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
| editorial | 100 | 64 | 5.690 | 0.604 | 4.687 | 4.660 | 0.040 | 0.640 |
| growth | 100 | 63 | 5.515 | 0.703 | 4.765 | 4.673 | 0.088 | 0.630 |
| specificity | 100 | 63 | 5.530 | 0.506 | 4.684 | 4.550 | 0.045 | 0.630 |
| trust_safety | 100 | 65 | 5.684 | 0.625 | 4.677 | 4.600 | 0.035 | 0.650 |

## Calibrated Selected Variant Counts

| objective | selected_variant | selected_count |
| --- | --- | --- |
| editorial | zero_shot | 57 |
| editorial | original | 28 |
| editorial | agentic_selected | 9 |
| editorial | generic_sft | 2 |
| editorial | specificity_sft | 2 |
| editorial | agentic_candidate | 1 |
| editorial | round2_final | 1 |
| growth | zero_shot | 68 |
| growth | original | 19 |
| growth | agentic_selected | 6 |
| growth | generic_sft | 3 |
| growth | agentic_candidate | 1 |
| growth | round1_final | 1 |
| growth | round2_final | 1 |
| growth | specificity_sft | 1 |
| specificity | zero_shot | 51 |
| specificity | original | 19 |
| specificity | agentic_selected | 16 |
| specificity | generic_sft | 9 |
| specificity | agentic_candidate | 2 |
| specificity | specificity_sft | 2 |
| specificity | round2_final | 1 |
| trust_safety | zero_shot | 54 |
| trust_safety | original | 31 |
| trust_safety | agentic_selected | 8 |
| trust_safety | generic_sft | 3 |
| trust_safety | specificity_sft | 2 |
| trust_safety | agentic_candidate | 1 |
| trust_safety | round2_final | 1 |

## Persona Coverage By Variant

| variant | rows | rows_with_persona_signal | mean_persona_overall | mean_persona_best_rate | mean_persona_overall_gap | persona_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- |
| agentic_candidate | 300 | 4 | 4.125 | 0.000 | 0.670 | 0.013 |
| agentic_selected | 100 | 97 | 3.745 | 0.020 | 0.919 | 0.970 |
| generic_sft | 100 | 98 | 2.959 | 0.025 | 1.672 | 0.980 |
| original | 100 | 100 | 4.010 | 0.460 | -0.086 | 1.000 |
| round1_final | 100 | 0 |  | 0.000 |  | 0.000 |
| round2_final | 100 | 9 | 3.611 | 0.000 | 0.991 | 0.090 |
| specificity_sft | 100 | 46 | 2.630 | 0.003 | 1.964 | 0.460 |
| zero_shot | 100 | 100 | 4.400 | 0.492 | 0.284 | 1.000 |

## Examples Where Persona Calibration Changed Selection

| objective | seed_id | category | base_selected_variant | base_selected_headline | selected_variant | selected_headline | selected_persona_adjustment | target_persona_overall | score_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| editorial | 4 | sports | agentic_candidate | Mike Jones Highlights Seahawks' Impressive Victory Over 49ers | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.725 | 5.000 | 0.682 |
| editorial | 21 | lifestyle | agentic_selected | Worcester Voters Elect One Person of Color to School Committee | zero_shot | Worcester's School Committee election shows limited progress on racial representation | 0.725 | 5.000 | 0.615 |
| editorial | 95 | weather | agentic_selected | Waves Hit Lake Superior and Lake Michigan Amid Severe Storm This Week | zero_shot | Intense storm brings 50-mph winds and large waves to Great Lakes shores | 0.725 | 5.000 | 0.449 |
| editorial | 38 | sports | agentic_selected | Vote for Grand Rapids' Top Player from Last Week's Playoff Openers | zero_shot | Voting Open for Grand Rapids Area's Top Player from Postseason Openers | 0.725 | 5.000 | 0.446 |
| editorial | 65 | travel | agentic_selected | Brevard County TRIAD's Senior Santa Program Delivers Cheer to Seniors | zero_shot | Volunteers Spread Holiday Cheer to Seniors Through Senior Santa Program in Brevard County | 0.725 | 5.000 | 0.439 |
| growth | 91 | news | agentic_candidate | Imaad Shah Zuberi to Plead Guilty to Concealing Foreign Agent Work and Campaign Violations | original | SoCal Man Who Donated $900K to Trump Will Plead Guilty to Hiding Work as Foreign Agent, Illegal Campaign Contributions | 0.850 | 5.000 | 0.801 |
| growth | 21 | lifestyle | agentic_selected | Worcester Voters Elect One Person of Color to School Committee | zero_shot | Worcester's School Committee election shows limited progress on racial representation | 0.850 | 5.000 | 0.661 |
| growth | 85 | weather | generic_sft | Brachiosaurus creates video showing snow in Point Place, Ohio | zero_shot | Video Captures Snow Progression in Point Place, Ohio, Amid Frozen Clock | 0.850 | 5.000 | 0.624 |
| growth | 100 | travel | generic_sft | Boca Raton shopping plaza in need of a refresh after losing tenants | original | Boca Raton's Mizner Park jazzes up plaza with new restaurants, stores, bars and entertainment | 0.850 | 5.000 | 0.544 |
| growth | 41 | finance | agentic_selected | Kevin Mooneyhan Reflects on Loss in Game 5 and Tomahawk Chop | original | Online petition to keep divisive Braves tomahawk chop nears 60,000 | 0.850 | 5.000 | 0.512 |
| specificity | 35 | sports | generic_sft | Astros are good at stealing signs and picking up on opposing pitchers' small tells | zero_shot | Yankees' James Paxton acknowledges tipping pitches after Astros matchup in April | 0.600 | 5.000 | 0.471 |
| specificity | 4 | sports | generic_sft | Seattle Seahawks win over 49ers | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.600 | 5.000 | 0.382 |
| specificity | 88 | sports | generic_sft | Finley: Bengals still looking for quarterback of the future | zero_shot | Bengals Continue Search for Future Quarterback After Finley's Performance | 0.600 | 5.000 | 0.345 |
| specificity | 21 | lifestyle | agentic_selected | Worcester Voters Elect One Person of Color to School Committee | zero_shot | Worcester's School Committee election shows limited progress on racial representation | 0.600 | 5.000 | 0.312 |
| specificity | 96 | travel | generic_sft | You'll be able to see it from your own moped driven by you next month | agentic_selected | Next Month, Explore the City on Your Own Moped | 0.600 | 5.000 | 0.239 |
| trust_safety | 62 | news | agentic_candidate | Family Discovers 2,246 Preserved Fetuses at Dr. Ulrich Klopfer's Illinois Home | zero_shot | Authorities discover additional preserved fetuses at Dr. Ulrich Klopfer's home | 0.750 | 5.000 | 0.748 |
| trust_safety | 4 | sports | agentic_candidate | Mike Jones Highlights Seahawks' Impressive Victory Over 49ers | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.750 | 5.000 | 0.601 |
| trust_safety | 95 | weather | agentic_selected | Waves Hit Lake Superior and Lake Michigan Amid Severe Storm This Week | zero_shot | Intense storm brings 50-mph winds and large waves to Great Lakes shores | 0.750 | 5.000 | 0.448 |
| trust_safety | 85 | weather | generic_sft | Brachiosaurus creates video showing snow in Point Place, Ohio | zero_shot | Video Captures Snow Progression in Point Place, Ohio, Amid Frozen Clock | 0.750 | 5.000 | 0.448 |
| trust_safety | 61 | foodanddrink | zero_shot | San Francisco Chefs Embrace Milk Bread Amid Sourdough Tradition | original | San Francisco chefs find nostalgia in Japanese milk bread | 0.750 | 5.000 | 0.447 |

## Potential Local Reward Overestimation Examples

These examples have local `pred_overall` at least 0.75 points higher than the mean persona `overall` score.

| seed_id | category | variant | headline | pred_overall | persona_mean_overall | persona_overall_gap | llm_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 81 | health | generic_sft | Aaron Archer loses 230 pounds and nearly 25 percent body fat | 4.750 | 1.000 | 3.750 | 1.000 |
| 11 | lifestyle | generic_sft | The King of Pop is a decade where you'll meet multiple princesses, a new Queen, The King, Prince and the future King | 4.729 | 1.000 | 3.729 | 2.000 |
| 41 | finance | specificity_sft | Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan, Kevin Mooneyhan | 4.689 | 1.000 | 3.689 | 1.000 |
| 73 | lifestyle | specificity_sft | 'Spooky wedding shoot': 'It's a great way to get a little dark inspiration for your big day' | 4.654 | 1.000 | 3.654 | 3.000 |
| 35 | sports | specificity_sft | Astros' James Paxton gets hit hard, allowing five earned runs on eight hits in just four innings | 4.781 | 1.250 | 3.531 | 4.000 |
| 41 | finance | generic_sft | Kevin Mooneyhan's tomahawk chop would be too much for Kevin Mooneyhan | 4.765 | 1.500 | 3.265 | 1.000 |
| 38 | sports | specificity_sft | Grand Rapids' top player votes for his favorite | 4.250 | 1.000 | 3.250 | 2.000 |
| 96 | travel | specificity_sft | You'll be able to see it from your own moped | 4.466 | 1.250 | 3.216 | 3.000 |
| 80 | finance | generic_sft | Tax havens in 20 countries that most often serve as tax havens | 4.582 | 1.500 | 3.082 | 4.000 |
| 80 | finance | specificity_sft | Tax havens in the 20 countries that most often serve as tax havens | 4.569 | 1.500 | 3.069 | 4.000 |
| 92 | sports | specificity_sft | Monty Williams feels first-year Michigan coach Juwan Howard could do anything, but he was one of the first coaches | 4.794 | 1.750 | 3.044 | 3.000 |
| 73 | lifestyle | generic_sft | 'Spooky wedding shoot' is a great way to get a little dark inspiration for your big day | 4.766 | 1.750 | 3.016 | 3.000 |

## Interpretation

Persona calibration turns audience votes into an operational selection signal. It does not replace local critics; it adjusts them when a candidate appears better aligned with the target audience for a specific objective.

This makes the system closer to the intended product: a headline review console where a user can switch between trust, growth, editorial, and specificity goals and see different recommended headlines.

## Caveats

- Persona signals are available for only the completed persona-vote subset.
- Persona votes are simulated with an LLM, not collected from real users.
- The calibrated selector should be treated as a demo/control-layer feature, not a production ranking model.
- Missing persona signals are treated as neutral, so unvoted candidates are not directly penalized but receive no persona boost.
