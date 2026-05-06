# Headline Review Demo Cases

This report describes the simplified product/demo dataset. It hides the full research candidate pool and exposes only a few meaningful options per article and objective.

## Simplified Product Flow

```text
article summary
-> hidden candidate pool
-> unified evaluator: quality + risk/safety + audience fit + objective fit
-> objective/persona-calibrated recommendation
-> show visible options with selected/not-selected explanations
```

## Files

- Input matrix: `data/processed/headline_persona_calibrated_candidate_matrix.csv`
- Input selection: `data/processed/headline_persona_calibrated_objective_selection.csv`
- Output demo cases: `data/processed/headline_review_demo_cases.csv`

## Coverage

| demo_rows | seed_count | objective_count | mean_visible_options | mean_hidden_pool_size |
| --- | --- | --- | --- | --- |
| 113 | 10 | 4 | 2.825 | 12.000 |

## Visible Option Counts

| objective | display_label | visible_count |
| --- | --- | --- |
| editorial | Recommended | 10 |
| editorial | Human baseline | 8 |
| editorial | Low-risk alternative | 8 |
| editorial | GenAI baseline | 2 |
| growth | Recommended | 10 |
| growth | Human baseline | 9 |
| growth | Low-risk alternative | 8 |
| growth | GenAI baseline | 1 |
| specificity | Human baseline | 10 |
| specificity | Recommended | 10 |
| specificity | Low-risk alternative | 8 |
| specificity | GenAI baseline | 1 |
| trust_safety | Recommended | 10 |
| trust_safety | Low-risk alternative | 8 |
| trust_safety | Human baseline | 7 |
| trust_safety | GenAI baseline | 3 |

## Recommended Variant Counts

| objective | variant | recommended_count |
| --- | --- | --- |
| editorial | zero_shot | 8 |
| editorial | original | 2 |
| growth | zero_shot | 9 |
| growth | original | 1 |
| specificity | zero_shot | 9 |
| specificity | generic_sft | 1 |
| trust_safety | zero_shot | 7 |
| trust_safety | original | 3 |

## Mean Recommended Unified Scores

| objective | mean_quality_score | mean_risk_score | mean_audience_score | mean_objective_fit_score | mean_support_score | mean_unified_decision_score |
| --- | --- | --- | --- | --- | --- | --- |
| editorial | 0.825 | 1.000 | 0.955 | 0.961 | 0.575 | 0.922 |
| growth | 0.848 | 1.000 | 0.950 | 0.932 | 0.636 | 0.929 |
| specificity | 0.851 | 1.000 | 0.940 | 0.924 | 0.642 | 0.915 |
| trust_safety | 0.804 | 1.000 | 0.955 | 0.964 | 0.503 | 0.927 |

## Recommended Examples

| objective | seed_id | category | variant | headline | quality_score | risk_score | audience_score | objective_fit_score | support_score | unified_decision_score | decision_explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| editorial | 1 | news | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 0.747 | 1.000 | 0.950 | 0.972 | 0.286 | 0.897 | Prioritizes balanced, compact, publication-ready news headlines. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| editorial | 2 | foodanddrink | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 0.738 | 0.998 | 1.000 | 0.973 | 0.250 | 0.901 | Prioritizes balanced, compact, publication-ready news headlines. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| editorial | 3 | travel | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 0.842 | 1.000 | 0.800 | 0.895 | 0.714 | 0.888 | Prioritizes balanced, compact, publication-ready news headlines. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. Its main terms are supported by the article summary. |
| editorial | 4 | sports | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.810 | 1.000 | 1.000 | 0.968 | 0.500 | 0.925 | Prioritizes balanced, compact, publication-ready news headlines. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| editorial | 5 | sports | zero_shot | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 0.929 | 1.000 | 1.000 | 0.972 | 0.857 | 0.968 | Prioritizes balanced, compact, publication-ready news headlines. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| growth | 1 | news | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 0.747 | 1.000 | 0.950 | 0.850 | 0.286 | 0.879 | Prioritizes engaging headlines while still controlling clickbait and trust risk. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| growth | 2 | foodanddrink | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings | 0.857 | 1.000 | 1.000 | 0.953 | 0.625 | 0.950 | Prioritizes engaging headlines while still controlling clickbait and trust risk. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| growth | 3 | travel | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 0.842 | 1.000 | 0.800 | 0.901 | 0.714 | 0.881 | Prioritizes engaging headlines while still controlling clickbait and trust risk. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. Its main terms are supported by the article summary. |
| growth | 4 | sports | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.810 | 1.000 | 1.000 | 0.941 | 0.500 | 0.935 | Prioritizes engaging headlines while still controlling clickbait and trust risk. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| growth | 5 | sports | zero_shot | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 0.929 | 1.000 | 1.000 | 0.941 | 0.857 | 0.965 | Prioritizes engaging headlines while still controlling clickbait and trust risk. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| specificity | 1 | news | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 0.747 | 1.000 | 0.950 | 0.897 | 0.286 | 0.873 | Prioritizes concrete, source-supported details without losing clarity. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| specificity | 2 | foodanddrink | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings | 0.857 | 1.000 | 1.000 | 0.945 | 0.625 | 0.932 | Prioritizes concrete, source-supported details without losing clarity. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| specificity | 3 | travel | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 0.842 | 1.000 | 0.800 | 0.870 | 0.714 | 0.874 | Prioritizes concrete, source-supported details without losing clarity. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. Its main terms are supported by the article summary. |
| specificity | 4 | sports | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.810 | 1.000 | 1.000 | 0.928 | 0.500 | 0.911 | Prioritizes concrete, source-supported details without losing clarity. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| specificity | 5 | sports | zero_shot | Cyclones and Longhorns Announce Starters for 2019 Big 12 Matchup | 0.929 | 1.000 | 1.000 | 0.954 | 0.857 | 0.958 | Prioritizes concrete, source-supported details without losing clarity. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| trust_safety | 1 | news | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 0.747 | 1.000 | 0.950 | 0.973 | 0.286 | 0.911 | Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| trust_safety | 2 | foodanddrink | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 0.738 | 0.998 | 1.000 | 0.975 | 0.250 | 0.916 | Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |
| trust_safety | 3 | travel | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 0.842 | 1.000 | 0.800 | 0.926 | 0.714 | 0.908 | Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. Its main terms are supported by the article summary. |
| trust_safety | 4 | sports | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 0.810 | 1.000 | 1.000 | 0.972 | 0.500 | 0.937 | Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. Its main terms are supported by the article summary. |
| trust_safety | 5 | sports | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 0.713 | 1.000 | 1.000 | 0.974 | 0.143 | 0.909 | Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces. Risk is low after folding clickbait into the safety score. Audience/persona scoring favors it. The local quality critic rates it highly. |

## Interpretation

The demo should use this file instead of the full 1,200-row research matrix. The UI can still mention that candidates come from a larger hidden pool, but it should show only the human baseline, GenAI baseline, low-risk alternative, and final recommendation.

The standalone clickbait critic is folded into the Risk/Safety score. The product-facing decision is shown through a unified scorecard: Quality, Risk/Safety, Audience Fit, and Objective Fit. Each non-selected option includes a short reason explaining why it lost to the recommendation.
