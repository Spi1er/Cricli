# Audience Persona Voting Profile

- Input: `data/processed/headline_audience_persona_votes.csv`
- Completed seed count: 100
- Persona count: 4
- Vote rows: 1,816

## Interpretation

Persona voting is the proposal-aligned audience layer. It should be treated as an evaluator module that estimates how different audience goals prefer different candidate headlines.

## Consensus Best Counts

| variant | consensus_best_count |
| --- | --- |
| zero_shot | 51 |
| original | 46 |
| generic_sft | 2 |
| agentic_selected | 1 |

## Persona Best Counts

| persona | variant | best_count |
| --- | --- | --- |
| busy_news_reader | original | 50 |
| busy_news_reader | zero_shot | 46 |
| busy_news_reader | generic_sft | 3 |
| busy_news_reader | agentic_selected | 1 |
| editorial_reviewer | zero_shot | 51 |
| editorial_reviewer | original | 45 |
| editorial_reviewer | agentic_selected | 2 |
| editorial_reviewer | generic_sft | 2 |
| growth_reader | zero_shot | 55 |
| growth_reader | original | 39 |
| growth_reader | agentic_selected | 4 |
| growth_reader | generic_sft | 1 |
| growth_reader | specificity_sft | 1 |
| trust_sensitive_reader | original | 50 |
| trust_sensitive_reader | zero_shot | 45 |
| trust_sensitive_reader | generic_sft | 4 |
| trust_sensitive_reader | agentic_selected | 1 |

## Mean Scores By Persona And Variant

| persona | variant | trust | engagement | clarity | audience_fit | overall |
| --- | --- | --- | --- | --- | --- | --- |
| busy_news_reader | zero_shot | 4.530 | 3.760 | 4.710 | 4.420 | 4.360 |
| busy_news_reader | agentic_candidate | 4.750 | 3.500 | 4.750 | 3.750 | 4.250 |
| busy_news_reader | original | 4.040 | 3.340 | 4.320 | 4.120 | 3.950 |
| busy_news_reader | agentic_selected | 4.216 | 3.175 | 4.268 | 3.670 | 3.639 |
| busy_news_reader | round2_final | 4.444 | 2.778 | 4.333 | 3.556 | 3.444 |
| busy_news_reader | generic_sft | 3.704 | 2.459 | 3.490 | 3.041 | 2.827 |
| busy_news_reader | specificity_sft | 3.326 | 2.109 | 3.217 | 2.674 | 2.500 |
| editorial_reviewer | zero_shot | 4.710 | 3.940 | 4.590 | 4.490 | 4.450 |
| editorial_reviewer | agentic_candidate | 4.750 | 3.250 | 4.750 | 4.000 | 4.250 |
| editorial_reviewer | original | 4.260 | 3.560 | 4.210 | 4.090 | 4.010 |
| editorial_reviewer | round2_final | 4.556 | 2.889 | 4.222 | 3.778 | 3.778 |
| editorial_reviewer | agentic_selected | 4.433 | 3.402 | 4.237 | 3.825 | 3.773 |
| editorial_reviewer | generic_sft | 3.867 | 2.520 | 3.571 | 3.194 | 2.949 |
| editorial_reviewer | specificity_sft | 3.500 | 2.217 | 3.261 | 2.913 | 2.587 |
| growth_reader | zero_shot | 4.150 | 4.480 | 4.430 | 4.370 | 4.410 |
| growth_reader | original | 3.640 | 4.270 | 3.970 | 4.010 | 4.010 |
| growth_reader | agentic_candidate | 4.250 | 3.750 | 4.500 | 3.250 | 3.750 |
| growth_reader | agentic_selected | 3.948 | 3.794 | 4.093 | 3.619 | 3.660 |
| growth_reader | round2_final | 4.000 | 3.222 | 4.333 | 3.333 | 3.444 |
| growth_reader | generic_sft | 3.418 | 2.755 | 3.439 | 2.949 | 2.755 |
| growth_reader | specificity_sft | 3.217 | 2.435 | 3.239 | 2.739 | 2.500 |
| trust_sensitive_reader | zero_shot | 4.710 | 3.620 | 4.630 | 4.410 | 4.380 |
| trust_sensitive_reader | agentic_candidate | 4.750 | 3.250 | 4.750 | 4.000 | 4.250 |
| trust_sensitive_reader | original | 4.410 | 3.300 | 4.290 | 4.230 | 4.070 |
| trust_sensitive_reader | agentic_selected | 4.474 | 3.278 | 4.216 | 3.866 | 3.907 |
| trust_sensitive_reader | round2_final | 4.556 | 2.667 | 4.333 | 3.889 | 3.778 |
| trust_sensitive_reader | generic_sft | 4.041 | 2.602 | 3.704 | 3.367 | 3.306 |
| trust_sensitive_reader | specificity_sft | 3.674 | 2.304 | 3.457 | 3.087 | 2.935 |

## Persona Disagreement

| distinct_best_variants | seed_count |
| --- | --- |
| 1 | 55 |
| 2 | 41 |
| 3 | 4 |

## Example Persona Winners

| seed_id | persona | variant | headline | overall | rationale |
| --- | --- | --- | --- | --- | --- |
| 1 | busy_news_reader | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Concise and clearly communicates the main point. |
| 1 | editorial_reviewer | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Strong newsworthy headline that is well-rounded. |
| 1 | growth_reader | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Strong balance of clarity and engagement. |
| 1 | trust_sensitive_reader | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 4 | Factual, relevant, and straightforward. |
| 2 | busy_news_reader | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 4 | Concise and easy to understand. |
| 2 | editorial_reviewer | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Well-rounded headline that covers key aspects. |
| 2 | growth_reader | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings | 5 | Strong call to action that draws interest. |
| 2 | trust_sensitive_reader | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Accurate and trustworthy without sensationalism. |
| 3 | busy_news_reader | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Informative and succinct. |
| 3 | editorial_reviewer | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Well-rounded title. |
| 3 | growth_reader | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Engaging and informative, balances well. |
| 3 | trust_sensitive_reader | original | The Best Roller Coasters Around the World | 4 | Accurate and non-clickbait wording. |
| 4 | busy_news_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Concise and clear, great for quick consumption. |
| 4 | editorial_reviewer | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Strong headline; informative and engaging. |
| 4 | growth_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Straightforward and likely to attract broad interest. |
| 4 | trust_sensitive_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Clear and factual, accurately reflects the game's outcome. |
| 5 | busy_news_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Clear and concise, perfect for quick reading. |
| 5 | editorial_reviewer | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Well-rounded and informative. |
| 5 | growth_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Attention-grabbing while remaining factual. |
| 5 | trust_sensitive_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | The headline is straightforward and trustworthy. |
| 6 | busy_news_reader | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Concise and clear with welcoming details. |
| 6 | editorial_reviewer | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Strong on engagement and clarity. |
| 6 | growth_reader | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Engaging and informative. |
| 6 | trust_sensitive_reader | original | Local Airman returns from 6 Months in Afghanistan | 4 | Accurate and factual, no clickbait. |

## Next Use

- Merge persona `overall`, `trust`, and `engagement` scores into the multi-agent candidate matrix.
- Compare objective selectors with and without persona rewards.
- Use persona disagreement as evidence that headline quality is audience-dependent.
