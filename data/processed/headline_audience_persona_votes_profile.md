# Audience Persona Voting Profile

- Input: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/headline_audience_persona_votes.csv`
- Completed seed count: 90
- Persona count: 4
- Vote rows: 1,676

## Interpretation

Persona voting is the proposal-aligned audience layer. It should be treated as an evaluator module that estimates how different audience goals prefer different candidate headlines.

## Consensus Best Counts

| variant | consensus_best_count |
| --- | --- |
| zero_shot | 49 |
| original | 40 |
| generic_sft | 1 |

## Persona Best Counts

| persona | variant | best_count |
| --- | --- | --- |
| busy_news_reader | zero_shot | 50 |
| busy_news_reader | original | 39 |
| busy_news_reader | generic_sft | 1 |
| editorial_reviewer | zero_shot | 48 |
| editorial_reviewer | original | 40 |
| editorial_reviewer | generic_sft | 2 |
| growth_reader | zero_shot | 53 |
| growth_reader | original | 34 |
| growth_reader | agentic_selected | 2 |
| growth_reader | agentic_candidate | 1 |
| trust_sensitive_reader | original | 45 |
| trust_sensitive_reader | zero_shot | 44 |
| trust_sensitive_reader | generic_sft | 1 |

## Mean Scores By Persona And Variant

| persona | variant | trust | engagement | clarity | audience_fit | overall |
| --- | --- | --- | --- | --- | --- | --- |
| busy_news_reader | zero_shot | 4.656 | 3.756 | 4.756 | 4.511 | 4.456 |
| busy_news_reader | original | 4.100 | 3.211 | 4.356 | 4.100 | 3.967 |
| busy_news_reader | agentic_candidate | 4.400 | 3.300 | 4.500 | 3.900 | 3.900 |
| busy_news_reader | agentic_selected | 4.326 | 3.169 | 4.270 | 3.775 | 3.663 |
| busy_news_reader | round2_final | 4.444 | 3.111 | 4.556 | 3.556 | 3.556 |
| busy_news_reader | generic_sft | 3.852 | 2.455 | 3.727 | 3.136 | 2.875 |
| busy_news_reader | specificity_sft | 3.605 | 2.186 | 3.209 | 2.628 | 2.419 |
| editorial_reviewer | zero_shot | 4.789 | 4.044 | 4.756 | 4.556 | 4.511 |
| editorial_reviewer | original | 4.289 | 3.556 | 4.278 | 4.111 | 4.089 |
| editorial_reviewer | agentic_candidate | 4.500 | 3.300 | 4.500 | 4.000 | 3.800 |
| editorial_reviewer | agentic_selected | 4.472 | 3.461 | 4.213 | 3.831 | 3.798 |
| editorial_reviewer | round2_final | 4.444 | 3.111 | 4.444 | 3.667 | 3.556 |
| editorial_reviewer | generic_sft | 3.875 | 2.670 | 3.705 | 3.239 | 3.034 |
| editorial_reviewer | specificity_sft | 3.605 | 2.326 | 3.233 | 2.721 | 2.488 |
| growth_reader | zero_shot | 4.256 | 4.533 | 4.467 | 4.478 | 4.489 |
| growth_reader | original | 3.611 | 4.178 | 3.956 | 4.067 | 4.078 |
| growth_reader | agentic_candidate | 4.200 | 3.800 | 4.300 | 3.700 | 3.800 |
| growth_reader | agentic_selected | 3.933 | 3.697 | 4.101 | 3.730 | 3.685 |
| growth_reader | round2_final | 4.111 | 3.111 | 4.111 | 3.222 | 3.333 |
| growth_reader | generic_sft | 3.375 | 2.909 | 3.511 | 2.841 | 2.795 |
| growth_reader | specificity_sft | 3.279 | 2.628 | 3.140 | 2.465 | 2.442 |
| trust_sensitive_reader | zero_shot | 4.778 | 3.722 | 4.644 | 4.544 | 4.422 |
| trust_sensitive_reader | original | 4.456 | 3.322 | 4.311 | 4.267 | 4.111 |
| trust_sensitive_reader | agentic_candidate | 4.500 | 3.300 | 4.500 | 4.200 | 4.000 |
| trust_sensitive_reader | agentic_selected | 4.539 | 3.404 | 4.292 | 4.011 | 4.000 |
| trust_sensitive_reader | round2_final | 4.444 | 3.111 | 4.333 | 3.667 | 3.556 |
| trust_sensitive_reader | generic_sft | 4.114 | 2.682 | 3.886 | 3.489 | 3.352 |
| trust_sensitive_reader | specificity_sft | 3.791 | 2.535 | 3.488 | 3.023 | 2.884 |

## Persona Disagreement

| distinct_best_variants | seed_count |
| --- | --- |
| 1 | 46 |
| 2 | 43 |
| 3 | 1 |

## Example Persona Winners

| seed_id | persona | variant | headline | overall | rationale |
| --- | --- | --- | --- | --- | --- |
| 1 | busy_news_reader | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Concise and highly informative. |
| 1 | editorial_reviewer | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Comprehensive and professional. |
| 1 | growth_reader | original | NASA's Christina Koch got a little bit messy during first all-female spacewalk | 4 | Engaging but could be perceived as slightly informal. |
| 1 | trust_sensitive_reader | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 5 | Clear and factual, indicating specific challenges faced. |
| 2 | busy_news_reader | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Concise and clear, conveys essential information. |
| 2 | editorial_reviewer | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Well-balanced and conveys both the event and the varietal. |
| 2 | growth_reader | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Captivating headline that promises a unique experience. |
| 2 | trust_sensitive_reader | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal | 5 | Accurate and clear about the event and grape variety. |
| 3 | busy_news_reader | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Concise and informative; good balance. |
| 3 | editorial_reviewer | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Balanced in terms of engagement and specifics. |
| 3 | growth_reader | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Highly engaging with relevant content. |
| 3 | trust_sensitive_reader | zero_shot | Roller Coasters Today: Speed Records and Thrilling Designs | 4 | Informative and clear, slightly more engaging. |
| 4 | busy_news_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Concise and informative; ideal for busy readers. |
| 4 | editorial_reviewer | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Strong headline; covers key aspects well. |
| 4 | growth_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | Very appealing and direct, perfect for clicks. |
| 4 | trust_sensitive_reader | zero_shot | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 5 | This is a straightforward and factual headline. |
| 5 | busy_news_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Concise and informative. |
| 5 | editorial_reviewer | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Well-balanced; accurate and engaging. |
| 5 | growth_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Engaging and clear; likely to attract clicks. |
| 5 | trust_sensitive_reader | original | Depth charts: Projected starters for Iowa State vs. No. 23 Texas | 5 | Accurate and straightforward, no sensationalism. |
| 6 | busy_news_reader | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Clear and informative with warm detail. |
| 6 | editorial_reviewer | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Well-rounded, detailed, and engaging. |
| 6 | growth_reader | zero_shot | Local airman returns home to Miami Valley to warm welcome from family and friends | 5 | Highly engaging and relatable. |
| 6 | trust_sensitive_reader | original | Local Airman returns from 6 Months in Afghanistan | 5 | Factual and straightforward, avoids exaggeration. |

## Next Use

- Merge persona `overall`, `trust`, and `engagement` scores into the multi-agent candidate matrix.
- Compare objective selectors with and without persona rewards.
- Use persona disagreement as evidence that headline quality is audience-dependent.
