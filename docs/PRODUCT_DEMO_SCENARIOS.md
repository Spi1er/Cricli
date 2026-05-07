# Product Demo Scenarios

This document turns the Cricli research pipeline into a presentation-ready product story.

## Product Positioning

Cricli is a headline review and selection console for content teams. It is not primarily a headline generator.

The product promise is:

```text
Given an article summary and multiple candidate headlines,
Cricli evaluates risk, quality, audience fit, and business objective,
then recommends a publishable headline with an explanation.
```

## Target Users

| User | Need | Cricli Value |
| --- | --- | --- |
| News editor | Choose a publishable headline quickly | Compares human, GenAI, SFT, and agentic candidates with quality and risk scores |
| Growth team | Increase engagement without clickbait | Shows attractive options while keeping risk visible |
| Content reviewer | Check AI-generated headlines before release | Flags risk, persona fit, and objective tradeoffs |

## Presentation Flow

Use this sequence for a short live demo:

1. Open the review console.
2. Explain that each article has a hidden pool of candidate headlines.
3. Select one of the featured scenarios below.
4. Switch objectives and show that the recommended headline can change.
5. Explain the tradeoff using risk, quality, audience fit, and objective fit.

The key message is:

```text
Direct GenAI is strong, but teams still need a controllable decision layer.
```

The first three scenarios are the recommended presentation path. The last two are backup scenarios that help answer common questions about whether Cricli simply rejects GenAI or whether SFT contributes anything useful.

## Scenario 1: Objective Changes Recommendation

| Field | Value |
| --- | --- |
| Demo seed | `10` |
| Article category | sports |
| User | Content editor preparing a sports headline |
| Business question | Should the team prioritize editorial balance, growth, trust, or specificity? |
| Recommended demo objective | Growth |

Why this case works:

The same article produces different recommended sources under different objectives.

| Objective | Recommended Source | Recommended Headline |
| --- | --- | --- |
| Editorial | original | High school football: First-round playoff pairings |
| Growth | zero_shot | Oklahoma Playoffs Begin Next Week with First-Round Pairings Announced |
| Specificity | generic_sft | Oklahoma's first-round playoff pairings begin next week |
| Trust / Safety | original | High school football: First-round playoff pairings |

Presenter takeaway:

```text
There is no universal best headline. The best choice depends on the publishing objective.
```

## Scenario 2: Trust / Safety Prefers A Conservative Headline

| Field | Value |
| --- | --- |
| Demo seed | `2` |
| Article category | foodanddrink |
| User | Editor reviewing a lifestyle or food headline |
| Business question | Should a strong GenAI title be used directly, or should the team keep a safer editorial title? |
| Recommended demo objective | Trust / Safety |

Why this case works:

Zero-shot wins under editorial, growth, and specificity, but Trust / Safety chooses the original headline.

| Objective | Recommended Source | Recommended Headline |
| --- | --- | --- |
| Editorial | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings |
| Growth | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings |
| Specificity | zero_shot | Lodi Celebrates International Tempranillo Day with Winery Tour and Tastings |
| Trust / Safety | original | Tour of Tempranillo offers taste of Lodi's take on the Spanish varietal |

Presenter takeaway:

```text
GenAI can be excellent, but trust-sensitive surfaces may still prefer a conservative editorial baseline.
```

## Scenario 3: Specificity Selects A More Concrete Alternative

| Field | Value |
| --- | --- |
| Demo seed | `9` |
| Article category | news |
| User | News editor checking whether the title is specific enough |
| Business question | Can the selector move away from the default best baseline when the objective changes? |
| Recommended demo objective | Specificity |

Why this case works:

Editorial, growth, and trust/safety choose the zero-shot headline, while specificity chooses an agentic candidate.

| Objective | Recommended Source | Recommended Headline |
| --- | --- | --- |
| Editorial | zero_shot | Driver Fatally Shot in St. Paul, Police Investigate Incident |
| Growth | zero_shot | Driver Fatally Shot in St. Paul, Police Investigate Incident |
| Specificity | agentic_candidate | Driver Fatally Shot in St. Paul on Sunday Night |
| Trust / Safety | zero_shot | Driver Fatally Shot in St. Paul, Police Investigate Incident |

Presenter takeaway:

```text
Objective-specific selection can choose a more concrete candidate when specificity is the explicit goal.
```

## Scenario 4: Validated GenAI Baseline

| Field | Value |
| --- | --- |
| Demo seed | `1` |
| Article category | news |
| User | Editor reviewing whether a direct GenAI title is ready to publish |
| Business question | Can Cricli confirm when the GenAI baseline is already the best tradeoff? |
| Recommended demo objective | Editorial |

Why this case works:

The same zero-shot headline wins under all four objectives.

| Objective | Recommended Source | Recommended Headline |
| --- | --- | --- |
| Editorial | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks |
| Growth | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks |
| Specificity | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks |
| Trust / Safety | zero_shot | NASA Astronauts on ISS Face Challenges of Long Spacewalks |

Presenter takeaway:

```text
Cricli is not designed to reject GenAI by default. It can validate a strong GenAI headline when the scores and persona signals support it.
```

## Scenario 5: SFT Adds Specificity

| Field | Value |
| --- | --- |
| Demo seed | `14` |
| Article category | finance |
| User | Finance editor checking whether the headline contains enough concrete detail |
| Business question | Can an SFT candidate add value when the objective emphasizes specificity? |
| Recommended demo objective | Specificity |

Why this case works:

Zero-shot wins under editorial, growth, and trust/safety, but a generic SFT candidate wins under specificity.

| Objective | Recommended Source | Recommended Headline |
| --- | --- | --- |
| Editorial | zero_shot | Critics urge Procter & Gamble to address deforestation and use recycled materials |
| Growth | zero_shot | Critics urge Procter & Gamble to address deforestation and use recycled materials |
| Specificity | generic_sft | Critics say Procter & Gamble contributes to deforestation and should use recycled materials for paper products |
| Trust / Safety | zero_shot | Critics urge Procter & Gamble to address deforestation and use recycled materials |

Presenter takeaway:

```text
The SFT generator is not the main product claim, but it can contribute useful candidates to the review pool under specific objectives.
```

## What To Say In The Demo

Short script:

```text
Cricli is not trying to replace GPT headline generation.
Instead, it reviews a pool of human, GenAI, SFT, rewrite, and agentic candidates.
For each candidate, it exposes quality, risk, audience fit, and objective fit.
When the business objective changes, the recommendation can change too.
This is the missing decision layer between raw GenAI output and publishing.
```

## What Not To Overclaim

- Do not claim that Cricli always generates better headlines than GPT.
- Do not claim that persona votes are real user studies.
- Do not claim that the system is full online RL.
- Do not hide the tradeoff: direct GenAI remains very strong, and Cricli adds controllability and explanation.

## Demo Assets

| Asset | Use |
| --- | --- |
| `demo/headline_review_console.html` | Stable static demo and fallback |
| `demo/gradio_app.py` | Live interactive demo |
| `data/processed/headline_review_demo_cases.csv` | Compact static HTML case list |
| `data/processed/headline_review_demo_cases_full.csv` | Full 100-seed Gradio case explorer |
| `data/processed/headline_persona_calibrated_objective_selection.csv` | Objective-specific recommendation source |
| `data/processed/headline_audience_persona_votes.csv` | Persona preference signal source |
