# Headline SFT Dataset Profile

This dataset is the first business-driven generator training layer. It creates one broad headline-generation SFT set and one specificity-aware SFT set for comparing whether targeted data and instructions improve title quality.

## Configuration

- Input: `/Users/pesun/STAT 5293 GenAI with LLM/Cricli/projects/data/processed/mind_headline_pool_with_clickbait_penalty.csv`
- Source field: `summary`
- Minimum source words: 20
- Title word range: 5-18
- Specificity max clickbait penalty: 0.2
- Require specificity signal: False
- Random state: 5293

## Dataset Sizes

| Variant | Rows | Train | Val | Test | Mean Clickbait Penalty | Clickbait Rate | Specific Signal Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | 6,793 | 5,448 | 658 | 687 | 0.1876 | 18.65% | 88.69% |
| specificity | 5,469 | 4,409 | 512 | 548 | 0.0022 | 0.00% | 88.33% |

## Generic Top Categories

| Value | Rows |
| --- | ---: |
| news | 2,498 |
| sports | 1,924 |
| finance | 382 |
| travel | 333 |
| weather | 300 |
| foodanddrink | 280 |
| lifestyle | 275 |
| health | 221 |
| video | 181 |
| autos | 151 |
| music | 81 |
| tv | 68 |

## Specificity Top Categories

| Value | Rows |
| --- | ---: |
| news | 2,269 |
| sports | 1,632 |
| finance | 330 |
| weather | 276 |
| travel | 250 |
| video | 151 |
| autos | 131 |
| health | 122 |
| lifestyle | 115 |
| foodanddrink | 67 |
| tv | 43 |
| music | 36 |

## Training Use

- M1 generic SFT: train on `headline_sft_generic_train.csv`, validate on `headline_sft_generic_val.csv`, test on `headline_sft_generic_test.csv`.
- M2 specificity-aware SFT: train on `headline_sft_specificity_train.csv`, validate on `headline_sft_specificity_val.csv`, test on `headline_sft_specificity_test.csv`.
- Evaluate both models on the same held-out generation seed or the SFT test split using local critics and LLM judge.
- Treat M2 as the first policy model aligned with the proposal goal: faithful, specific, non-clickbait headline generation.

## Example Rows

### Generic

| split   | category   | source_text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | target_title                                                          |
|:--------|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------|
| train   | lifestyle  | Maybe you're desperate to come up with presents for loved ones this holiday season, but you're going to want to scratch those scented candles and holiday sweaters off the gift list and think again about what friends and family will really want to unwrap. Some items are better left on store shelves.                                                                                                                                                                                                                                                                                                                                                                                                                                     | 25 Gifts No One Wants to Get This Holiday Season                      |
| train   | news       | The federal judge presiding over the case of ex-National Security Adviser Michael Flynn has canceled a hearing scheduled for next week.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Flynn hearing canceled after lawyer claims FBI manipulated files      |
| train   | lifestyle  | From Amazon to your computer's power button, these objects, brands, and photos have hidden symbols, surprising origins, or lesser-known meanings.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 28 objects and photos that have hidden signs or symbols               |
| train   | sports     | A two-time Pro Bowler, Darius Slay did not directly answer when asked Thursday whether he wants to stay with the Detroit Lions at the trade deadline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Detroit Lions' Darius Slay: 'Nobody's safe' after Quandre Diggs trade |
| train   | video      | Passengers aboard a cruise visiting Wilsons Promontory off the coast of Victoria, Australia, were treated to a double whale breach on November 3. Rosie Morgan, a passenger on a cruise run by Pennicott Wilderness Journeys, captured footage of two humpback whales (a mother and a calf) breaching at once, a sight which left her and other passengers in stunned delight. Pennicott Wilderness Journeys shared the footage to Facebook, summing up the excitement of those on board with the words, "this is just a little bit awesome." The post added, "Skipper Robert Pennicott found this mum and bub having fun in the Wilsons Promontory Marine National Park today. What a wonderful experience." Credit: Rosie Morgan via Storyful | Amazing Double Breach Leaves Whale Watchers Stunned                   |

### Specificity

| split   | category   | source_text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | target_title                                                           |
|:--------|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
| train   | news       | The federal judge presiding over the case of ex-National Security Adviser Michael Flynn has canceled a hearing scheduled for next week.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Flynn hearing canceled after lawyer claims FBI manipulated files       |
| train   | sports     | A two-time Pro Bowler, Darius Slay did not directly answer when asked Thursday whether he wants to stay with the Detroit Lions at the trade deadline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Detroit Lions' Darius Slay: 'Nobody's safe' after Quandre Diggs trade  |
| train   | video      | Passengers aboard a cruise visiting Wilsons Promontory off the coast of Victoria, Australia, were treated to a double whale breach on November 3. Rosie Morgan, a passenger on a cruise run by Pennicott Wilderness Journeys, captured footage of two humpback whales (a mother and a calf) breaching at once, a sight which left her and other passengers in stunned delight. Pennicott Wilderness Journeys shared the footage to Facebook, summing up the excitement of those on board with the words, "this is just a little bit awesome." The post added, "Skipper Robert Pennicott found this mum and bub having fun in the Wilsons Promontory Marine National Park today. What a wonderful experience." Credit: Rosie Morgan via Storyful | Amazing Double Breach Leaves Whale Watchers Stunned                    |
| train   | news       | In May 2017, former Republican Rep. Leonard Lance crossed party lines and voted against the GOP health care repeal, a proposal deeply unpopular with voters in New Jersey's 7th District, which he had represented in Washington for nearly a decade. A year later, Lance again joined Democrats to oppose the Republican tax cut bill. Although he supported portions of the bill and its overall intent, he decided to vote against it because it would hurt...                                                                                                                                                                                                                                                                               | Decline of local journalism is likely increasing voter polarization    |
| train   | finance    | Former president of Scottsdale Community College and community leader Art DeCabooter died on Wednesday after a long bout with Parkinson's disease.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Art DeCabooter, longtime Scottsdale Community College head, dies at 78 |
