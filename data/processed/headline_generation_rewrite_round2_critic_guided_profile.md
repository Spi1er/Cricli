# Round-2 Critic-Guided Rewrite Profile

- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/headline_generation_rewrite_round2_critic_guided_scored_100.csv`
- Total rows: 100
- Round-2 target rows: 7
- Threshold: 0.50

## Full 100-Row Comparison

| Stage | Mean penalty | Clickbait rate |
| --- | ---: | ---: |
| Original | 0.2688 | 27.00% |
| Zero-shot | 0.0879 | 9.00% |
| Round-1 final | 0.0755 | 7.00% |
| Round-2 final | 0.0656 | 6.00% |

## Round-2 Target Rows Only

| Metric | Value |
| --- | ---: |
| Round-1 rewritten mean penalty | 0.9988 |
| Round-2 mean penalty | 0.8565 |
| Mean delta vs round 1 | -0.1423 |
| Median delta vs round 1 | -0.0001 |
| Rows improved vs round 1 | 85.71% |
| Rows below threshold after round 2 | 14.29% |

## Round-2 Examples

| Seed | Category | Round-1 penalty | Round-2 penalty | Delta | Round-1 title | Round-2 title |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 89 | lifestyle | 0.9989 | 0.0003 | -0.9986 | Michelle Mero Riedel Maintains a Photogenic Garden in Oakdale | Michelle Mero Riedel Cultivates Garden in Oakdale for Photography |
| 69 | lifestyle | 0.9998 | 0.9981 | -0.0018 | Surprising Facts About Manatees and Their Unique Behaviors | Facts About Manatees and Their Characteristics |
| 63 | foodanddrink | 0.9999 | 0.9997 | -0.0002 | Vintage Christmas Desserts from Grandma's Cookbook for Holiday Celebrations | Vintage Christmas Desserts Featured in Taste of Home Article |
| 73 | lifestyle | 0.9999 | 0.9999 | -0.0001 | Halloween-Inspired Wedding Shoot Provides Ideas for Dark-Themed Celebrations | Halloween Wedding Shoot Offers Inspiration for Themed Celebrations |
| 90 | foodanddrink | 0.9999 | 0.9998 | -0.0001 | Food52 Features Chopped Salad Recipe by Nancy Silverton from Pizzeria Mozza | Food52 Highlights Chopped Salad Recipe from Nancy Silverton at Pizzeria Mozza |
| 72 | health | 0.9994 | 0.9993 | -0.0001 | Connor Murphy Discusses Techniques for Manipulating Appearance in 2016 Video | Connor Murphy Explains Appearance Manipulation Techniques in 2016 YouTube Video |
| 37 | lifestyle | 0.9937 | 0.9985 | 0.0048 | Wedding Gowns Made from Quilted Northern Toilet Paper and Craft Materials | Wedding Gowns Constructed from Quilted Northern Toilet Paper and Adhesives |

## Interpretation

Round 2 adds stricter lexical constraints after the clickbait critic still flags a title. This shows both the benefit and the limitation of prompt-based rewriting: some phrases remain highly scored because the penalty model has learned genre/style cues from lifestyle and listicle headlines.
