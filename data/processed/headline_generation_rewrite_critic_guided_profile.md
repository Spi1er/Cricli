# Critic-Guided Rewrite Clickbait Penalty Profile

- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/headline_generation_rewrite_critic_guided_scored_100.csv`
- Total rows: 100
- Rewrite target threshold: 0.50
- Rewritten target rows: 9

## Full 100-Row Comparison

- Original mean penalty: 0.2688
- Zero-shot mean penalty: 0.0879
- Final mean penalty: 0.0755
- Original clickbait rate: 27.00%
- Zero-shot clickbait rate: 9.00%
- Final clickbait rate: 7.00%

## Rewritten Target Rows Only

- Target zero-shot mean penalty: 0.9587
- Target rewritten mean penalty: 0.8213
- Mean delta vs zero-shot: -0.1373
- Median delta vs zero-shot: -0.0002
- Rows improved vs zero-shot: 88.89%
- Rows below threshold after rewrite: 22.22%
- Mean delta vs original: -0.0676

## Rewritten Examples

| Seed | Category | Zero-shot penalty | Rewrite penalty | Delta | Zero-shot title | Rewritten title |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 86 | foodanddrink | 0.9917 | 0.0033 | -0.9884 | Hash Kitchen in Phoenix Features Arizona's Largest Bloody Mary Bar with 60 Ingredients | Hash Kitchen in Phoenix Offers 60-Ingredient Bloody Mary Bar |
| 11 | lifestyle | 0.6379 | 0.3972 | -0.2406 | Decade Features Multiple Princesses, New Queen, King, and Future King of Pop | Decade Includes Several Princesses, New Queen, and Future King of Pop |
| 37 | lifestyle | 0.9992 | 0.9937 | -0.0055 | Wedding Gowns Created from Quilted Northern Toilet Paper and Craft Supplies | Wedding Gowns Made from Quilted Northern Toilet Paper and Craft Materials |
| 89 | lifestyle | 0.9999 | 0.9989 | -0.0010 | Michelle Mero Riedel's Oakdale Garden Perfectly Suited for Photography | Michelle Mero Riedel Maintains a Photogenic Garden in Oakdale |
| 72 | health | 0.9995 | 0.9994 | -0.0002 | Connor Murphy's 2016 YouTube Video Highlights Photo Appearance Manipulation Techniques | Connor Murphy Discusses Techniques for Manipulating Appearance in 2016 Video |
| 69 | lifestyle | 1.0000 | 0.9998 | -0.0001 | 13 Fascinating Facts About Manatees You May Not Know | Surprising Facts About Manatees and Their Unique Behaviors |
| 63 | foodanddrink | 1.0000 | 0.9999 | -0.0000 | 30 Vintage Christmas Desserts Inspired by Grandma for Your Holiday Feast | Vintage Christmas Desserts from Grandma's Cookbook for Holiday Celebrations |
| 73 | lifestyle | 0.9999 | 0.9999 | -0.0000 | Spooky Wedding Shoot Offers Dark Inspiration for Halloween Lovers | Halloween-Inspired Wedding Shoot Provides Ideas for Dark-Themed Celebrations |
| 90 | foodanddrink | 0.9999 | 0.9999 | 0.0001 | Food52 Highlights Chopped Salad Popularized by Nancy Silverton at Pizzeria Mozza | Food52 Features Chopped Salad Recipe by Nancy Silverton from Pizzeria Mozza |

## Interpretation

This report measures whether critic-guided rewriting reduces the clickbait penalty for the subset of zero-shot headlines that remained above the threshold. It only evaluates clickbait style, not factual faithfulness or audience preference.
