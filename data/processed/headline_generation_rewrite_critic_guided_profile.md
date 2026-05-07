# Critic-Guided Rewrite Clickbait Penalty Profile

- Output: `data/processed/headline_generation_rewrite_critic_guided_scored_100.csv`
- Total rows: 100
- Rewrite target threshold: 0.50
- Rewritten target rows: 9

## Full 100-Row Comparison

- Original mean penalty: 0.2688
- Zero-shot mean penalty: 0.0879
- Final mean penalty: 0.0814
- Original clickbait rate: 27.00%
- Zero-shot clickbait rate: 9.00%
- Final clickbait rate: 8.00%

## Rewritten Target Rows Only

- Target zero-shot mean penalty: 0.9587
- Target rewritten mean penalty: 0.8865
- Mean delta vs zero-shot: -0.0721
- Median delta vs zero-shot: -0.0005
- Rows improved vs zero-shot: 88.89%
- Rows below threshold after rewrite: 11.11%
- Mean delta vs original: -0.0024

## Rewritten Examples

| Seed | Category | Zero-shot penalty | Rewrite penalty | Delta | Zero-shot title | Rewritten title |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 11 | lifestyle | 0.6379 | 0.0003 | -0.6376 | Decade Features Multiple Princesses, New Queen, King, and Future King of Pop | Decade Includes New Royal Figures and Future King of Pop |
| 86 | foodanddrink | 0.9917 | 0.9820 | -0.0096 | Hash Kitchen in Phoenix Features Arizona's Largest Bloody Mary Bar with 60 Ingredients | Hash Kitchen in Phoenix Offers 60 Ingredients at Its Bloody Mary Bar |
| 89 | lifestyle | 0.9999 | 0.9990 | -0.0009 | Michelle Mero Riedel's Oakdale Garden Perfectly Suited for Photography | Michelle Mero Riedel Maintains a Photogenic Garden in Oakdale |
| 90 | foodanddrink | 0.9999 | 0.9991 | -0.0007 | Food52 Highlights Chopped Salad Popularized by Nancy Silverton at Pizzeria Mozza | Food52 Features Chopped Salad Created by Nancy Silverton at Pizzeria Mozza |
| 73 | lifestyle | 0.9999 | 0.9994 | -0.0005 | Spooky Wedding Shoot Offers Dark Inspiration for Halloween Lovers | Halloween-Inspired Wedding Shoot Provides Ideas for Themed Celebrations |
| 37 | lifestyle | 0.9992 | 0.9990 | -0.0002 | Wedding Gowns Created from Quilted Northern Toilet Paper and Craft Supplies | Wedding Gowns Made from Quilted Northern Toilet Paper and Craft Materials |
| 63 | foodanddrink | 1.0000 | 1.0000 | -0.0000 | 30 Vintage Christmas Desserts Inspired by Grandma for Your Holiday Feast | Vintage Christmas Desserts Inspired by Grandma for Your Holiday Menu |
| 69 | lifestyle | 1.0000 | 1.0000 | -0.0000 | 13 Fascinating Facts About Manatees You May Not Know | Surprising Facts About Manatees You Might Not Know |
| 72 | health | 0.9995 | 0.9999 | 0.0004 | Connor Murphy's 2016 YouTube Video Highlights Photo Appearance Manipulation Techniques | Connor Murphy's 2016 Video Explores Techniques for Manipulating Appearance in Photos |

## Interpretation

This report measures whether critic-guided rewriting reduces the clickbait penalty for the subset of zero-shot headlines that remained above the threshold. It only evaluates clickbait style, not factual faithfulness or audience preference.
