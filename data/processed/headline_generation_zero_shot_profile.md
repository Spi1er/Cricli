# Zero-Shot Headline Clickbait Penalty Profile

- Output: `/Users/pesun/STAT 5293 GenAI with LLM/Circli/projects/data/processed/headline_generation_zero_shot_scored_100.csv`
- Rows: 100
- Threshold: 0.50
- Original mean penalty: 0.2688
- Zero-shot mean penalty: 0.0879
- Mean delta (zero-shot - original): -0.1810
- Median delta: -0.0001
- Improved rows: 71.00%
- Worsened rows: 29.00%
- Original predicted clickbait rate: 27.00%
- Zero-shot predicted clickbait rate: 9.00%

## Category Profile

| Category | Rows | Original mean | Zero-shot mean | Mean delta | Original rate | Zero-shot rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| health | 10 | 0.5000 | 0.1001 | -0.3999 | 50.00% | 10.00% |
| travel | 8 | 0.3586 | 0.0016 | -0.3569 | 37.50% | 0.00% |
| lifestyle | 10 | 0.7002 | 0.4638 | -0.2364 | 70.00% | 50.00% |
| autos | 5 | 0.2000 | 0.0001 | -0.1999 | 20.00% | 0.00% |
| weather | 5 | 0.1709 | 0.0001 | -0.1708 | 20.00% | 0.00% |
| sports | 20 | 0.1642 | 0.0003 | -0.1640 | 15.00% | 0.00% |
| news | 25 | 0.1524 | 0.0054 | -0.1471 | 16.00% | 0.00% |
| finance | 10 | 0.0060 | 0.0002 | -0.0058 | 0.00% | 0.00% |
| foodanddrink | 7 | 0.4290 | 0.4275 | -0.0015 | 42.86% | 42.86% |

## Biggest Penalty Reductions

| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |
| ---: | ---: | ---: | --- | --- | --- |
| -0.9998 | 0.9999 | 0.0001 | lifestyle | Help find photos of these 11 Vietnam Veterans from Detroit | Campaign seeks missing photos of 11 Detroit veterans from Vietnam Memorial |
| -0.9998 | 0.9999 | 0.0001 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | NASA Astronauts on ISS Face Challenges of Long Spacewalks |
| -0.9998 | 1.0000 | 0.0002 | news | 5 delightful doggies to adopt now in Seattle | Dogs Available for Adoption at Centers in Seattle Area |
| -0.9998 | 0.9999 | 0.0001 | lifestyle | Adam's Corner celebrates the bravery of our Armed Forces and bring joy to children \| Opinion | Adam's Corner and Fisher House Offer Support for Military Families |
| -0.9998 | 1.0000 | 0.0002 | health | How This Guy Lost 40 Pounds and Got Shredded for His 40th Birthday | Aaron Archer loses 40 pounds through HIIT workouts and calorie tracking |
| -0.9998 | 1.0000 | 0.0002 | lifestyle | 11 Powerful Products That Cut Your Cleaning Time in Half | Effective Cleaning Products That Save Time and Avoid Harsh Chemicals |
| -0.9997 | 1.0000 | 0.0003 | health | 8 Questions to Ask Yourself Before Using CBD | CBD is increasingly available in food, drinks, and skincare products |
| -0.9996 | 0.9999 | 0.0004 | travel | The Best Roller Coasters Around the World | Roller Coasters Today: Speed Records and Thrilling Designs |
| -0.9995 | 0.9996 | 0.0001 | travel | Top things to do in Tampa Bay this weekend: Oct. 18-20 | Tampa Theatre Hosts Halloween Film Festival with Live Shows and Talks |
| -0.9995 | 0.9996 | 0.0001 | sports | Do some couch scouting for Dolphins draft: Here are top prospects to watch this weekend | Dolphins Prepare for NFL Draft, Showcase Top College Prospects in Week 11 |
| -0.9995 | 0.9999 | 0.0004 | health | You Asked: What's the Most Effective Machine to Use at the Gym? | Maximizing Gym Workouts in 20 to 30 Minutes for Targeted Muscle Training |
| -0.9992 | 0.9994 | 0.0002 | autos | Horsepower! The 1,500hp Hellcat King | Duane Roots' 1,500hp Charger Hellcat Features E90 Fuel and Nitrous Boost |

## Biggest Penalty Increases

| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |
| ---: | ---: | ---: | --- | --- | --- |
| 0.9993 | 0.0006 | 0.9999 | lifestyle | Photographer tends a picture-perfect garden in Oakdale | Michelle Mero Riedel's Oakdale Garden Perfectly Suited for Photography |
| 0.0108 | 0.0002 | 0.0110 | travel | Texas quail rebounding after a dismal 2018 | Hunting bobwhite quail remains a cherished tradition in Texas |
| 0.0008 | 0.0002 | 0.0011 | news | Last month hottest October on record: EU climate service | October 2023 recorded as the hottest October globally, exceeding previous records |
| 0.0007 | 0.0001 | 0.0009 | finance | A Scottsdale mansion with a rooftop view deck, home theater sells for $2.65M | Scottsdale Mansion Among Metro Phoenix's Priciest Home Sales This Week |
| 0.0006 | 0.0001 | 0.0007 | foodanddrink | 2 more business openings on tap at Hill Center Brentwood | Eat the Frog Fitness and MOOYAH Burgers to Open at Hill Center Brentwood |
| 0.0003 | 0.0001 | 0.0005 | news | Trump equates the Smollett case to impeachment inquiry | Trump Compares Jussie Smollett Hate Crime Allegation to Impeachment Process |
| 0.0003 | 0.0001 | 0.0004 | news | Hartford's Weaver forging a new identity at campus shared by public, magnet students | Students at Hartford's Weaver campus aim to unite two school communities |
| 0.0002 | 0.0001 | 0.0003 | finance | Online petition to keep divisive Braves tomahawk chop nears 60,000 | Kevin Mooneyhan faces disappointment after Game 5 loss and losing tomahawk chop |
| 0.0001 | 0.0002 | 0.0003 | news | Sacramento Man Takes Wildfire Evacuees And Their 18 Dogs | Sacramento man houses Healdsburg couple and their 18 dogs, including puppies |
| 0.0001 | 0.0001 | 0.0002 | sports | Giovani Bernard suffers knee injury vs. Ravens, eventually returns | Bengals' Giovani Bernard suffers knee injury during Week 10 game against Ravens |
| 0.0001 | 0.0001 | 0.0001 | lifestyle | Fire damages St. Bernadette Catholic School in south Seattle | St. Bernadette Catholic School Cancels Classes After Overnight Fire Damage |
| 0.0000 | 0.0001 | 0.0001 | finance | Environmentalists' new target? Charmin toilet paper | Critics urge Procter & Gamble to address deforestation and use recycled materials |

## Interpretation

Negative deltas mean the API zero-shot headline is less clickbait-like according to the fine-tuned DistilBERT penalty critic. This does not evaluate faithfulness or attractiveness; those need separate critics or human/LLM-judge evaluation.
