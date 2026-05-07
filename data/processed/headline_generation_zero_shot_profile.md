# Zero-Shot Headline Clickbait Penalty Profile

- Output: `data/processed/headline_generation_zero_shot_scored_100.csv`
- Rows: 100
- Threshold: 0.50
- Original mean penalty: 0.2688
- Zero-shot mean penalty: 0.0891
- Mean delta (zero-shot - original): -0.1797
- Median delta: -0.0001
- Improved rows: 62.00%
- Worsened rows: 38.00%
- Original predicted clickbait rate: 27.00%
- Zero-shot predicted clickbait rate: 9.00%

## Category Profile

| Category | Rows | Original mean | Zero-shot mean | Mean delta | Original rate | Zero-shot rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| health | 10 | 0.5000 | 0.1820 | -0.3180 | 50.00% | 20.00% |
| travel | 8 | 0.3586 | 0.0415 | -0.3171 | 37.50% | 0.00% |
| lifestyle | 10 | 0.7002 | 0.3998 | -0.3004 | 70.00% | 40.00% |
| autos | 5 | 0.2000 | 0.0001 | -0.1999 | 20.00% | 0.00% |
| weather | 5 | 0.1709 | 0.0001 | -0.1708 | 20.00% | 0.00% |
| news | 25 | 0.1524 | 0.0018 | -0.1506 | 16.00% | 0.00% |
| foodanddrink | 7 | 0.4290 | 0.2875 | -0.1415 | 42.86% | 28.57% |
| sports | 20 | 0.1642 | 0.0293 | -0.1349 | 15.00% | 5.00% |
| finance | 10 | 0.0060 | 0.0118 | 0.0058 | 0.00% | 0.00% |

## Biggest Penalty Reductions

| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |
| ---: | ---: | ---: | --- | --- | --- |
| -0.9998 | 1.0000 | 0.0001 | health | 8 Questions to Ask Yourself Before Using CBD | CBD Now Available in Foods, Drinks, and Skincare Products |
| -0.9998 | 0.9999 | 0.0001 | lifestyle | Help find photos of these 11 Vietnam Veterans from Detroit | Wall of Faces campaign seeks missing photos of 11 Detroit Vietnam veterans |
| -0.9998 | 1.0000 | 0.0002 | news | 5 delightful doggies to adopt now in Seattle | Dogs Available for Adoption at Seattle Area Pet Centers |
| -0.9998 | 0.9999 | 0.0002 | lifestyle | Adam's Corner celebrates the bravery of our Armed Forces and bring joy to children \| Opinion | Adam's Corner and Fisher House Offer Support for Military Families |
| -0.9997 | 0.9999 | 0.0002 | news | NASA's Christina Koch got a little bit messy during first all-female spacewalk | NASA astronauts face challenges and physical demands during spacewalks on ISS |
| -0.9997 | 1.0000 | 0.0003 | lifestyle | 11 Powerful Products That Cut Your Cleaning Time in Half | Efficient Cleaning Products Fight Stains and Scents Without Harsh Chemicals |
| -0.9997 | 1.0000 | 0.0003 | health | How This Guy Lost 40 Pounds and Got Shredded for His 40th Birthday | Aaron Archer loses 40 pounds through HIIT workouts and calorie tracking |
| -0.9996 | 1.0000 | 0.0003 | lifestyle | 50+ Amazing Things That Happened in the '50s | New decade introduces multiple princesses, a new Queen, and future King of Pop |
| -0.9996 | 0.9996 | 0.0001 | travel | Top things to do in Tampa Bay this weekend: Oct. 18-20 | Tampa Theatre Announces Live Shows for Halloween Film Festival |
| -0.9995 | 0.9996 | 0.0001 | sports | Do some couch scouting for Dolphins draft: Here are top prospects to watch this weekend | Miami Dolphins Consider Top College Prospects Ahead of 2020 NFL Draft |
| -0.9995 | 0.9999 | 0.0005 | travel | The Best Roller Coasters Around the World | Roller Coasters Today: Speed Records and Thrilling Designs |
| -0.9993 | 0.9994 | 0.0002 | autos | Horsepower! The 1,500hp Hellcat King | Duane Roots' 1,500hp Charger Hellcat Powered by E90 and Nitrous |

## Biggest Penalty Increases

| Delta | Original penalty | Zero-shot penalty | Category | Original title | Zero-shot title |
| ---: | ---: | ---: | --- | --- | --- |
| 0.9993 | 0.0006 | 0.9999 | lifestyle | Photographer tends a picture-perfect garden in Oakdale | Michelle Mero Riedel's Oakdale garden offers a perfect backdrop for photography |
| 0.3241 | 0.0002 | 0.3243 | travel | Texas quail rebounding after a dismal 2018 | Bobwhite quail hunting remains a cherished tradition in Texas |
| 0.0798 | 0.0001 | 0.0799 | finance | A Scottsdale mansion with a rooftop view deck, home theater sells for $2.65M | Nearly 8,000 Square-Foot Scottsdale Mansion Among Phoenix's Priciest Sales This Week |
| 0.0363 | 0.0005 | 0.0368 | finance | What apartments will $900 rent you in Pleasant Valley right now? | Apartment Rentals in Pleasant Valley: What $900/Month Can Get You |
| 0.0318 | 0.0007 | 0.0325 | foodanddrink | San Francisco chefs find nostalgia in Japanese milk bread | San Francisco Chefs Embrace Milk Bread Despite Sourdough Culture |
| 0.0032 | 0.0001 | 0.0034 | sports | At-track photos: 2019 Kansas playoff weekend | Photos from NASCAR’s playoff weekend at Kansas Speedway available now |
| 0.0014 | 0.0001 | 0.0015 | foodanddrink | 2 more business openings on tap at Hill Center Brentwood | Eat the Frog Fitness and MOOYAH Burgers opening at Hill Center Brentwood |
| 0.0007 | 0.0001 | 0.0009 | news | Trump equates the Smollett case to impeachment inquiry | Trump Labels Jussie Smollett Hate Crime Allegation and Impeachment a "Scam |
| 0.0003 | 0.0001 | 0.0004 | news | Homeless Become More Visible in Austin, Sparking Political Clash | Texan Officials Criticize Austin's Homelessness Policies Amid Rising Encampments |
| 0.0002 | 0.0001 | 0.0003 | finance | Environmentalists' new target? Charmin toilet paper | Critics urge Procter & Gamble to reduce deforestation and use recycled materials |
| 0.0002 | 0.0001 | 0.0003 | news | Toxic PCBs linger in schools; EPA, lawmakers fail to act | Teachers at Sky Valley Education Center evacuate students after fluorescent lights catch fire |
| 0.0002 | 0.0001 | 0.0003 | finance | Online petition to keep divisive Braves tomahawk chop nears 60,000 | Kevin Mooneyhan expresses concern over losing tomahawk chop after NLDS Game 5 loss |

## Interpretation

Negative deltas mean the API zero-shot headline is less clickbait-like according to the fine-tuned DistilBERT penalty critic. This does not evaluate faithfulness or attractiveness; those need separate critics or human/LLM-judge evaluation.
