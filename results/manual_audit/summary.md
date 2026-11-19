# Observed manual audit

Observed one reviewer's judgments on 30 selected Urban degradation cases from e0 v2 seed 0 with the linear draft. The current e6 submission uses a different draft and refiner.
The sample was selected for degradation and does not estimate dataset-wide category prevalence.

A denotes unlabeled real roads. B denotes incomplete or truncated labels. C denotes incorrect model additions. D denotes almost no red. Mixed labels retain the reviewer's original ordering.
Added precision is computed against the benchmark annotation. real_road_pct is the reviewer's percentage of red pixels judged to be real roads.

| red_main | Cases | Median delta A4-y0 | Median added precision | Median real_road_pct | Rated cases |
|---|---:|---:|---:|---:|---:|
| A | 20 | -0.026058325 | 0.061495343 | 100.000000 | 20 |
| A+C | 2 | -0.009050731 | 0.254548024 | 50.000000 | 2 |
| B | 4 | -0.024432024 | 0.212549912 | 100.000000 | 4 |
| B+A | 1 | -0.022150467 | 0.183889602 | 100.000000 | 1 |
| C | 1 | -0.051730695 | 0.200181433 | 0.000000 | 1 |
| D | 2 | -0.005462629 | 0.309468284 | 0.000000 | 1 |

Observed 30 joined rows with every audit_no from 1 through 30. Observed 29 paired percentage/precision ratings; missing percentages are excluded.
Observed median human real-road fraction 1.000000000, median annotated added precision 0.133603239, and median paired human-minus-annotation difference 0.824129695.
Observed Pearson correlation between human real-road fraction and annotated added precision -0.539789779.

Observed the following descriptive groups by the supplied human percentage.

| real_road_pct | Cases | Median added precision |
|---|---:|---:|
| 0 | 2 | 0.409559000 |
| 25 | 1 | 0.133603239 |
| 50 | 2 | 0.254548024 |
| 75 | 4 | 0.280371928 |
| 100 | 20 | 0.060588081 |
