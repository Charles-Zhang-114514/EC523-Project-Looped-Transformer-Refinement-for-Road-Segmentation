# Observed draft overfit check

Observed Train samples use seed 0, without replacement, independently in each domain. Val-rep scores below are read from existing reports, not recomputed for this comparison. Train and Val distribution differences affect both heads; the registered statistic is the difference of gaps.

| Head | Set | Domain | K | IoU | clDice | BIoU | A |
|---|---|---|---|---:|---:|---:|---:|
| linear | Train subset | Overall | y0_draft | 0.662302217 | 0.727406769 | 0.086891632 | 0.566751465 |
| linear | Train subset | Overall | 1 | 0.643825516 | 0.703890695 | 0.079003934 | 0.548880753 |
| linear | Train subset | Overall | 2 | 0.643857401 | 0.703346039 | 0.079054450 | 0.548743402 |
| linear | Train subset | Overall | 4 | 0.648517637 | 0.707541224 | 0.081406785 | 0.552802542 |
| linear | Train subset | Urban | y0_draft | 0.675689779 | 0.725986030 | 0.065638178 | 0.568768334 |
| linear | Train subset | Urban | 1 | 0.658848386 | 0.701670473 | 0.060458366 | 0.552017008 |
| linear | Train subset | Urban | 2 | 0.659709326 | 0.700894589 | 0.061013383 | 0.552325716 |
| linear | Train subset | Urban | 4 | 0.664960780 | 0.706398243 | 0.063353535 | 0.557070570 |
| linear | Train subset | Rural | y0_draft | 0.607730791 | 0.731786649 | 0.128431061 | 0.549087603 |
| linear | Train subset | Rural | 1 | 0.584828459 | 0.709216532 | 0.114807284 | 0.528140646 |
| linear | Train subset | Rural | 2 | 0.582143162 | 0.708996643 | 0.113729136 | 0.526516401 |
| linear | Train subset | Rural | 4 | 0.584881782 | 0.710204188 | 0.115861426 | 0.528674433 |
| linear | Val-rep (recorded) | Overall | y0_draft | 0.617080422 | 0.667221040 | 0.086717549 | 0.526050033 |
| linear | Val-rep (recorded) | Overall | 1 | 0.616915429 | 0.682670191 | 0.093347342 | 0.531928240 |
| linear | Val-rep (recorded) | Overall | 2 | 0.616530061 | 0.682627795 | 0.093620495 | 0.531777468 |
| linear | Val-rep (recorded) | Overall | 4 | 0.617819735 | 0.683572362 | 0.094275029 | 0.532836582 |
| linear | Val-rep (recorded) | Urban | y0_draft | 0.671417590 | 0.714745205 | 0.082375189 | 0.566607394 |
| linear | Val-rep (recorded) | Urban | 1 | 0.661501968 | 0.707798484 | 0.083750010 | 0.559840531 |
| linear | Val-rep (recorded) | Urban | 2 | 0.660478505 | 0.706502100 | 0.083997549 | 0.558989393 |
| linear | Val-rep (recorded) | Urban | 4 | 0.662532884 | 0.709309745 | 0.084681886 | 0.560995743 |
| linear | Val-rep (recorded) | Rural | y0_draft | 0.506060429 | 0.610699900 | 0.091814653 | 0.454603115 |
| linear | Val-rep (recorded) | Rural | 1 | 0.526090598 | 0.655231257 | 0.104629414 | 0.480540559 |
| linear | Val-rep (recorded) | Rural | 2 | 0.526788142 | 0.656846851 | 0.104997867 | 0.481447700 |
| linear | Val-rep (recorded) | Rural | 4 | 0.526760614 | 0.655549601 | 0.105608639 | 0.481166915 |
| A1-i | Train subset | Overall | y0_draft | 0.628957689 | 0.724263041 | 0.072980697 | 0.546353896 |
| A1-i | Train subset | Overall | 1 | 0.639791557 | 0.731420606 | 0.077118670 | 0.554745694 |
| A1-i | Train subset | Overall | 2 | 0.650371826 | 0.740809332 | 0.081252765 | 0.563679266 |
| A1-i | Train subset | Overall | 4 | 0.667868450 | 0.751832625 | 0.088694116 | 0.577222836 |
| A1-i | Train subset | Urban | y0_draft | 0.632996454 | 0.706277274 | 0.046217746 | 0.537624959 |
| A1-i | Train subset | Urban | 1 | 0.644735124 | 0.715012592 | 0.049930357 | 0.546857411 |
| A1-i | Train subset | Urban | 2 | 0.656090892 | 0.726215560 | 0.053609451 | 0.556632005 |
| A1-i | Train subset | Urban | 4 | 0.674655071 | 0.738815292 | 0.059622838 | 0.570896691 |
| A1-i | Train subset | Rural | y0_draft | 0.611586770 | 0.761774184 | 0.125887016 | 0.559503043 |
| A1-i | Train subset | Rural | 1 | 0.618721142 | 0.764902134 | 0.130147417 | 0.564860695 |
| A1-i | Train subset | Rural | 2 | 0.626174526 | 0.769940862 | 0.134429393 | 0.570955400 |
| A1-i | Train subset | Rural | 4 | 0.639431212 | 0.777588304 | 0.143813337 | 0.581754765 |
| A1-i | Val-rep (recorded) | Overall | y0_draft | 0.622292764 | 0.695215840 | 0.092674934 | 0.538246121 |
| A1-i | Val-rep (recorded) | Overall | 1 | 0.625810513 | 0.694865577 | 0.093985396 | 0.540162009 |
| A1-i | Val-rep (recorded) | Overall | 2 | 0.628706930 | 0.694409762 | 0.095073766 | 0.541691147 |
| A1-i | Val-rep (recorded) | Overall | 4 | 0.631769963 | 0.693535021 | 0.095773322 | 0.543100152 |
| A1-i | Val-rep (recorded) | Urban | y0_draft | 0.665863264 | 0.721756819 | 0.086455522 | 0.566749782 |
| A1-i | Val-rep (recorded) | Urban | 1 | 0.671139560 | 0.723842648 | 0.088422922 | 0.570407159 |
| A1-i | Val-rep (recorded) | Urban | 2 | 0.675746472 | 0.726182134 | 0.090257909 | 0.573779458 |
| A1-i | Val-rep (recorded) | Urban | 4 | 0.681530051 | 0.730885803 | 0.091868105 | 0.578404387 |
| A1-i | Val-rep (recorded) | Rural | y0_draft | 0.530191817 | 0.662104901 | 0.100134166 | 0.483754212 |
| A1-i | Val-rep (recorded) | Rural | 1 | 0.530552297 | 0.658842535 | 0.100635607 | 0.483056030 |
| A1-i | Val-rep (recorded) | Rural | 2 | 0.530412412 | 0.655128094 | 0.100815989 | 0.481907832 |
| A1-i | Val-rep (recorded) | Rural | 4 | 0.528799166 | 0.647817875 | 0.100422696 | 0.478829485 |

| Head | Overall Train y0 A minus Val-rep y0 A |
|---|---:|
| linear | +0.040701433 |
| A1-i | +0.008107776 |
| A1-i minus linear gap | -0.032593657 |

| Head | Set | Domain | K | A minus own y0 A |
|---|---|---|---|---:|
| linear | Train subset | Overall | 1 | -0.017870712 |
| linear | Train subset | Overall | 2 | -0.018008063 |
| linear | Train subset | Overall | 4 | -0.013948923 |
| linear | Train subset | Urban | 1 | -0.016751326 |
| linear | Train subset | Urban | 2 | -0.016442618 |
| linear | Train subset | Urban | 4 | -0.011697764 |
| linear | Train subset | Rural | 1 | -0.020946956 |
| linear | Train subset | Rural | 2 | -0.022571202 |
| linear | Train subset | Rural | 4 | -0.020413170 |
| linear | Val-rep | Overall | 1 | +0.005878207 |
| linear | Val-rep | Overall | 2 | +0.005727435 |
| linear | Val-rep | Overall | 4 | +0.006786549 |
| linear | Val-rep | Urban | 1 | -0.006766863 |
| linear | Val-rep | Urban | 2 | -0.007618002 |
| linear | Val-rep | Urban | 4 | -0.005611652 |
| linear | Val-rep | Rural | 1 | +0.025937444 |
| linear | Val-rep | Rural | 2 | +0.026844585 |
| linear | Val-rep | Rural | 4 | +0.026563800 |
| A1-i | Train subset | Overall | 1 | +0.008391798 |
| A1-i | Train subset | Overall | 2 | +0.017325369 |
| A1-i | Train subset | Overall | 4 | +0.030868939 |
| A1-i | Train subset | Urban | 1 | +0.009232452 |
| A1-i | Train subset | Urban | 2 | +0.019007046 |
| A1-i | Train subset | Urban | 4 | +0.033271732 |
| A1-i | Train subset | Rural | 1 | +0.005357652 |
| A1-i | Train subset | Rural | 2 | +0.011452357 |
| A1-i | Train subset | Rural | 4 | +0.022251721 |
| A1-i | Val-rep | Overall | 1 | +0.001915888 |
| A1-i | Val-rep | Overall | 2 | +0.003445026 |
| A1-i | Val-rep | Overall | 4 | +0.004854031 |
| A1-i | Val-rep | Urban | 1 | +0.003657376 |
| A1-i | Val-rep | Urban | 2 | +0.007029675 |
| A1-i | Val-rep | Urban | 4 | +0.011654605 |
| A1-i | Val-rep | Rural | 1 | -0.000698181 |
| A1-i | Val-rep | Rural | 2 | -0.001846380 |
| A1-i | Val-rep | Rural | 4 | -0.004924727 |

## Registered predictions

Observed P1: False.
Observed P2: True.
Observed decision rule: abandon overfitting explanation; random K and loss weights.

CPU or limited-sample runs verify execution only; they do not adjudicate full-data predictions.

## Recorded Val provenance

- linear: results/exp/e0_v2_s0/metrics.json; JSON path evaluation.groups.<domain>.<K>.scores.
- A1-i: results/exp/e6_convdraft_s0/metrics.json; JSON path evaluation.groups.<domain>.<K>.scores.

## Full Val-rep draft regression

- linear: PASS; A=0.526050032825; delta=+3.28253586535e-08.
- A1-i: PASS; A=0.538246120907; delta=+1.20907098222e-07.
