# Refiner v1 diagnostics

Observed results only; no training or model selection was performed.
Train uses the first sorted filenames in each domain; Val uses the complete split unless limited.
Train is not an independent test set. Differences alone do not establish overfitting or loss/metric mismatch.

## Train subset vs Val, threshold 0.5

| Group | K/reference | Train IoU | Train clDice | Train BIoU | Train A | Val IoU | Val clDice | Val BIoU | Val A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | y0_draft | 0.646158 | 0.705086 | 0.063247 | 0.547254 | 0.664708 | 0.705534 | 0.084496 | 0.560913 |
| Urban | 1 | 0.691218 | 0.734064 | 0.077085 | 0.581245 | 0.623301 | 0.648576 | 0.083383 | 0.522900 |
| Urban | 2 | 0.692504 | 0.727358 | 0.079367 | 0.580333 | 0.618806 | 0.644496 | 0.083380 | 0.519428 |
| Urban | 4 | 0.693393 | 0.728933 | 0.080210 | 0.581418 | 0.617951 | 0.644538 | 0.084419 | 0.519221 |
| Urban | 8 | 0.686673 | 0.724520 | 0.077745 | 0.576242 | 0.615484 | 0.645257 | 0.083795 | 0.518078 |
| Rural | y0_draft | 0.572701 | 0.701749 | 0.117376 | 0.520351 | 0.522136 | 0.619433 | 0.095428 | 0.465984 |
| Rural | 1 | 0.625069 | 0.759256 | 0.133991 | 0.567109 | 0.498225 | 0.593281 | 0.084502 | 0.443997 |
| Rural | 2 | 0.628143 | 0.744951 | 0.138896 | 0.565336 | 0.495949 | 0.584970 | 0.086362 | 0.440738 |
| Rural | 4 | 0.631940 | 0.738249 | 0.146914 | 0.566828 | 0.496696 | 0.581826 | 0.089909 | 0.440878 |
| Rural | 8 | 0.634476 | 0.736388 | 0.154931 | 0.569141 | 0.497894 | 0.576782 | 0.093532 | 0.440688 |
| Overall | y0_draft | 0.632907 | 0.703360 | 0.080312 | 0.543524 | 0.617080 | 0.666980 | 0.089365 | 0.526507 |
| Overall | 1 | 0.679186 | 0.739240 | 0.094725 | 0.580310 | 0.583831 | 0.621299 | 0.083840 | 0.495073 |
| Overall | 2 | 0.680818 | 0.730971 | 0.097845 | 0.579269 | 0.580372 | 0.615905 | 0.084584 | 0.491874 |
| Overall | 4 | 0.682288 | 0.730705 | 0.100961 | 0.580548 | 0.580016 | 0.614814 | 0.086631 | 0.491779 |
| Overall | 8 | 0.677427 | 0.726637 | 0.101356 | 0.576976 | 0.578710 | 0.612888 | 0.087698 | 0.490761 |

## Val threshold scan (selected on Val; optimistic)

| Group | K/reference | Threshold | A |
| --- | --- | --- | --- |
| Urban | y0_draft | 0.100000 | 0.441175 |
| Urban | y0_draft | 0.200000 | 0.504608 |
| Urban | y0_draft | 0.300000 | 0.535529 |
| Urban | y0_draft | 0.400000 | 0.552641 |
| Urban | y0_draft | 0.500000 | 0.560913 |
| Urban | y0_draft | 0.600000 | 0.562058 |
| Urban | y0_draft | 0.700000 | 0.554922 |
| Urban | y0_draft | 0.800000 | 0.534419 |
| Urban | y0_draft | 0.900000 | 0.483218 |
| Urban | 1 | 0.100000 | 0.421555 |
| Urban | 1 | 0.200000 | 0.466405 |
| Urban | 1 | 0.300000 | 0.491986 |
| Urban | 1 | 0.400000 | 0.510336 |
| Urban | 1 | 0.500000 | 0.522900 |
| Urban | 1 | 0.600000 | 0.531239 |
| Urban | 1 | 0.700000 | 0.533839 |
| Urban | 1 | 0.800000 | 0.525998 |
| Urban | 1 | 0.900000 | 0.491355 |
| Urban | 4 | 0.100000 | 0.422679 |
| Urban | 4 | 0.200000 | 0.464146 |
| Urban | 4 | 0.300000 | 0.489073 |
| Urban | 4 | 0.400000 | 0.506328 |
| Urban | 4 | 0.500000 | 0.519221 |
| Urban | 4 | 0.600000 | 0.527261 |
| Urban | 4 | 0.700000 | 0.529545 |
| Urban | 4 | 0.800000 | 0.522508 |
| Urban | 4 | 0.900000 | 0.489097 |
| Rural | y0_draft | 0.100000 | 0.421719 |
| Rural | y0_draft | 0.200000 | 0.472499 |
| Rural | y0_draft | 0.300000 | 0.483368 |
| Rural | y0_draft | 0.400000 | 0.479122 |
| Rural | y0_draft | 0.500000 | 0.465984 |
| Rural | y0_draft | 0.600000 | 0.445458 |
| Rural | y0_draft | 0.700000 | 0.417358 |
| Rural | y0_draft | 0.800000 | 0.375884 |
| Rural | y0_draft | 0.900000 | 0.307099 |
| Rural | 1 | 0.100000 | 0.408929 |
| Rural | 1 | 0.200000 | 0.431251 |
| Rural | 1 | 0.300000 | 0.440979 |
| Rural | 1 | 0.400000 | 0.444547 |
| Rural | 1 | 0.500000 | 0.443997 |
| Rural | 1 | 0.600000 | 0.439335 |
| Rural | 1 | 0.700000 | 0.427995 |
| Rural | 1 | 0.800000 | 0.402653 |
| Rural | 1 | 0.900000 | 0.343014 |
| Rural | 4 | 0.100000 | 0.393122 |
| Rural | 4 | 0.200000 | 0.421467 |
| Rural | 4 | 0.300000 | 0.434281 |
| Rural | 4 | 0.400000 | 0.439953 |
| Rural | 4 | 0.500000 | 0.440878 |
| Rural | 4 | 0.600000 | 0.436913 |
| Rural | 4 | 0.700000 | 0.426515 |
| Rural | 4 | 0.800000 | 0.403657 |
| Rural | 4 | 0.900000 | 0.349589 |
| Overall | y0_draft | 0.100000 | 0.435599 |
| Overall | y0_draft | 0.200000 | 0.494422 |
| Overall | y0_draft | 0.300000 | 0.517643 |
| Overall | y0_draft | 0.400000 | 0.526517 |
| Overall | y0_draft | 0.500000 | 0.526507 |
| Overall | y0_draft | 0.600000 | 0.519213 |
| Overall | y0_draft | 0.700000 | 0.503865 |
| Overall | y0_draft | 0.800000 | 0.474986 |
| Overall | y0_draft | 0.900000 | 0.416385 |
| Overall | 1 | 0.100000 | 0.417439 |
| Overall | 1 | 0.200000 | 0.454293 |
| Overall | 1 | 0.300000 | 0.474265 |
| Overall | 1 | 0.400000 | 0.487284 |
| Overall | 1 | 0.500000 | 0.495073 |
| Overall | 1 | 0.600000 | 0.498587 |
| Overall | 1 | 0.700000 | 0.495899 |
| Overall | 1 | 0.800000 | 0.481315 |
| Overall | 1 | 0.900000 | 0.436574 |
| Overall | 4 | 0.100000 | 0.412925 |
| Overall | 4 | 0.200000 | 0.449683 |
| Overall | 4 | 0.300000 | 0.470257 |
| Overall | 4 | 0.400000 | 0.483320 |
| Overall | 4 | 0.500000 | 0.491779 |
| Overall | 4 | 0.600000 | 0.495303 |
| Overall | 4 | 0.700000 | 0.492743 |
| Overall | 4 | 0.800000 | 0.479488 |
| Overall | 4 | 0.900000 | 0.437597 |

## Best thresholds (selected on Val; optimistic)

| Group | K/reference | Threshold | A |
| --- | --- | --- | --- |
| Urban | y0_draft | 0.600000 | 0.562058 |
| Urban | 1 | 0.700000 | 0.533839 |
| Urban | 4 | 0.700000 | 0.529545 |
| Rural | y0_draft | 0.300000 | 0.483368 |
| Rural | 1 | 0.400000 | 0.444547 |
| Rural | 4 | 0.500000 | 0.440878 |
| Overall | y0_draft | 0.400000 | 0.526517 |
| Overall | 1 | 0.600000 | 0.498587 |
| Overall | 4 | 0.600000 | 0.495303 |

## Val flip decomposition

| Group | K | added_correct | added_wrong | deleted_correct | deleted_wrong | added_correct/valid | added_wrong/valid | deleted_correct/valid | deleted_wrong/valid | Correct/changed | Correct/added | Correct/deleted | Added/valid | Deleted/valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 2103829 | 7707359 | 2232387 | 1351180 | 0.003083 | 0.011295 | 0.003272 | 0.001980 | 0.323725 | 0.214432 | 0.622951 | 0.014378 | 0.005252 |
| Urban | 2 | 2208914 | 8311690 | 2345765 | 1465882 | 0.003237 | 0.012181 | 0.003438 | 0.002148 | 0.317792 | 0.209961 | 0.615420 | 0.015418 | 0.005586 |
| Urban | 4 | 2160735 | 8247159 | 2579631 | 1662109 | 0.003167 | 0.012086 | 0.003780 | 0.002436 | 0.323583 | 0.207605 | 0.608154 | 0.015253 | 0.006216 |
| Urban | 8 | 2076652 | 8172684 | 2912526 | 2001183 | 0.003043 | 0.011977 | 0.004268 | 0.002933 | 0.329035 | 0.202613 | 0.592735 | 0.015021 | 0.007201 |
| Rural | 1 | 1179036 | 1778551 | 1859912 | 1990117 | 0.001154 | 0.001740 | 0.001820 | 0.001947 | 0.446404 | 0.398648 | 0.483090 | 0.002894 | 0.003767 |
| Rural | 2 | 1214992 | 1814715 | 2078753 | 2189825 | 0.001189 | 0.001776 | 0.002034 | 0.002143 | 0.451304 | 0.401026 | 0.486990 | 0.002965 | 0.004177 |
| Rural | 4 | 1226535 | 1780657 | 2177651 | 2243526 | 0.001200 | 0.001742 | 0.002131 | 0.002195 | 0.458268 | 0.407867 | 0.492550 | 0.002943 | 0.004326 |
| Rural | 8 | 1201752 | 1683951 | 2283434 | 2281418 | 0.001176 | 0.001648 | 0.002234 | 0.002232 | 0.467775 | 0.416450 | 0.500221 | 0.002824 | 0.004467 |
| Overall | 1 | 3282865 | 9485910 | 4092299 | 3341297 | 0.001926 | 0.005566 | 0.002401 | 0.001961 | 0.365064 | 0.257101 | 0.550514 | 0.007492 | 0.004362 |
| Overall | 2 | 3423906 | 10126405 | 4424518 | 3655707 | 0.002009 | 0.005942 | 0.002596 | 0.002145 | 0.362840 | 0.252681 | 0.547574 | 0.007951 | 0.004741 |
| Overall | 4 | 3387270 | 10027816 | 4757282 | 3905635 | 0.001987 | 0.005884 | 0.002791 | 0.002292 | 0.368899 | 0.252497 | 0.549155 | 0.007871 | 0.005083 |
| Overall | 8 | 3278404 | 9856635 | 5195960 | 4282601 | 0.001924 | 0.005783 | 0.003049 | 0.002513 | 0.374746 | 0.249592 | 0.548180 | 0.007707 | 0.005562 |

## Tracking fixed y0 skeleton pixels

| Group | K | Gap covered | Gap total | Gap repair fraction | Terminal covered | Terminal total | Terminal coverage fraction | Covered lost | Covered total | New break fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 46639 | 162024 | 0.287852 | 15730 | 125115 | 0.125724 | 33126 | 739678 | 0.044784 |
| Urban | 2 | 50396 | 162024 | 0.311040 | 17161 | 125115 | 0.137162 | 33647 | 739678 | 0.045489 |
| Urban | 4 | 50883 | 162024 | 0.314046 | 17366 | 125115 | 0.138800 | 36381 | 739678 | 0.049185 |
| Urban | 8 | 49566 | 162024 | 0.305918 | 16947 | 125115 | 0.135451 | 42669 | 739678 | 0.057686 |
| Rural | 1 | 62696 | 269473 | 0.232662 | 17643 | 142020 | 0.124229 | 93582 | 507264 | 0.184484 |
| Rural | 2 | 63901 | 269473 | 0.237133 | 18203 | 142020 | 0.128172 | 100807 | 507264 | 0.198727 |
| Rural | 4 | 64081 | 269473 | 0.237801 | 18268 | 142020 | 0.128630 | 102962 | 507264 | 0.202975 |
| Rural | 8 | 61569 | 269473 | 0.228479 | 17285 | 142020 | 0.121708 | 105117 | 507264 | 0.207223 |
| Overall | 1 | 109335 | 431497 | 0.253385 | 33373 | 267135 | 0.124929 | 126708 | 1246942 | 0.101615 |
| Overall | 2 | 114297 | 431497 | 0.264885 | 35364 | 267135 | 0.132383 | 134454 | 1246942 | 0.107827 |
| Overall | 4 | 114964 | 431497 | 0.266431 | 35634 | 267135 | 0.133393 | 139343 | 1246942 | 0.111748 |
| Overall | 8 | 111135 | 431497 | 0.257557 | 34232 | 267135 | 0.128145 | 147786 | 1246942 | 0.118519 |

## New fragment shapes

| Group | K | Count | Area Q1 | Area median | Area Q3 | Width Q1 | Width median | Width Q3 | Height Q1 | Height median | Height Q3 | Single cell fraction | Aligned edges/all edges | Any aligned edge/fragment | All aligned edges/fragment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 8443 | 7.000000 | 29.000000 | 117.000000 | 3.000000 | 8.000000 | 15.000000 | 3.000000 | 6.000000 | 12.000000 | 0.752102 | 0.477111 | 0.909866 | 0.064669 |
| Urban | 4 | 9925 | 6.000000 | 23.000000 | 98.000000 | 3.000000 | 7.000000 | 14.000000 | 2.000000 | 5.000000 | 11.000000 | 0.780353 | 0.503501 | 0.930680 | 0.072645 |
| Rural | 1 | 2680 | 6.000000 | 20.000000 | 80.000000 | 2.000000 | 6.000000 | 12.000000 | 2.000000 | 5.000000 | 10.000000 | 0.798134 | 0.490019 | 0.914552 | 0.069403 |
| Rural | 4 | 3788 | 5.000000 | 16.000000 | 55.000000 | 2.000000 | 5.000000 | 10.000000 | 2.000000 | 5.000000 | 9.000000 | 0.830253 | 0.512078 | 0.930570 | 0.068374 |
| Overall | 1 | 11123 | 7.000000 | 27.000000 | 106.500000 | 3.000000 | 7.000000 | 14.000000 | 2.000000 | 6.000000 | 12.000000 | 0.763193 | 0.480221 | 0.910995 | 0.065810 |
| Overall | 4 | 13713 | 6.000000 | 21.000000 | 82.000000 | 3.000000 | 6.000000 | 13.000000 | 2.000000 | 5.000000 | 11.000000 | 0.794137 | 0.505870 | 0.930650 | 0.071465 |

## Urban: worst delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 3655.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3818.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3922.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 4147.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3662.png | True | 0.688547 | 0.255132 | -0.433415 | added_wrong |
| 3803.png | True | 0.554848 | 0.135990 | -0.418858 | deleted_wrong |
| 3823.png | True | 0.540332 | 0.137713 | -0.402619 | deleted_wrong |
| 3559.png | True | 0.468022 | 0.081554 | -0.386468 | deleted_wrong |
| 3788.png | True | 0.369540 | 0.009573 | -0.359967 | deleted_wrong |
| 3643.png | True | 0.705998 | 0.372859 | -0.333139 | added_wrong |

## Urban: best delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 4128.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3942.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3866.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3757.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3707.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3625.png | True | 0.000000 | 0.249035 | 0.249035 | added_correct |
| 3601.png | True | 0.268737 | 0.496183 | 0.227446 | added_correct |
| 4075.png | True | 0.397482 | 0.624686 | 0.227204 | added_correct |
| 3520.png | True | 0.280362 | 0.495532 | 0.215170 | added_correct |
| 4185.png | True | 0.252781 | 0.460251 | 0.207469 | added_correct |

## Rural: worst delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 2546.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2616.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2779.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2898.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2918.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2987.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3202.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3348.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3364.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3452.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |

## Rural: best delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 3498.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3487.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3486.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3481.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3480.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3464.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3460.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3431.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3413.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3410.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |

Coverage is exact skeleton-pixel intersection, with no dilation or distance tolerance.
Skeletons and gap categories use the original shared Zhang/8-connectivity classifier; baseline sets stay fixed across K.
New fragments require zero exact overlap with y0 and GT (no two-pixel tolerance). Bounding maxima are inclusive.
A boundary coordinate is aligned when mod 16 is 0 or 15; both per-edge and per-fragment rates are reported.
Undefined descriptive fractions are null/NA. Empty-GT image scores follow official summarize without overrides.
Triptychs leave unchanged FP uncolored; each selected image is inferred only once, with packed masks held in bounded RAM.
Flip counts and official 0.5 Counts passed an independent comparison with train_refiner.score_image on every Val image.
