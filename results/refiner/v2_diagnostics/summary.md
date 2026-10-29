# Refiner v1 diagnostics

Observed results only; no training or model selection was performed.
Train uses the first sorted filenames in each domain; Val uses the complete split unless limited.
Train is not an independent test set. Differences alone do not establish overfitting or loss/metric mismatch.

## Train subset vs Val, threshold 0.5

| Group | K/reference | Train IoU | Train clDice | Train BIoU | Train A | Val IoU | Val clDice | Val BIoU | Val A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | y0_draft | 0.646158 | 0.705086 | 0.063247 | 0.547254 | 0.664708 | 0.705534 | 0.084496 | 0.560913 |
| Urban | 1 | 0.641301 | 0.693815 | 0.061898 | 0.541174 | 0.659857 | 0.702787 | 0.086002 | 0.557965 |
| Urban | 2 | 0.646956 | 0.698075 | 0.063365 | 0.545574 | 0.661385 | 0.704615 | 0.086824 | 0.559442 |
| Urban | 4 | 0.658708 | 0.706919 | 0.067209 | 0.554871 | 0.664809 | 0.708944 | 0.087759 | 0.562639 |
| Urban | 8 | 0.672398 | 0.716676 | 0.074475 | 0.566097 | 0.663912 | 0.711526 | 0.087107 | 0.562835 |
| Rural | y0_draft | 0.572701 | 0.701749 | 0.117376 | 0.520351 | 0.522136 | 0.619433 | 0.095428 | 0.465984 |
| Rural | 1 | 0.562991 | 0.692671 | 0.114688 | 0.512235 | 0.537647 | 0.649296 | 0.105528 | 0.484718 |
| Rural | 2 | 0.565613 | 0.695070 | 0.116824 | 0.514692 | 0.538379 | 0.648517 | 0.106333 | 0.485011 |
| Rural | 4 | 0.572072 | 0.704269 | 0.122154 | 0.521747 | 0.535895 | 0.641052 | 0.106509 | 0.481565 |
| Rural | 8 | 0.580754 | 0.708460 | 0.131150 | 0.529145 | 0.526895 | 0.625548 | 0.105584 | 0.472229 |
| Overall | y0_draft | 0.632907 | 0.703360 | 0.080312 | 0.543524 | 0.617080 | 0.666980 | 0.089365 | 0.526507 |
| Overall | 1 | 0.626778 | 0.693164 | 0.078824 | 0.537103 | 0.619020 | 0.678124 | 0.094711 | 0.531889 |
| Overall | 2 | 0.631797 | 0.696959 | 0.080602 | 0.541107 | 0.620296 | 0.678801 | 0.095512 | 0.532891 |
| Overall | 4 | 0.642510 | 0.705928 | 0.085050 | 0.550043 | 0.621629 | 0.677950 | 0.096120 | 0.533423 |
| Overall | 8 | 0.655204 | 0.714073 | 0.093095 | 0.560443 | 0.617818 | 0.672579 | 0.095344 | 0.529752 |

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
| Urban | 1 | 0.100000 | 0.426126 |
| Urban | 1 | 0.200000 | 0.485153 |
| Urban | 1 | 0.300000 | 0.520436 |
| Urban | 1 | 0.400000 | 0.543795 |
| Urban | 1 | 0.500000 | 0.557965 |
| Urban | 1 | 0.600000 | 0.563157 |
| Urban | 1 | 0.700000 | 0.557794 |
| Urban | 1 | 0.800000 | 0.536910 |
| Urban | 1 | 0.900000 | 0.482852 |
| Urban | 4 | 0.100000 | 0.437441 |
| Urban | 4 | 0.200000 | 0.496593 |
| Urban | 4 | 0.300000 | 0.530820 |
| Urban | 4 | 0.400000 | 0.552078 |
| Urban | 4 | 0.500000 | 0.562639 |
| Urban | 4 | 0.600000 | 0.562368 |
| Urban | 4 | 0.700000 | 0.550962 |
| Urban | 4 | 0.800000 | 0.522812 |
| Urban | 4 | 0.900000 | 0.459897 |
| Rural | y0_draft | 0.100000 | 0.421719 |
| Rural | y0_draft | 0.200000 | 0.472499 |
| Rural | y0_draft | 0.300000 | 0.483368 |
| Rural | y0_draft | 0.400000 | 0.479122 |
| Rural | y0_draft | 0.500000 | 0.465984 |
| Rural | y0_draft | 0.600000 | 0.445458 |
| Rural | y0_draft | 0.700000 | 0.417358 |
| Rural | y0_draft | 0.800000 | 0.375884 |
| Rural | y0_draft | 0.900000 | 0.307099 |
| Rural | 1 | 0.100000 | 0.410461 |
| Rural | 1 | 0.200000 | 0.464560 |
| Rural | 1 | 0.300000 | 0.487804 |
| Rural | 1 | 0.400000 | 0.493451 |
| Rural | 1 | 0.500000 | 0.484718 |
| Rural | 1 | 0.600000 | 0.464014 |
| Rural | 1 | 0.700000 | 0.431506 |
| Rural | 1 | 0.800000 | 0.383400 |
| Rural | 1 | 0.900000 | 0.307357 |
| Rural | 4 | 0.100000 | 0.420353 |
| Rural | 4 | 0.200000 | 0.472819 |
| Rural | 4 | 0.300000 | 0.493152 |
| Rural | 4 | 0.400000 | 0.494863 |
| Rural | 4 | 0.500000 | 0.481565 |
| Rural | 4 | 0.600000 | 0.455841 |
| Rural | 4 | 0.700000 | 0.417567 |
| Rural | 4 | 0.800000 | 0.364082 |
| Rural | 4 | 0.900000 | 0.282342 |
| Overall | y0_draft | 0.100000 | 0.435599 |
| Overall | y0_draft | 0.200000 | 0.494422 |
| Overall | y0_draft | 0.300000 | 0.517643 |
| Overall | y0_draft | 0.400000 | 0.526517 |
| Overall | y0_draft | 0.500000 | 0.526507 |
| Overall | y0_draft | 0.600000 | 0.519213 |
| Overall | y0_draft | 0.700000 | 0.503865 |
| Overall | y0_draft | 0.800000 | 0.474986 |
| Overall | y0_draft | 0.900000 | 0.416385 |
| Overall | 1 | 0.100000 | 0.421880 |
| Overall | 1 | 0.200000 | 0.479280 |
| Overall | 1 | 0.300000 | 0.509976 |
| Overall | 1 | 0.400000 | 0.526550 |
| Overall | 1 | 0.500000 | 0.531889 |
| Overall | 1 | 0.600000 | 0.527065 |
| Overall | 1 | 0.700000 | 0.511114 |
| Overall | 1 | 0.800000 | 0.479456 |
| Overall | 1 | 0.900000 | 0.416293 |
| Overall | 4 | 0.100000 | 0.432668 |
| Overall | 4 | 0.200000 | 0.489460 |
| Overall | 4 | 0.300000 | 0.518377 |
| Overall | 4 | 0.400000 | 0.532099 |
| Overall | 4 | 0.500000 | 0.533423 |
| Overall | 4 | 0.600000 | 0.523281 |
| Overall | 4 | 0.700000 | 0.501358 |
| Overall | 4 | 0.800000 | 0.463139 |
| Overall | 4 | 0.900000 | 0.392360 |

## Best thresholds (selected on Val; optimistic)

| Group | K/reference | Threshold | A |
| --- | --- | --- | --- |
| Urban | y0_draft | 0.600000 | 0.562058 |
| Urban | 1 | 0.600000 | 0.563157 |
| Urban | 4 | 0.500000 | 0.562639 |
| Rural | y0_draft | 0.300000 | 0.483368 |
| Rural | 1 | 0.400000 | 0.493451 |
| Rural | 4 | 0.400000 | 0.494863 |
| Overall | y0_draft | 0.400000 | 0.526517 |
| Overall | 1 | 0.500000 | 0.531889 |
| Overall | 4 | 0.500000 | 0.533423 |

## Train vs Val best thresholds (split-selected; optimistic)

| Group | K/reference | Train threshold | Train A | Val threshold | Val A |
| --- | --- | --- | --- | --- | --- |
| Urban | y0_draft | 0.700000 | 0.570085 | 0.600000 | 0.562058 |
| Urban | 1 | 0.700000 | 0.571533 | 0.600000 | 0.563157 |
| Rural | y0_draft | 0.700000 | 0.534798 | 0.300000 | 0.483368 |
| Rural | 1 | 0.700000 | 0.542996 | 0.400000 | 0.493451 |
| Overall | y0_draft | 0.700000 | 0.564617 | 0.400000 | 0.526517 |
| Overall | 1 | 0.700000 | 0.567819 | 0.500000 | 0.531889 |

## Val flip decomposition

| Group | K | added_correct | added_wrong | deleted_correct | deleted_wrong | added_correct/valid | added_wrong/valid | deleted_correct/valid | deleted_wrong/valid | Correct/changed | Correct/added | Correct/deleted | Added/valid | Deleted/valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 1019443 | 2054348 | 113674 | 50458 | 0.001494 | 0.003011 | 0.000167 | 0.000074 | 0.349952 | 0.331657 | 0.692577 | 0.004505 | 0.000241 |
| Urban | 2 | 993030 | 1922842 | 388573 | 191735 | 0.001455 | 0.002818 | 0.000569 | 0.000281 | 0.395175 | 0.340560 | 0.669598 | 0.004273 | 0.000850 |
| Urban | 4 | 748663 | 1303394 | 1170234 | 653620 | 0.001097 | 0.001910 | 0.001715 | 0.000958 | 0.495083 | 0.364835 | 0.641627 | 0.003007 | 0.002673 |
| Urban | 8 | 500995 | 778835 | 2833055 | 1915909 | 0.000734 | 0.001141 | 0.004152 | 0.002808 | 0.553021 | 0.391454 | 0.596563 | 0.001876 | 0.006960 |
| Rural | 1 | 1051868 | 1043864 | 56429 | 21154 | 0.001029 | 0.001021 | 0.000055 | 0.000021 | 0.509957 | 0.501910 | 0.727337 | 0.002051 | 0.000076 |
| Rural | 2 | 1019299 | 953108 | 186627 | 83214 | 0.000997 | 0.000933 | 0.000183 | 0.000081 | 0.537820 | 0.516779 | 0.691618 | 0.001930 | 0.000264 |
| Rural | 4 | 831063 | 716031 | 519369 | 282305 | 0.000813 | 0.000701 | 0.000508 | 0.000276 | 0.574953 | 0.537177 | 0.647856 | 0.001514 | 0.000784 |
| Rural | 8 | 610848 | 488840 | 1188500 | 826148 | 0.000598 | 0.000478 | 0.001163 | 0.000808 | 0.577763 | 0.555474 | 0.589929 | 0.001076 | 0.001971 |
| Overall | 1 | 2071311 | 3098212 | 170103 | 71612 | 0.001215 | 0.001818 | 0.000100 | 0.000042 | 0.414215 | 0.400677 | 0.703734 | 0.003033 | 0.000142 |
| Overall | 2 | 2012329 | 2875950 | 575200 | 274949 | 0.001181 | 0.001687 | 0.000338 | 0.000161 | 0.450913 | 0.411664 | 0.676587 | 0.002868 | 0.000499 |
| Overall | 4 | 1579726 | 2019425 | 1689603 | 935925 | 0.000927 | 0.001185 | 0.000991 | 0.000549 | 0.525220 | 0.438916 | 0.643529 | 0.002112 | 0.001541 |
| Overall | 8 | 1111843 | 1267675 | 4021555 | 2742057 | 0.000652 | 0.000744 | 0.002360 | 0.001609 | 0.561449 | 0.467256 | 0.594587 | 0.001396 | 0.003969 |

## Tracking fixed y0 skeleton pixels

| Group | K | Gap covered | Gap total | Gap repair fraction | Terminal covered | Terminal total | Terminal coverage fraction | Covered lost | Covered total | New break fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 29763 | 162024 | 0.183695 | 8347 | 125115 | 0.066715 | 621 | 739678 | 0.000840 |
| Urban | 2 | 29710 | 162024 | 0.183368 | 8837 | 125115 | 0.070631 | 2603 | 739678 | 0.003519 |
| Urban | 4 | 24141 | 162024 | 0.148996 | 7250 | 125115 | 0.057947 | 8966 | 739678 | 0.012121 |
| Urban | 8 | 18000 | 162024 | 0.111095 | 5312 | 125115 | 0.042457 | 27459 | 739678 | 0.037123 |
| Rural | 1 | 54071 | 269473 | 0.200655 | 9680 | 142020 | 0.068159 | 483 | 507264 | 0.000952 |
| Rural | 2 | 52260 | 269473 | 0.193934 | 9520 | 142020 | 0.067033 | 2185 | 507264 | 0.004307 |
| Rural | 4 | 43431 | 269473 | 0.161170 | 7892 | 142020 | 0.055570 | 8433 | 507264 | 0.016624 |
| Rural | 8 | 32999 | 269473 | 0.122458 | 5876 | 142020 | 0.041374 | 26098 | 507264 | 0.051449 |
| Overall | 1 | 83834 | 431497 | 0.194286 | 18027 | 267135 | 0.067483 | 1104 | 1246942 | 0.000885 |
| Overall | 2 | 81970 | 431497 | 0.189967 | 18357 | 267135 | 0.068718 | 4788 | 1246942 | 0.003840 |
| Overall | 4 | 67572 | 431497 | 0.156599 | 15142 | 267135 | 0.056683 | 17399 | 1246942 | 0.013953 |
| Overall | 8 | 50999 | 431497 | 0.118191 | 11188 | 267135 | 0.041881 | 53557 | 1246942 | 0.042951 |

## New fragment shapes

| Group | K | Count | Area Q1 | Area median | Area Q3 | Width Q1 | Width median | Width Q3 | Height Q1 | Height median | Height Q3 | Single cell fraction | Aligned edges/all edges | Any aligned edge/fragment | All aligned edges/fragment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 1 | 2706 | 8.000000 | 23.000000 | 60.000000 | 3.000000 | 6.000000 | 13.000000 | 2.000000 | 5.000000 | 9.000000 | 0.642646 | 0.045547 | 0.163710 | 0.000000 |
| Urban | 4 | 2156 | 8.000000 | 23.000000 | 61.000000 | 3.000000 | 7.000000 | 14.000000 | 2.000000 | 5.000000 | 8.000000 | 0.617347 | 0.048237 | 0.170686 | 0.000000 |
| Rural | 1 | 2232 | 7.000000 | 19.000000 | 40.000000 | 3.000000 | 5.000000 | 10.000000 | 2.000000 | 4.000000 | 8.000000 | 0.691756 | 0.040547 | 0.149194 | 0.000000 |
| Rural | 4 | 1868 | 7.000000 | 18.000000 | 40.000000 | 3.000000 | 5.000000 | 10.000000 | 2.000000 | 4.000000 | 8.000000 | 0.694861 | 0.042559 | 0.153640 | 0.000000 |
| Overall | 1 | 4938 | 7.000000 | 21.000000 | 49.750000 | 3.000000 | 5.000000 | 11.000000 | 2.000000 | 5.000000 | 8.000000 | 0.664844 | 0.043287 | 0.157149 | 0.000000 |
| Overall | 4 | 4024 | 7.000000 | 21.000000 | 49.000000 | 3.000000 | 6.000000 | 12.000000 | 2.000000 | 4.000000 | 8.000000 | 0.653330 | 0.045601 | 0.162773 | 0.000000 |

## Urban: worst delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 3617.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3655.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 3922.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 4176.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 4068.png | True | 0.714392 | 0.610122 | -0.104269 | added_wrong |
| 4187.png | True | 0.552431 | 0.455590 | -0.096840 | added_wrong |
| 3981.png | True | 0.714124 | 0.638641 | -0.075483 | added_wrong |
| 3782.png | True | 0.267108 | 0.193890 | -0.073218 | added_wrong |
| 3833.png | True | 0.752153 | 0.681466 | -0.070687 | added_wrong |
| 3764.png | True | 0.666173 | 0.596556 | -0.069617 | added_wrong |

## Urban: best delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 3866.png | False | 0.000000 | 1.000000 | 1.000000 | deleted_correct |
| 3744.png | True | 0.160607 | 0.317789 | 0.157181 | added_correct |
| 3520.png | True | 0.280362 | 0.429269 | 0.148907 | added_correct |
| 4185.png | True | 0.252781 | 0.396304 | 0.143523 | added_correct |
| 3878.png | True | 0.087286 | 0.218865 | 0.131579 | added_correct |
| 3614.png | True | 0.113135 | 0.240228 | 0.127093 | added_correct |
| 3708.png | True | 0.235458 | 0.360417 | 0.124959 | added_correct |
| 3544.png | True | 0.311767 | 0.427684 | 0.115917 | added_correct |
| 3527.png | True | 0.314798 | 0.430579 | 0.115782 | added_correct |
| 4075.png | True | 0.397482 | 0.512445 | 0.114964 | added_correct |

## Rural: worst delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 2582.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2619.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2638.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2668.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2676.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2747.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2779.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2818.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2832.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |
| 2836.png | False | 1.000000 | 0.000000 | -1.000000 | added_wrong |

## Rural: best delta A(K1-y0)

| Filename | GT road | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- | --- |
| 3488.png | True | 0.148470 | 0.416927 | 0.268457 | added_wrong |
| 2880.png | True | 0.326019 | 0.575174 | 0.249155 | added_correct |
| 2724.png | True | 0.132918 | 0.370020 | 0.237103 | added_correct |
| 2958.png | True | 0.248579 | 0.459908 | 0.211330 | added_correct |
| 3181.png | True | 0.230946 | 0.439269 | 0.208323 | added_correct |
| 3462.png | True | 0.216883 | 0.420074 | 0.203191 | added_correct |
| 3337.png | True | 0.119605 | 0.322695 | 0.203090 | added_correct |
| 3425.png | True | 0.028930 | 0.226799 | 0.197869 | added_correct |
| 2897.png | True | 0.146020 | 0.338366 | 0.192346 | added_correct |
| 3442.png | True | 0.245375 | 0.435720 | 0.190344 | added_correct |

## Urban: worst delta A(K1-y0), GT-road images only

| Filename | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- |
| 4068.png | 0.714392 | 0.610122 | -0.104269 | added_wrong |
| 4187.png | 0.552431 | 0.455590 | -0.096840 | added_wrong |
| 3981.png | 0.714124 | 0.638641 | -0.075483 | added_wrong |
| 3782.png | 0.267108 | 0.193890 | -0.073218 | added_wrong |
| 3833.png | 0.752153 | 0.681466 | -0.070687 | added_wrong |
| 3764.png | 0.666173 | 0.596556 | -0.069617 | added_wrong |
| 3582.png | 0.534337 | 0.467105 | -0.067232 | added_wrong |
| 4056.png | 0.392923 | 0.330140 | -0.062782 | added_wrong |
| 3681.png | 0.641817 | 0.583116 | -0.058701 | added_wrong |
| 3679.png | 0.730392 | 0.672121 | -0.058271 | added_wrong |

## Urban: best delta A(K1-y0), GT-road images only

| Filename | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- |
| 3744.png | 0.160607 | 0.317789 | 0.157181 | added_correct |
| 3520.png | 0.280362 | 0.429269 | 0.148907 | added_correct |
| 4185.png | 0.252781 | 0.396304 | 0.143523 | added_correct |
| 3878.png | 0.087286 | 0.218865 | 0.131579 | added_correct |
| 3614.png | 0.113135 | 0.240228 | 0.127093 | added_correct |
| 3708.png | 0.235458 | 0.360417 | 0.124959 | added_correct |
| 3544.png | 0.311767 | 0.427684 | 0.115917 | added_correct |
| 3527.png | 0.314798 | 0.430579 | 0.115782 | added_correct |
| 4075.png | 0.397482 | 0.512445 | 0.114964 | added_correct |
| 3951.png | 0.070680 | 0.179860 | 0.109180 | added_correct |

## Rural: worst delta A(K1-y0), GT-road images only

| Filename | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- |
| 3153.png | 0.375029 | 0.250146 | -0.124883 | added_wrong |
| 3117.png | 0.222941 | 0.101371 | -0.121570 | added_wrong |
| 2959.png | 0.422693 | 0.305991 | -0.116703 | added_wrong |
| 2672.png | 0.195574 | 0.080808 | -0.114765 | added_wrong |
| 3177.png | 0.502642 | 0.408746 | -0.093896 | added_wrong |
| 3512.png | 0.396671 | 0.311869 | -0.084802 | added_wrong |
| 3179.png | 0.358286 | 0.280898 | -0.077388 | added_wrong |
| 2801.png | 0.213049 | 0.152240 | -0.060810 | added_wrong |
| 3006.png | 0.513633 | 0.459893 | -0.053741 | added_wrong |
| 2965.png | 0.589492 | 0.538970 | -0.050522 | added_wrong |

## Rural: best delta A(K1-y0), GT-road images only

| Filename | y0 A | K1 A | Delta A | Largest flip category |
| --- | --- | --- | --- | --- |
| 3488.png | 0.148470 | 0.416927 | 0.268457 | added_wrong |
| 2880.png | 0.326019 | 0.575174 | 0.249155 | added_correct |
| 2724.png | 0.132918 | 0.370020 | 0.237103 | added_correct |
| 2958.png | 0.248579 | 0.459908 | 0.211330 | added_correct |
| 3181.png | 0.230946 | 0.439269 | 0.208323 | added_correct |
| 3462.png | 0.216883 | 0.420074 | 0.203191 | added_correct |
| 3337.png | 0.119605 | 0.322695 | 0.203090 | added_correct |
| 3425.png | 0.028930 | 0.226799 | 0.197869 | added_correct |
| 2897.png | 0.146020 | 0.338366 | 0.192346 | added_correct |
| 3442.png | 0.245375 | 0.435720 | 0.190344 | added_correct |

Coverage is exact skeleton-pixel intersection, with no dilation or distance tolerance.
Skeletons and gap categories use the original shared Zhang/8-connectivity classifier; baseline sets stay fixed across K.
New fragments require zero exact overlap with y0 and GT (no two-pixel tolerance). Bounding maxima are inclusive.
A boundary coordinate is aligned when mod 16 is 0 or 15; both per-edge and per-fragment rates are reported.
Undefined descriptive fractions are null/NA. Empty-GT image scores follow official summarize without overrides.
Triptychs leave unchanged FP uncolored; each selected image is inferred only once, with packed masks held in bounded RAM.
Flip counts and official 0.5 Counts passed an independent comparison with train_refiner.score_image on every Val image.
