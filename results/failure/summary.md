# Failure analysis

Observed only frozen inference is used. Valid pixels have raw label !=0; GT road is label 3. Official per-image scores are retained, but analysis A/delta-A is not applicable for images with no GT road. These images are counted separately, including their added pixels; aggregate official Counts still include them. An empty road annotation is not evidence that the image has no visible road; prior manual audits do not alter labels here.

Observed connection proxy: one predicted eight-connected component overlapping >=2 distinct y0 components is one merge, regardless of how many components it joins. Added pixels in that component determine its label: <50% GT-road is false, >=50% is correct. Ignore regions are excluded before connectivity. This is a geometric approximation, not proof of a road-network connection.

Observed disappearing-road pixels are GT road covered by y0 but deleted at K; new-break pixels are the existing fixed-y0 GT skeleton subset lost at K. Gap sets and new-break sets reuse refiner_diagnostics.baseline_skeleton_sets through eval_stress.analyze_image.

Observed case selection: only model seed 0; sorted candidate pools and RNG seed 0; up to three random examples per category plus each category's minimum/maximum applicable K4-y0 delta. Duplicates within a category are combined and listed in the visualization index. The no-GT category additionally includes maximum K4 added pixels; delta extrema are inapplicable there. Named positive/negative and connection/stress categories are predeclared; explicit extrema are not random examples.

Observed each cell below is for one trained seed; domain/corruption groups overlap and must not be summed indiscriminately.

| Run | Group | Images | No GT | Better fraction | Worse fraction | Unchanged fraction | False merges | False/image | Correct merges | Correct/image | Disappearing road pixels | K8-K4 < -0.01 fraction | No-GT added pixels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary/s0 | Overall | 835 | 207 | 0.5971337579617835 | 0.35987261146496813 | 0.042993630573248405 | 689 | 0.8251497005988024 | 538 | 0.644311377245509 | 200196 | 0.10509554140127389 | 251775 |
| ordinary/s0 | Urban | 339 | 53 | 0.44755244755244755 | 0.527972027972028 | 0.024475524475524476 | 379 | 1.1179941002949854 | 184 | 0.5427728613569321 | 147972 | 0.09090909090909091 | 160654 |
| ordinary/s0 | Rural | 496 | 154 | 0.7222222222222222 | 0.21929824561403508 | 0.05847953216374269 | 310 | 0.625 | 354 | 0.7137096774193549 | 52224 | 0.11695906432748537 | 91121 |
| stress/s0 | Overall | 835 | 207 | 0.6098726114649682 | 0.3503184713375796 | 0.03980891719745223 | 675 | 0.8083832335329342 | 558 | 0.6682634730538922 | 196717 | 0.12898089171974522 | 250915 |
| stress/s0 | Urban | 339 | 53 | 0.46153846153846156 | 0.5209790209790209 | 0.017482517482517484 | 364 | 1.0737463126843658 | 187 | 0.551622418879056 | 146955 | 0.11538461538461539 | 156621 |
| stress/s0 | Rural | 496 | 154 | 0.7339181286549707 | 0.20760233918128654 | 0.05847953216374269 | 311 | 0.6270161290322581 | 371 | 0.7479838709677419 | 49762 | 0.14035087719298245 | 94294 |
| stress/s0 | shadow | 279 | 72 | 0.5893719806763285 | 0.391304347826087 | 0.01932367149758454 | 208 | 0.7455197132616488 | 162 | 0.5806451612903226 | 76043 | 0.14009661835748793 | 119053 |
| stress/s0 | Urban/shadow | 113 | 21 | 0.41304347826086957 | 0.5869565217391305 | 0.0 | 116 | 1.0265486725663717 | 52 | 0.46017699115044247 | 51740 | 0.11956521739130435 | 79763 |
| stress/s0 | Rural/shadow | 166 | 51 | 0.7304347826086957 | 0.23478260869565218 | 0.034782608695652174 | 92 | 0.5542168674698795 | 110 | 0.6626506024096386 | 24303 | 0.1565217391304348 | 39290 |
| stress/s0 | occlusion | 278 | 69 | 0.6076555023923444 | 0.3349282296650718 | 0.05741626794258373 | 237 | 0.8525179856115108 | 193 | 0.6942446043165468 | 58165 | 0.12440191387559808 | 55695 |
| stress/s0 | Urban/occlusion | 113 | 17 | 0.4479166666666667 | 0.5104166666666666 | 0.041666666666666664 | 132 | 1.168141592920354 | 56 | 0.49557522123893805 | 43683 | 0.09375 | 28954 |
| stress/s0 | Rural/occlusion | 165 | 52 | 0.7433628318584071 | 0.18584070796460178 | 0.07079646017699115 | 105 | 0.6363636363636364 | 137 | 0.8303030303030303 | 14482 | 0.1504424778761062 | 26741 |
| stress/s0 | blur | 278 | 66 | 0.6320754716981132 | 0.32547169811320753 | 0.04245283018867924 | 230 | 0.8273381294964028 | 203 | 0.7302158273381295 | 62509 | 0.12264150943396226 | 76167 |
| stress/s0 | Urban/blur | 113 | 15 | 0.5204081632653061 | 0.46938775510204084 | 0.01020408163265306 | 116 | 1.0265486725663717 | 79 | 0.6991150442477876 | 51532 | 0.1326530612244898 | 47904 |
| stress/s0 | Rural/blur | 165 | 51 | 0.7280701754385965 | 0.20175438596491227 | 0.07017543859649122 | 114 | 0.6909090909090909 | 124 | 0.7515151515151515 | 10977 | 0.11403508771929824 | 28263 |
| ordinary/s1 | Overall | 835 | 207 | 0.5971337579617835 | 0.3519108280254777 | 0.050955414012738856 | 365 | 0.437125748502994 | 386 | 0.4622754491017964 | 700 | 0.0 | 99125 |
| ordinary/s1 | Urban | 339 | 53 | 0.44755244755244755 | 0.5244755244755245 | 0.027972027972027972 | 179 | 0.528023598820059 | 121 | 0.35693215339233036 | 658 | 0.0 | 54486 |
| ordinary/s1 | Rural | 496 | 154 | 0.7222222222222222 | 0.20760233918128654 | 0.07017543859649122 | 186 | 0.375 | 265 | 0.5342741935483871 | 42 | 0.0 | 44639 |
| stress/s1 | Overall | 835 | 207 | 0.6242038216560509 | 0.3359872611464968 | 0.03980891719745223 | 340 | 0.40718562874251496 | 399 | 0.47784431137724553 | 700 | 0.0 | 97734 |
| stress/s1 | Urban | 339 | 53 | 0.46503496503496505 | 0.506993006993007 | 0.027972027972027972 | 164 | 0.4837758112094395 | 130 | 0.3834808259587021 | 652 | 0.0 | 51744 |
| stress/s1 | Rural | 496 | 154 | 0.7573099415204678 | 0.19298245614035087 | 0.049707602339181284 | 176 | 0.3548387096774194 | 269 | 0.5423387096774194 | 48 | 0.0 | 45990 |
| stress/s1 | shadow | 279 | 72 | 0.5942028985507246 | 0.3864734299516908 | 0.01932367149758454 | 118 | 0.4229390681003584 | 107 | 0.3835125448028674 | 246 | 0.0 | 44155 |
| stress/s1 | Urban/shadow | 113 | 21 | 0.42391304347826086 | 0.5760869565217391 | 0.0 | 57 | 0.504424778761062 | 31 | 0.2743362831858407 | 215 | 0.0 | 25461 |
| stress/s1 | Rural/shadow | 166 | 51 | 0.7304347826086957 | 0.23478260869565218 | 0.034782608695652174 | 61 | 0.3674698795180723 | 76 | 0.4578313253012048 | 31 | 0.0 | 18694 |
| stress/s1 | occlusion | 278 | 69 | 0.6411483253588517 | 0.3014354066985646 | 0.05741626794258373 | 111 | 0.39928057553956836 | 146 | 0.5251798561151079 | 217 | 0.0 | 23741 |
| stress/s1 | Urban/occlusion | 113 | 17 | 0.4583333333333333 | 0.4895833333333333 | 0.052083333333333336 | 57 | 0.504424778761062 | 38 | 0.336283185840708 | 210 | 0.0 | 9149 |
| stress/s1 | Rural/occlusion | 165 | 52 | 0.7964601769911505 | 0.1415929203539823 | 0.061946902654867256 | 54 | 0.32727272727272727 | 108 | 0.6545454545454545 | 7 | 0.0 | 14592 |
| stress/s1 | blur | 278 | 66 | 0.6367924528301887 | 0.32075471698113206 | 0.04245283018867924 | 111 | 0.39928057553956836 | 146 | 0.5251798561151079 | 237 | 0.0 | 29838 |
| stress/s1 | Urban/blur | 113 | 15 | 0.5102040816326531 | 0.45918367346938777 | 0.030612244897959183 | 50 | 0.4424778761061947 | 61 | 0.5398230088495575 | 227 | 0.0 | 17134 |
| stress/s1 | Rural/blur | 165 | 51 | 0.7456140350877193 | 0.20175438596491227 | 0.05263157894736842 | 61 | 0.3696969696969697 | 85 | 0.5151515151515151 | 10 | 0.0 | 12704 |
| ordinary/s2 | Overall | 835 | 207 | 0.6050955414012739 | 0.35987261146496813 | 0.03503184713375796 | 663 | 0.7940119760479042 | 585 | 0.7005988023952096 | 10387 | 0.31369426751592355 | 213927 |
| ordinary/s2 | Urban | 339 | 53 | 0.44405594405594406 | 0.5384615384615384 | 0.017482517482517484 | 376 | 1.1091445427728615 | 194 | 0.5722713864306784 | 9098 | 0.43006993006993005 | 119535 |
| ordinary/s2 | Rural | 496 | 154 | 0.7397660818713451 | 0.21052631578947367 | 0.049707602339181284 | 287 | 0.5786290322580645 | 391 | 0.7883064516129032 | 1289 | 0.21637426900584794 | 94392 |
| stress/s2 | Overall | 835 | 207 | 0.6210191082802548 | 0.34554140127388533 | 0.03343949044585987 | 666 | 0.7976047904191617 | 577 | 0.6910179640718563 | 9878 | 0.3057324840764331 | 216352 |
| stress/s2 | Urban | 339 | 53 | 0.46503496503496505 | 0.5174825174825175 | 0.017482517482517484 | 356 | 1.0501474926253687 | 192 | 0.5663716814159292 | 8635 | 0.4195804195804196 | 117234 |
| stress/s2 | Rural | 496 | 154 | 0.7514619883040936 | 0.20175438596491227 | 0.04678362573099415 | 310 | 0.625 | 385 | 0.7762096774193549 | 1243 | 0.21052631578947367 | 99118 |
| stress/s2 | shadow | 279 | 72 | 0.5942028985507246 | 0.3864734299516908 | 0.01932367149758454 | 205 | 0.7347670250896058 | 167 | 0.5985663082437276 | 3781 | 0.3188405797101449 | 101822 |
| stress/s2 | Urban/shadow | 113 | 21 | 0.41304347826086957 | 0.5869565217391305 | 0.0 | 113 | 1.0 | 55 | 0.48672566371681414 | 3198 | 0.4673913043478261 | 62530 |
| stress/s2 | Rural/shadow | 166 | 51 | 0.7391304347826086 | 0.22608695652173913 | 0.034782608695652174 | 92 | 0.5542168674698795 | 112 | 0.6746987951807228 | 583 | 0.2 | 39292 |
| stress/s2 | occlusion | 278 | 69 | 0.6363636363636364 | 0.3157894736842105 | 0.04784688995215311 | 230 | 0.8273381294964028 | 197 | 0.7086330935251799 | 2505 | 0.3157894736842105 | 47477 |
| stress/s2 | Urban/occlusion | 113 | 17 | 0.4479166666666667 | 0.5104166666666666 | 0.041666666666666664 | 126 | 1.1150442477876106 | 51 | 0.45132743362831856 | 2200 | 0.4270833333333333 | 18084 |
| stress/s2 | Rural/occlusion | 165 | 52 | 0.7964601769911505 | 0.1504424778761062 | 0.05309734513274336 | 104 | 0.6303030303030303 | 146 | 0.8848484848484849 | 305 | 0.22123893805309736 | 29393 |
| stress/s2 | blur | 278 | 66 | 0.6320754716981132 | 0.33490566037735847 | 0.0330188679245283 | 231 | 0.8309352517985612 | 213 | 0.7661870503597122 | 3592 | 0.2830188679245283 | 67053 |
| stress/s2 | Urban/blur | 113 | 15 | 0.5306122448979592 | 0.45918367346938777 | 0.01020408163265306 | 117 | 1.0353982300884956 | 86 | 0.7610619469026548 | 3237 | 0.3673469387755102 | 36620 |
| stress/s2 | Rural/blur | 165 | 51 | 0.7192982456140351 | 0.22807017543859648 | 0.05263157894736842 | 114 | 0.6909090909090909 | 127 | 0.7696969696969697 | 355 | 0.21052631578947367 | 30433 |

| Run | Group | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- |
| ordinary/s0 | Overall | -0.055082332311468814 | -0.011787772929950341 | 0.010126506518248743 | 0.053627312598837594 | 0.14619759836138382 |
| ordinary/s0 | Urban | -0.06806497131990236 | -0.025569477345950137 | -0.0023683784926480933 | 0.020877886177818578 | 0.10782280495054118 |
| ordinary/s0 | Rural | -0.035247897184657526 | 0.0 | 0.02919847648353671 | 0.07619420938023501 | 0.168431186871901 |
| stress/s0 | Overall | -0.056019004769701786 | -0.0113426730635727 | 0.01081407033319115 | 0.05408268141865878 | 0.14472168242335234 |
| stress/s0 | Urban | -0.07123190709389124 | -0.024911413874609767 | -0.0006960884804956824 | 0.022596028321995143 | 0.10902861450360377 |
| stress/s0 | Rural | -0.03238099861315283 | 0.0 | 0.02937705991369355 | 0.07465095909529554 | 0.16254777809145288 |
| stress/s0 | shadow | -0.06300952065145098 | -0.017257746943839358 | 0.009322409112144103 | 0.0508373434122566 | 0.1554983727058136 |
| stress/s0 | Urban/shadow | -0.07006345010350316 | -0.024985783491473457 | -0.005574731886882844 | 0.017122361617569026 | 0.07171436773768876 |
| stress/s0 | Rural/shadow | -0.041597832609567204 | 0.0 | 0.028059713617230003 | 0.0813268152940946 | 0.16204792013893424 |
| stress/s0 | occlusion | -0.04820302728528462 | -0.007445407856016439 | 0.013399964742700499 | 0.05774171690081942 | 0.14882998582436768 |
| stress/s0 | Urban/occlusion | -0.07528974341163733 | -0.030201247026106992 | -0.0013471834557121365 | 0.018441395780372988 | 0.11399927867520086 |
| stress/s0 | Rural/occlusion | -0.0199280378516554 | 0.0 | 0.03730057237508527 | 0.0732451760465776 | 0.17232019321870357 |
| stress/s0 | blur | -0.04748956809553935 | -0.008892308668978844 | 0.010445189821018774 | 0.05142910381765073 | 0.12795241425240295 |
| stress/s0 | Urban/blur | -0.058171992074783865 | -0.01923214053063299 | 0.0011423895807360362 | 0.036129130442817486 | 0.11283196787925968 |
| stress/s0 | Rural/blur | -0.0321699230659698 | 3.227705211023002e-06 | 0.020857327736448286 | 0.07013951857153211 | 0.1376400375933763 |
| ordinary/s1 | Overall | -0.017144180040856657 | -0.0027862994592197377 | 0.0036825776838887714 | 0.0266564005657768 | 0.0791381580613446 |
| ordinary/s1 | Urban | -0.018567692284784054 | -0.0062754462540098865 | -0.00041540674494316976 | 0.007674512601586754 | 0.043891057338347195 |
| ordinary/s1 | Rural | -0.013503555764454065 | 0.0 | 0.013427579781410959 | 0.04211020017661696 | 0.085763174828913 |
| stress/s1 | Overall | -0.017625492330837877 | -0.0023342860967103807 | 0.004756682691082814 | 0.024237777038919175 | 0.07974479571031014 |
| stress/s1 | Urban | -0.020382806526396423 | -0.005630661809255928 | -0.0003314299542296617 | 0.007421027652345034 | 0.045110896574847195 |
| stress/s1 | Rural | -0.011857494880160371 | 0.0003604785598337462 | 0.01310557633310494 | 0.04173627589248438 | 0.09624249630853678 |
| stress/s1 | shadow | -0.01832560787149474 | -0.003693202836918208 | 0.003353127951573831 | 0.026261822459903467 | 0.0763141512555949 |
| stress/s1 | Urban/shadow | -0.02820970122539218 | -0.007734002919123413 | -0.0010415346246668844 | 0.004844985199621066 | 0.023082651817439608 |
| stress/s1 | Rural/shadow | -0.01215824197640658 | 0.0 | 0.014552438561617986 | 0.04551249257096726 | 0.1008383821236002 |
| stress/s1 | occlusion | -0.012811265778174374 | -0.0017414074100330357 | 0.0058829118226310095 | 0.026234330726210897 | 0.08478377877864103 |
| stress/s1 | Urban/occlusion | -0.017835696553565976 | -0.006072724873136093 | 0.0 | 0.006638480083375808 | 0.04792284501010756 |
| stress/s1 | Rural/occlusion | -0.00768243712343144 | 0.0019799562122064795 | 0.016899518025009552 | 0.0394314266102227 | 0.09958412917900786 |
| stress/s1 | blur | -0.019157390778247418 | -0.002145982876256808 | 0.004256086842176043 | 0.02098648506309101 | 0.07289916837908009 |
| stress/s1 | Urban/blur | -0.019878736365019526 | -0.0037922274297878256 | 0.000132308875185283 | 0.012915345627847491 | 0.05170056381814786 |
| stress/s1 | Rural/blur | -0.015385790832633034 | 9.762992859043223e-05 | 0.007546097020393272 | 0.035877384490145645 | 0.08063188774401042 |
| ordinary/s2 | Overall | -0.040843679919302316 | -0.009624427136761454 | 0.00939375789936231 | 0.0494199841646909 | 0.1478939736498648 |
| ordinary/s2 | Urban | -0.0511211337300438 | -0.019129578931798286 | -0.0019317414652452736 | 0.01630007207551193 | 0.07774473557192055 |
| ordinary/s2 | Rural | -0.030368367518592863 | 0.0 | 0.026597171897234567 | 0.08078548698845295 | 0.16746607233723967 |
| stress/s2 | Overall | -0.040981670264390084 | -0.009261934086492079 | 0.0099170425715146 | 0.054800604770407796 | 0.14090493465731949 |
| stress/s2 | Urban | -0.049451551844674674 | -0.018362237776593293 | -0.0014075060715712413 | 0.016548109348712634 | 0.08213463364287735 |
| stress/s2 | Rural | -0.02782835724320467 | 0.00027235373950453565 | 0.024258847287835145 | 0.079700230440203 | 0.17300566904002676 |
| stress/s2 | shadow | -0.04691392235897237 | -0.011737375220072899 | 0.008204867168298602 | 0.055984137801434436 | 0.1449690075303294 |
| stress/s2 | Urban/shadow | -0.058733045952796124 | -0.02105641024252028 | -0.0053945402180990976 | 0.010353661625747634 | 0.054740788074503305 |
| stress/s2 | Rural/shadow | -0.03271767736734763 | 0.0 | 0.024575065820919284 | 0.08991860421830727 | 0.18589996497468228 |
| stress/s2 | occlusion | -0.03591823995655232 | -0.007605584638250118 | 0.012115532168385679 | 0.06135197792095792 | 0.154481216034959 |
| stress/s2 | Urban/occlusion | -0.055358598806322934 | -0.02041923541222801 | -0.0005840541919039377 | 0.015010061601436087 | 0.08690937085756002 |
| stress/s2 | Rural/occlusion | -0.01800653459696122 | 0.004912872649744882 | 0.030891878809216622 | 0.07917907300738275 | 0.16748834137676008 |
| stress/s2 | blur | -0.033592800240070955 | -0.008211535549024013 | 0.009202969351483264 | 0.046704628806528516 | 0.13043899249309365 |
| stress/s2 | Urban/blur | -0.03383636158746293 | -0.011893456904344818 | 0.0030870135016175526 | 0.023604245819191474 | 0.0863761258487109 |
| stress/s2 | Rural/blur | -0.032399376109385626 | 0.0 | 0.017057343673785155 | 0.06410780472929918 | 0.14052459697485353 |

Observed all detailed per-K official Counts, official scores, per-seed grouped totals and per-image data are in metrics.json/per_image.csv. No-GT A is blank in CSV; *_official_A retains the official definition. Confidence/probability maps and features are not report artifacts.

Observed visualization pools and selected counts:

{
  "clean_Urban_positive": {
    "candidates": 129,
    "selected": 4
  },
  "clean_Urban_negative": {
    "candidates": 152,
    "selected": 5
  },
  "clean_Rural_positive": {
    "candidates": 248,
    "selected": 5
  },
  "clean_Rural_negative": {
    "candidates": 75,
    "selected": 4
  },
  "false_connection": {
    "candidates": 720,
    "selected": 5
  },
  "correct_connection": {
    "candidates": 539,
    "selected": 5
  },
  "later_degradation": {
    "candidates": 147,
    "selected": 5
  },
  "stress_shadow": {
    "candidates": 81,
    "selected": 5
  },
  "stress_occlusion": {
    "candidates": 70,
    "selected": 5
  },
  "stress_blur": {
    "candidates": 69,
    "selected": 5
  },
  "no_gt_additions": {
    "candidates": 218,
    "selected": 4
  }
}

Observed model-seed means and sample standard deviations (ddof=1; no averaging across domains/categories):

| Condition/group | Better | Worse | Unchanged | False merges | Correct merges | Disappearing pixels | Later degradation fraction | No-GT added pixels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary/Overall | 0.599788 +/- 0.004597 | 0.357219 +/- 0.004597 | 0.042994 +/- 0.007962 | 572.333333 +/- 180.025924 | 503.000000 +/- 104.014422 | 70427.666667 +/- 112486.998112 | 0.139597 +/- 0.159668 | 188275.666667 +/- 79492.130436 |
| ordinary/Urban | 0.446387 +/- 0.002019 | 0.530303 +/- 0.007279 | 0.023310 +/- 0.005341 | 311.333333 +/- 114.613844 | 166.333333 +/- 39.576929 | 52576.000000 +/- 82723.068197 | 0.173660 +/- 0.226662 | 111558.333333 +/- 53531.593142 |
| ordinary/Rural | 0.728070 +/- 0.010129 | 0.212476 +/- 0.006087 | 0.059454 +/- 0.010269 | 261.000000 +/- 65.962110 | 336.666667 +/- 64.763673 | 17851.666667 +/- 29773.842989 | 0.111111 +/- 0.108306 | 76717.333333 +/- 27828.752439 |
| stress/Overall | 0.618365 +/- 0.007525 | 0.343949 +/- 0.007297 | 0.037686 +/- 0.003677 | 560.333333 +/- 190.867319 | 511.333333 +/- 97.746270 | 69098.333333 +/- 110616.237426 | 0.144904 +/- 0.153487 | 188333.666667 +/- 80342.236354 |
| stress/Urban | 0.463869 +/- 0.002019 | 0.515152 +/- 0.007279 | 0.020979 +/- 0.006056 | 294.666667 +/- 113.231327 | 169.666667 +/- 34.443190 | 52080.666667 +/- 82260.479067 | 0.178322 +/- 0.216755 | 108533.000000 +/- 52977.135002 |
| stress/Rural | 0.747563 +/- 0.012173 | 0.200780 +/- 0.007359 | 0.051657 +/- 0.006087 | 265.666667 +/- 77.655221 | 341.666667 +/- 63.319297 | 17017.666667 +/- 28363.718556 | 0.116959 +/- 0.107195 | 79800.666667 +/- 29380.071976 |
| stress/shadow | 0.592593 +/- 0.002789 | 0.388084 +/- 0.002789 | 0.019324 +/- 0.000000 | 177.000000 +/- 51.117512 | 145.333333 +/- 33.291641 | 26690.000000 +/- 42777.482546 | 0.152979 +/- 0.159810 | 88343.333333 +/- 39226.055656 |
| stress/Urban/shadow | 0.416667 +/- 0.006276 | 0.583333 +/- 0.006276 | 0.000000 +/- 0.000000 | 95.333333 +/- 33.231511 | 46.000000 +/- 13.076697 | 18384.333333 +/- 28925.333988 | 0.195652 +/- 0.242808 | 55918.000000 +/- 27748.255963 |
| stress/Rural/shadow | 0.733333 +/- 0.005020 | 0.231884 +/- 0.005020 | 0.034783 +/- 0.000000 | 81.666667 +/- 17.897858 | 99.333333 +/- 20.231988 | 8305.666667 +/- 13856.846010 | 0.118841 +/- 0.105190 | 32425.333333 +/- 11891.683537 |
| stress/occlusion | 0.628389 +/- 0.018115 | 0.317384 +/- 0.016803 | 0.054226 +/- 0.005525 | 192.666667 +/- 70.811958 | 178.666667 +/- 28.360771 | 20295.666667 +/- 32815.751421 | 0.146730 +/- 0.159074 | 42304.333333 +/- 16593.127774 |
| stress/Urban/occlusion | 0.451389 +/- 0.006014 | 0.503472 +/- 0.012028 | 0.045139 +/- 0.006014 | 105.000000 +/- 41.677332 | 48.333333 +/- 9.291573 | 15364.333333 +/- 24544.860691 | 0.173611 +/- 0.224462 | 18729.000000 +/- 9918.242032 |
| stress/Rural/occlusion | 0.778761 +/- 0.030656 | 0.159292 +/- 0.023414 | 0.061947 +/- 0.008850 | 87.666667 +/- 29.160476 | 130.333333 +/- 19.857828 | 4931.333333 +/- 8272.461927 | 0.123894 +/- 0.112984 | 23575.333333 +/- 7891.988617 |
| stress/blur | 0.633648 +/- 0.002723 | 0.327044 +/- 0.007205 | 0.039308 +/- 0.005447 | 190.666667 +/- 68.995169 | 187.333333 +/- 36.143234 | 22112.666667 +/- 35024.445982 | 0.135220 +/- 0.141928 | 57686.000000 +/- 24543.829713 |
| stress/Urban/blur | 0.520408 +/- 0.010204 | 0.462585 +/- 0.005891 | 0.017007 +/- 0.011783 | 94.333333 +/- 38.397048 | 75.333333 +/- 12.897028 | 18332.000000 +/- 28791.405402 | 0.166667 +/- 0.186021 | 33886.000000 +/- 15566.126429 |
| stress/Rural/blur | 0.730994 +/- 0.013399 | 0.210526 +/- 0.015193 | 0.058480 +/- 0.010129 | 96.333333 +/- 30.599564 | 112.000000 +/- 23.430749 | 3780.666667 +/- 6234.594320 | 0.108187 +/- 0.105385 | 23800.000000 +/- 9670.477599 |

## Verification and summary tables

Observed job8506494 completed on NVIDIA L40S at source HEAD 47cd7bcece9ea535005bd0d3e68ff63f47bbb203. Scheduler accounting records failed=0, exit_status=0 and wallclock=1193 seconds (19 minutes 53 seconds). Log lines 1/4 identify GPU/HEAD; lines 6-8 and 11-13 contain six direct Count/score PASS checks; lines 9/14 record cache removal; line 15 records COMPLETE records=5010 visualizations=52. No Traceback, Killed, h_rt, Disk quota or No space was observed.

Observed 5010 CSV and JSON records = three seeds * two paired conditions * 835 images. All 225 grouped per-K Count/score recomputations match per-image records exactly. All 30 Overall A comparisons against controlled clean and stress-seed523 results pass tolerance 1e-5, with maximum absolute difference 1.812761714070632e-7. CSV cells contain identifiers/categories/scalars, not source paths. IDs such as urban/3515 are dataset-relative identifiers, not filesystem paths.

Observed tables use K4 unless stated otherwise, and trained seeds 0/1/2. Means and sample standard deviations use ddof=1; no three-seed predictions are pooled as if independent images. Group counts, area totals and per-image distributions are computed within each seed before seed averaging. Percent columns below multiply fractions by 100.

Observed delta-A distributions exclude annotation-empty images: analysis A is not applicable there. Official Counts and official single-image scores remain available and those images stay in official aggregate scores. Better means delta >= 1e-4, worse means delta <= -1e-4, unchanged means abs(delta)<1e-4. GT-empty means annotation-empty, not proof that no road is visible.

### Table 1. Observed per-image K4-minus-y0 A distribution

| Condition | Domain | Applicable images | Better % | Worse % | Unchanged % | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary | Urban | 286 +/- 0 | 44.638695 +/- 0.201871 | 53.030303 +/- 0.727855 | 2.331002 +/- 0.534100 | -0.045918 +/- 0.025156 | -0.016992 +/- 0.009823 | -0.001572 +/- 0.001025 | 0.014951 +/- 0.006704 | 0.076486 +/- 0.031984 |
| ordinary | Rural | 342 +/- 0 | 72.807018 +/- 1.012895 | 21.247563 +/- 0.608674 | 5.945419 +/- 1.026867 | -0.026373 +/- 0.011409 | 0.000000 +/- 0.000000 | 0.023074 +/- 0.008455 | 0.066363 +/- 0.021129 | 0.140553 +/- 0.047452 |
| shadow | Urban | 92 +/- 0 | 41.666667 +/- 0.627555 | 58.333333 +/- 0.627555 | 0.000000 +/- 0.000000 | -0.052335 +/- 0.021648 | -0.017925 +/- 0.009042 | -0.004004 +/- 0.002567 | 0.010774 +/- 0.006149 | 0.049846 +/- 0.024683 |
| shadow | Rural | 115 +/- 0 | 73.333333 +/- 0.502044 | 23.188406 +/- 0.502044 | 3.478261 +/- 0.000000 | -0.028825 +/- 0.015101 | 0.000000 +/- 0.000000 | 0.022396 +/- 0.007012 | 0.072253 +/- 0.023553 | 0.149595 +/- 0.043877 |
| occlusion | Urban | 96 +/- 0 | 45.138889 +/- 0.601407 | 50.347222 +/- 1.202813 | 4.513889 +/- 0.601407 | -0.049495 +/- 0.029172 | -0.018898 +/- 0.012136 | -0.000644 +/- 0.000676 | 0.013363 +/- 0.006071 | 0.082944 +/- 0.033216 |
| occlusion | Rural | 113 +/- 0 | 77.876106 +/- 3.065577 | 15.929204 +/- 2.341373 | 6.194690 +/- 0.884956 | -0.015206 +/- 0.006586 | 0.002298 +/- 0.002472 | 0.028364 +/- 0.010433 | 0.063952 +/- 0.021442 | 0.146464 +/- 0.040671 |
| blur | Urban | 98 +/- 0 | 52.040816 +/- 1.020408 | 46.258503 +/- 0.589133 | 1.700680 +/- 1.178266 | -0.037296 +/- 0.019380 | -0.011639 +/- 0.007723 | 0.001454 +/- 0.001502 | 0.024216 +/- 0.011619 | 0.083636 +/- 0.030658 |
| blur | Rural | 114 +/- 0 | 73.099415 +/- 1.339934 | 21.052632 +/- 1.519343 | 5.847953 +/- 1.012895 | -0.026652 +/- 0.009757 | 0.000034 +/- 0.000055 | 0.015154 +/- 0.006857 | 0.056708 +/- 0.018290 | 0.119599 +/- 0.033777 |

### Table 2. Observed connection proxy

Observed one predicted 8-connected component overlapping >=2 y0 components counts as one merge. Added GT-road fraction <50% makes it false, otherwise correct. Each total is the per-seed image-set count, reported as mean +/- sample sd. False share = false/(false+correct), computed within each seed before averaging. These are geometric annotation-based events, not verified road-graph edges.

| Condition | Domain | False total | False/image | Correct total | Correct/image | False share % |
| --- | --- | --- | --- | --- | --- | --- |
| ordinary | Urban | 311.333 +/- 114.614 | 0.918387 +/- 0.338094 | 166.333 +/- 39.577 | 0.490659 +/- 0.116746 | 64.316506 +/- 4.083311 |
| ordinary | Rural | 261.000 +/- 65.962 | 0.526210 +/- 0.132988 | 336.667 +/- 64.764 | 0.678763 +/- 0.130572 | 43.419605 +/- 2.881315 |
| shadow | Urban | 95.333 +/- 33.232 | 0.843658 +/- 0.294084 | 46.000 +/- 13.077 | 0.407080 +/- 0.115723 | 67.027417 +/- 2.147071 |
| shadow | Rural | 81.667 +/- 17.898 | 0.491968 +/- 0.107818 | 99.333 +/- 20.232 | 0.598394 +/- 0.121879 | 45.056047 +/- 0.510800 |
| occlusion | Urban | 105.000 +/- 41.677 | 0.929204 +/- 0.368826 | 48.333 +/- 9.292 | 0.427729 +/- 0.082226 | 67.133069 +/- 6.196573 |
| occlusion | Rural | 87.667 +/- 29.160 | 0.531313 +/- 0.176730 | 130.333 +/- 19.858 | 0.789899 +/- 0.120350 | 39.440588 +/- 5.364097 |
| blur | Urban | 94.333 +/- 38.397 | 0.834808 +/- 0.339797 | 75.333 +/- 12.897 | 0.666667 +/- 0.114133 | 54.055898 +/- 7.858359 |
| blur | Rural | 96.333 +/- 30.600 | 0.583838 +/- 0.185452 | 112.000 +/- 23.431 | 0.678788 +/- 0.142005 | 45.660962 +/- 3.373499 |

### Table 3. Observed disappearing roads and later-step degradation

Observed disappearing area counts valid GT-road pixels covered by y0 and deleted by K4. This differs from the skeleton-only new-break count. Later-step degradation uses A(K8)-A(K4)<-0.01, excluding annotation-empty images from its denominator.

| Condition | Domain | Disappearing GT-road pixels | K8 degradation images | K8 degradation % |
| --- | --- | --- | --- | --- |
| ordinary | Urban | 52576.000 +/- 82723.068 | 49.667 +/- 64.825 | 17.365967 +/- 22.666226 |
| ordinary | Rural | 17851.667 +/- 29773.843 | 38.000 +/- 37.041 | 11.111111 +/- 10.830561 |
| shadow | Urban | 18384.333 +/- 28925.334 | 18.000 +/- 22.338 | 19.565217 +/- 24.280769 |
| shadow | Rural | 8305.667 +/- 13856.846 | 13.667 +/- 12.097 | 11.884058 +/- 10.518984 |
| occlusion | Urban | 15364.333 +/- 24544.861 | 16.667 +/- 21.548 | 17.361111 +/- 22.446245 |
| occlusion | Rural | 4931.333 +/- 8272.462 | 14.000 +/- 12.767 | 12.389381 +/- 11.298359 |
| blur | Urban | 18332.000 +/- 28791.405 | 16.333 +/- 18.230 | 16.666667 +/- 18.602053 |
| blur | Rural | 3780.667 +/- 6234.594 | 12.333 +/- 12.014 | 10.818713 +/- 10.538492 |

### Table 4. Observed annotation-empty images

Observed Val-rep has 207 distinct annotation-empty image IDs, repeated as paired ordinary/stress inputs and across three model seeds. It is not 414 independent images. Per-image distributions include zero-added images and use added_correct+added_wrong at K4. Empty-annotation images have no analysis A.

| Condition | Domain | No-GT images | Total added pixels | Added/image mean | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary | Urban | 53 +/- 0 | 111558.3 +/- 53531.6 | 2104.9 +/- 1010.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 351.7 +/- 221.1 | 2596.3 +/- 1288.2 | 8401.3 +/- 3739.0 |
| ordinary | Rural | 154 +/- 0 | 76717.3 +/- 27828.8 | 498.2 +/- 180.7 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 1.3 +/- 1.3 | 536.6 +/- 198.4 | 2493.0 +/- 998.7 |
| shadow | Urban | 21 +/- 0 | 55918.0 +/- 27748.3 | 2662.8 +/- 1321.3 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 338.0 +/- 194.7 | 3360.3 +/- 1640.1 | 9882.0 +/- 4900.0 |
| shadow | Rural | 51 +/- 0 | 32425.3 +/- 11891.7 | 635.8 +/- 233.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 10.3 +/- 6.4 | 1091.5 +/- 428.2 | 2469.0 +/- 1046.8 |
| occlusion | Urban | 17 +/- 0 | 18729.0 +/- 9918.2 | 1101.7 +/- 583.4 | 0.0 +/- 0.0 | 21.7 +/- 21.5 | 300.0 +/- 136.5 | 1827.3 +/- 912.7 | 3369.4 +/- 1943.6 |
| occlusion | Rural | 52 +/- 0 | 23575.3 +/- 7892.0 | 453.4 +/- 151.8 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 284.1 +/- 81.0 | 2770.8 +/- 976.3 |
| blur | Urban | 15 +/- 0 | 33886.0 +/- 15566.1 | 2259.1 +/- 1037.7 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 360.0 +/- 243.2 | 4814.5 +/- 2141.3 | 7757.4 +/- 3577.9 |
| blur | Rural | 51 +/- 0 | 23800.0 +/- 9670.5 | 466.7 +/- 189.6 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 564.5 +/- 256.1 | 2026.5 +/- 830.4 |

Observed top five annotation-empty IDs per paired condition, ranked by mean K4 added pixels over three model seeds (not by A); ordinary and stress lists may overlap:

| Condition | Rank | ID | Domain | Added pixels mean +/- sd | Added pixels seeds 0;1;2 |
| --- | --- | --- | --- | --- | --- |
| ordinary | 1 | urban/3738 | Urban | 12046.0 +/- 6100.0 | 17866; 5700; 12572 |
| ordinary | 2 | urban/3965 | Urban | 9344.0 +/- 5668.8 | 15649; 4668; 7715 |
| ordinary | 3 | urban/3942 | Urban | 8916.7 +/- 4867.3 | 11300; 3317; 12133 |
| ordinary | 4 | urban/3958 | Urban | 8389.7 +/- 3311.5 | 11101; 4699; 9369 |
| ordinary | 5 | urban/3960 | Urban | 8284.0 +/- 3918.0 | 11248; 3842; 9762 |
| stress | 1 | urban/3738 | Urban | 11100.3 +/- 5587.6 | 16295; 5189; 11817 |
| stress | 2 | urban/3960 | Urban | 9377.3 +/- 4785.9 | 13519; 4138; 10475 |
| stress | 3 | urban/3958 | Urban | 8995.0 +/- 4162.4 | 12492; 4391; 10102 |
| stress | 4 | urban/3942 | Urban | 8698.0 +/- 4909.7 | 10898; 3073; 12123 |
| stress | 5 | urban/3965 | Urban | 8195.0 +/- 4750.4 | 13491; 4310; 6784 |

### Table 5. Observed visualization index

Observed all 52 panels use model seed0, size 5120x1064, and stay under ignored runs/failure_vis. Selection uses sorted pools and RNG seed0: up to three random cases plus delta extrema per category. Within-category duplicates are consolidated, so a case may have both random and extreme reasons. No-GT delta extrema are inapplicable; that category includes the maximum-added case. Random sampling is conditional on the declared category; explicit extrema are not random cases.

| Category | Candidates | Panels | Random cases | Extrema/addition maximum |
| --- | --- | --- | --- | --- |
| clean_Urban_positive | 129 | 4 | ordinary:urban/3893; ordinary:urban/3820; ordinary:urban/4046 | minimum_delta=ordinary:urban/3820; maximum_delta=ordinary:urban/3520 |
| clean_Urban_negative | 152 | 5 | ordinary:urban/3522; ordinary:urban/3575; ordinary:urban/3545 | minimum_delta=ordinary:urban/3663; maximum_delta=ordinary:urban/4146 |
| clean_Rural_positive | 248 | 5 | ordinary:rural/3197; ordinary:rural/3065; ordinary:rural/3447 | minimum_delta=ordinary:rural/2551; maximum_delta=ordinary:rural/3462 |
| clean_Rural_negative | 75 | 4 | ordinary:rural/3177; ordinary:rural/3050; ordinary:rural/3129 | minimum_delta=ordinary:rural/3177; maximum_delta=ordinary:rural/3269 |
| false_connection | 720 | 5 | stress:urban/3702; stress:rural/3232; ordinary:urban/3592 | minimum_delta=stress:urban/3833; maximum_delta=ordinary:rural/3205 |
| correct_connection | 539 | 5 | stress:urban/3601; stress:rural/2758; ordinary:rural/2694 | minimum_delta=ordinary:rural/3177; maximum_delta=stress:rural/3462 |
| later_degradation | 147 | 5 | ordinary:rural/3186; stress:urban/3699; ordinary:rural/2957 | minimum_delta=ordinary:rural/2998; maximum_delta=stress:rural/3288 |
| stress_shadow | 81 | 5 | stress:rural/3472; stress:urban/3725; stress:rural/2801 | minimum_delta=stress:rural/3177; maximum_delta=stress:urban/3987 |
| stress_occlusion | 70 | 5 | stress:rural/2604; stress:rural/2598; stress:urban/3567 | minimum_delta=stress:urban/3643; maximum_delta=stress:urban/3979 |
| stress_blur | 69 | 5 | stress:urban/4189; stress:urban/3711; stress:urban/3828 | minimum_delta=stress:rural/3512; maximum_delta=stress:rural/3183 |
| no_gt_additions | 218 | 4 | stress:rural/3240; ordinary:urban/3734; ordinary:urban/3942 | maximum_added=ordinary:urban/3738 |

Observed provenance examples: groups["ordinary/s0"].Urban.delta_A4_y0.better = 0.44755244755244755; groups["stress/s0"]["Urban/occlusion"].groups["4"].false_connection = 132. Per-image area and no-GT rank inputs are per_image[i].steps["4"].added / disappearing_road_pixels / false_connection, equivalently per_image.csv columns 4_added / 4_disappearing_road_pixels / 4_false_connection, joined by (seed, condition, id). Quantiles in table 1 come from groups[run][group].delta_A4_y0.quantiles; table 3 uses K8_worse_than_K4_by_001. Table 5 copies runs/failure_vis/index.csv category/id/reasons, without submitting that runtime file.

Observed verification does not rerun inference or alter the job-produced metrics.json/per_image.csv. Node-local scratch is no longer directly inspectable; cleanup evidence is log lines 9 and 14.
