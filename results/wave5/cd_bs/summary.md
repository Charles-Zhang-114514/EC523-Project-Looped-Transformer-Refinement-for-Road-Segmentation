# Wave 5 CD/BS acceptance

## Gates

| Gate | Result | Observed evidence |
| --- | --- | --- |
| CG1 | PASS | Selection and final evaluation corruption_applied=false for all CD seeds. Both selected and actually modified fractions lie in [0.45,0.55]; see counts below. |
| BG1 | PASS | All BS evaluation reports carry delta_bound=1.0; all 30060 recorded step maxima <=1.0. |
| KG | PASS | 25050 clean/stress endpoints across six runs equal K; full seven-run count 29225. |
| Cache/configuration | PASS | All seven jobs use passed fp32 manifest 06d97973fcec14cc113b7c3468d768e1e0a6d9b3d27998fd6aa14cf07c13dc90; selection=834 images, D4 on, detail head off. |

## Preregistered decisions

| Prediction/rule | Result | Calculation |
| --- | --- | --- |
| CD-P1 | holds | 0.547493098 - 0.545751875 = 0.001741223; interval [-0.001,0.003] |
| CD-P2 | holds | 0.002126578 - 0.000273423 = 0.001853154 >= 0.001 |
| CD-P3 | holds | 0.702350481 - 0.697518971 = 0.004831510 > 0 |
| CD-P4 | does_not_hold | best epochs [1, 1, 2]; 1/3 >=2, required 2/3 |
| BS-P1 | does_not_hold | mean climb 0.000781832 vs 0.003, baseline 0.000273423 |
| BS-P2 | does_not_hold | K8-K4 -0.004419960, -0.008526572, -0.009158693; 0/3 >=-0.001 |
| BS-P3 | holds | 0.544899809 - 0.545215200 = -0.000315391; interval [-0.002,0.002] |
| CB-R1 (CD) | pass | CD_s0, Val-sel L=0.553449912; LB=0.549580012 / 0.576998154 / 0.497705772; delta E0=0.002944039 / -0.000977502 / 0.009884247; Overall >=0.548636 and Urban/Rural delta >=-0.003 |
| CB-R2 (CD) | not_climbing | 0/3 jointly satisfy climb>=0.003 and K8-K4>=-0.001; required 2/3 |
| CB-R1 (BS) | fail | BS_s2, Val-sel L=0.550146960; LB=0.545475219 / 0.576262193 / 0.487441294; delta E0=-0.001160754 / -0.001713463 / -0.000380231; Overall >=0.548636 and Urban/Rural delta >=-0.003 |
| CB-R2 (BS) | not_climbing | 0/3 jointly satisfy climb>=0.003 and K8-K4>=-0.001; required 2/3 |

Observed CB-R1 selects **CD_s0**; CB-R2 marks neither CD nor BS as achieving climbing.

## Jobs and protocol

Observed all jobs run on NVIDIA L40S in Exclusive_Process mode, complete ten epochs and FOCUS_EVAL_COMPLETE, and have accounting failed=0/exit_status=0. No Traceback/OOM/killed/timeout appears in the logs. Wall time comes from existing local accounting; memory is a saved allocated CUDA peak, not an external job-wide monitor.

| Job | Run | Wall time | Peak GiB | Best epoch |
| --- | --- | --- | --- | --- |
| 8821483 | CD_s0 | 01:00:53 | 6.219810 | 1 |
| 8821484 | CD_s1 | 00:59:04 | 6.219810 | 1 |
| 8821485 | CD_s2 | 01:00:53 | 6.219810 | 2 |
| 8821486 | BS_s0 | 01:00:52 | 6.266687 | 1 |
| 8821487 | BS_s1 | 00:56:51 | 6.266687 | 1 |
| 8821488 | BS_s2 | 00:59:00 | 6.266687 | 1 |
| 8821489 | HO_s0 | 00:58:29 | 6.219810 | 1 |

Observed CD/BS/HO use scratch initialization, lr=3e-4, 500 warmup steps + cosine, ten epochs, batch=4, K_train=4, BCE pos_weight=4, weight_decay=0.01 and clipping=1.0. Selection uses all 834 Val-sel images and L weights 0.2/0.2/0.6. CD changes only draft_corrupt=0.5; BS changes delta_bound=1 and step_weights=0.1/0.2/0.3/0.4; HO excludes 100 Train IDs/domain. All best.pt and last.pt files exist and remain uncommitted.

Observed head SHA=9429dacae36969948eddd5f4551c3fed9567a1acfee6526164fe1839bc3b568f; split SHA=bdd7d7a89237c96305f0ad404a5c9ae1254ed8cbc0b6e0e4fcc7b514bd323c68. The shared manifest records passed RG1/RG2, float32 and 4191 cached images.

## CD/BG gate measurements

| Run | Selected / presentations | Selected fraction | Actually modified | Modified fraction |
| --- | --- | --- | --- | --- |
| CD_s0 | 12784/25220 | 0.506899286 | 12460 | 0.494052339 |
| CD_s1 | 12463/25220 | 0.494171293 | 12136 | 0.481205393 |
| CD_s2 | 12758/25220 | 0.505868358 | 12458 | 0.493973037 |

| BS run | C at evaluation | Recorded step maxima | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| BS_s0 | 1.0 | 10020 | 0.179485664 | 0.381892353 |
| BS_s1 | 1.0 | 10020 | 0.131273359 | 0.246462986 |
| BS_s2 | 1.0 | 10020 | 0.060491111 | 0.209723264 |

## Per-seed scores

Observed each score below uses the selected best checkpoint. Domain order is Overall / Urban / Rural. Climb and tail are Overall A(K4)-A(K1) and A(K8)-A(K4).

| Run | Val-sel L (O/U/R) | Val-rep L (O/U/R) | Val-rep LB (O/U/R) | Climb | K8-K4 | K4 clDice | Best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CD_s0 | 0.553449912 / 0.567192279 / 0.526594422 | 0.550174639 / 0.577577070 / 0.498214592 | 0.549580012 / 0.576998154 / 0.497705772 | 0.004237166 | -0.001036903 | 0.707456992 | 1 |
| CD_s1 | 0.550084700 / 0.565431104 / 0.519887751 | 0.546090819 / 0.576083134 / 0.489160461 | 0.545663653 / 0.575669332 / 0.488819609 | 0.001321554 | 0.000238213 | 0.698817406 | 1 |
| CD_s2 | 0.550606048 / 0.563998479 / 0.523724415 | 0.546213836 / 0.574323150 / 0.492193680 | 0.545738207 / 0.573845561 / 0.491814470 | 0.000821013 | 0.000642392 | 0.700777044 | 2 |
| BS_s0 | 0.549718003 / 0.563426425 / 0.522372690 | 0.545734118 / 0.573960455 / 0.491638515 | 0.545225470 / 0.573443141 / 0.491260963 | 0.000627712 | -0.004419960 | 0.700448043 | 1 |
| BS_s1 | 0.548582685 / 0.566627332 / 0.513946463 | 0.544537189 / 0.576767667 / 0.483896769 | 0.543998738 / 0.576167223 / 0.483580189 | 0.000051800 | -0.008526572 | 0.691952530 | 1 |
| BS_s2 | 0.550146960 / 0.566664584 / 0.518157708 | 0.545978142 / 0.576783449 / 0.487781869 | 0.545475219 / 0.576262193 / 0.487441294 | 0.001665984 | -0.009158693 | 0.696291760 | 1 |

## Three-seed mean and sample standard deviation

Observed standard deviations use ddof=1.

| Group | Quantity | Domain | Mean | Sample std |
| --- | --- | --- | --- | --- |
| CD | Val_rep_L | Overall | 0.547493098 | 0.002323097 |
| CD | Val_rep_L | Urban | 0.575994451 | 0.001628772 |
| CD | Val_rep_L | Rural | 0.493189578 | 0.004608490 |
| CD | Val_rep_LB | Overall | 0.546993958 | 0.002239899 |
| CD | Val_rep_LB | Urban | 0.575504349 | 0.001582759 |
| CD | Val_rep_LB | Rural | 0.492779950 | 0.004521071 |
| CD | Val_sel_L | Overall | 0.551380220 | 0.001811262 |
| CD | climb | Overall | 0.002126578 | 0.001844878 |
| CD | K8_minus_K4 | Overall | -0.000052100 | 0.000876481 |
| CD | K4_cldice | Overall | 0.702350481 | 0.004529613 |
| BS | Val_rep_L | Overall | 0.545416483 | 0.000771204 |
| BS | Val_rep_L | Urban | 0.575837191 | 0.001625320 |
| BS | Val_rep_L | Rural | 0.487772384 | 0.003870882 |
| BS | Val_rep_LB | Overall | 0.544899809 | 0.000790279 |
| BS | Val_rep_LB | Urban | 0.575290852 | 0.001600869 |
| BS | Val_rep_LB | Rural | 0.487427482 | 0.003840406 |
| BS | Val_sel_L | Overall | 0.549482549 | 0.000808281 |
| BS | climb | Overall | 0.000781832 | 0.000818054 |
| BS | K8_minus_K4 | Overall | -0.007368408 | 0.002572917 |
| BS | K4_cldice | Overall | 0.696230778 | 0.004248085 |
| RS_D4 | Val_rep_L | Overall | 0.545751875 | 0.001691810 |
| RS_D4 | Val_rep_L | Urban | 0.576838454 | 0.002449245 |
| RS_D4 | Val_rep_L | Rural | 0.487209759 | 0.002778749 |
| RS_D4 | Val_rep_LB | Overall | 0.545215200 | 0.001717469 |
| RS_D4 | Val_rep_LB | Urban | 0.576255889 | 0.002450596 |
| RS_D4 | Val_rep_LB | Rural | 0.486867551 | 0.002784524 |
| RS_D4 | Val_sel_L | Overall | 0.549529465 | 0.002197813 |
| RS_D4 | climb | Overall | 0.000273423 | 0.001878512 |
| RS_D4 | K8_minus_K4 | Overall | -0.007670053 | 0.005456991 |
| RS_D4 | K4_cldice | Overall | 0.697518971 | 0.002263443 |

| Group | Val-sel domain | Mean | Sample std |
| --- | --- | --- | --- |
| CD | Overall | 0.551380220 | 0.001811262 |
| CD | Urban | 0.565540621 | 0.001599714 |
| CD | Rural | 0.523402196 | 0.003364926 |
| BS | Overall | 0.549482549 | 0.000808281 |
| BS | Urban | 0.565572781 | 0.001858892 |
| BS | Rural | 0.518158954 | 0.004213114 |

## Paired differences against RS-D4 with the same seed

| Run | Delta L (O/U/R) | Delta LB (O/U/R) | Delta climb | Delta K8-K4 | Delta K4 clDice |
| --- | --- | --- | --- | --- | --- |
| CD_s0 | 0.004255782 / 0.002257950 / 0.008142155 | 0.004192857 / 0.002244964 / 0.008000706 | 0.005061617 | 0.001497552 | 0.007961154 |
| CD_s1 | 0.002108053 / 0.000550811 / 0.004637130 | 0.002245432 / 0.000738591 / 0.004680378 | 0.002119320 | 0.013638023 | 0.003767468 |
| CD_s2 | -0.001140166 / -0.005340769 / 0.005160172 | -0.001102015 / -0.005238176 / 0.005056116 | -0.001621475 | 0.007718285 | 0.002765907 |
| BS_s0 | -0.000184739 / -0.001358664 / 0.001566077 | -0.000161686 / -0.001310049 / 0.001555897 | 0.001452163 | -0.001885505 | 0.000952205 |
| BS_s1 | 0.000554424 / 0.001235344 / -0.000626563 | 0.000580516 / 0.001236482 / -0.000559042 | 0.000849567 | 0.004873237 | -0.003097409 |
| BS_s2 | -0.001375860 / -0.002880469 / 0.000748361 | -0.001365003 / -0.002821544 / 0.000682940 | -0.000776504 | -0.002082799 | -0.001719377 |

## K4 component changes

Observed component entries are three-seed means of same-seed differences against RS-D4.

| Group | Domain | Delta IoU | Delta clDice | Delta BIoU |
| --- | --- | --- | --- | --- |
| CD | Overall | 0.001333185 | 0.004831510 | 0.001352232 |
| CD | Urban | -0.000985392 | -0.002691330 | -0.000472572 |
| CD | Rural | 0.006933204 | 0.014508867 | 0.003403324 |
| BS | Overall | 0.000351365 | -0.001288194 | 0.000296379 |
| BS | Urban | -0.000147250 | -0.002458801 | 0.000153995 |
| BS | Rural | 0.001118491 | -0.000100980 | 0.000493332 |

## Selected K curves

Observed seeds are chosen only by Val-sel L: CD*=CD_s0, BS*=BS_s2; archived RS-D4*=seed 2.

| Configuration | Domain | A(K1) | A(K2) | A(K4) | A(K8) |
| --- | --- | --- | --- | --- | --- |
| E0 | Overall | 0.545533261 | 0.546859589 | 0.547761448 | 0.545221979 |
| E0 | Urban | 0.573773721 | 0.576770231 | 0.580642531 | 0.582903293 |
| E0 | Rural | 0.491153714 | 0.489992607 | 0.486556779 | 0.477642432 |
| RS_D4_star | Overall | 0.545605338 | 0.547021197 | 0.548047826 | 0.540971932 |
| RS_D4_star | Urban | 0.573852596 | 0.577167581 | 0.582433139 | 0.582279543 |
| RS_D4_star | Rural | 0.491204031 | 0.489769834 | 0.484731226 | 0.468875616 |
| CD_star | Overall | 0.547227631 | 0.549251173 | 0.551464797 | 0.550427894 |
| CD_star | Urban | 0.574663796 | 0.576875166 | 0.578782129 | 0.575089423 |
| CD_star | Rural | 0.494367268 | 0.496458385 | 0.500082436 | 0.504959542 |
| BS_star | Overall | 0.544800158 | 0.545692126 | 0.546466142 | 0.537307450 |
| BS_star | Urban | 0.572150854 | 0.574343916 | 0.579140826 | 0.578590086 |
| BS_star | Rural | 0.491770250 | 0.490637907 | 0.485500396 | 0.464840919 |

Observed CD* Overall rises from K1 to K4 then falls by 0.001036903 at K8. This is 0.000036903 below the preregistered tail tolerance; no threshold is changed. Neither CD nor BS has any seed passing both CB-R2 conditions.

Observed E0: Overall has a nonmonotonic shape (K4-K1=0.002228187, K8-K4=-0.002539469); Urban rises throughout (K4-K1=0.006868810, K8-K4=0.002260762); Rural falls throughout (K4-K1=-0.004596935, K8-K4=-0.008914347).
Observed RS_D4_star: Overall has a nonmonotonic shape (K4-K1=0.002442488, K8-K4=-0.007075894); Urban has a nonmonotonic shape (K4-K1=0.008580543, K8-K4=-0.000153596); Rural falls throughout (K4-K1=-0.006472805, K8-K4=-0.015855610).
Observed CD_star: Overall has a nonmonotonic shape (K4-K1=0.004237166, K8-K4=-0.001036903); Urban has a nonmonotonic shape (K4-K1=0.004118332, K8-K4=-0.003692706); Rural rises throughout (K4-K1=0.005715168, K8-K4=0.004877106).
Observed BS_star: Overall has a nonmonotonic shape (K4-K1=0.001665984, K8-K4=-0.009158693); Urban has a nonmonotonic shape (K4-K1=0.006989972, K8-K4=-0.000550740); Rural falls throughout (K4-K1=-0.006269854, K8-K4=-0.020659477).

## Val-sel L by epoch and domain

| Run | Epoch | Overall L | Urban L | Rural L |
| --- | --- | --- | --- | --- |
| CD_s0 | 1 | 0.553449912 | 0.567192279 | 0.526594422 |
| CD_s0 | 2 | 0.538500330 | 0.543906523 | 0.524642036 |
| CD_s0 | 3 | 0.546961922 | 0.563917870 | 0.514088400 |
| CD_s0 | 4 | 0.546736129 | 0.563525859 | 0.514541552 |
| CD_s0 | 5 | 0.532090730 | 0.543203575 | 0.508611091 |
| CD_s0 | 6 | 0.539417005 | 0.558046894 | 0.503256919 |
| CD_s0 | 7 | 0.534823318 | 0.552158957 | 0.500799333 |
| CD_s0 | 8 | 0.530680967 | 0.548954493 | 0.494877107 |
| CD_s0 | 9 | 0.527772908 | 0.544096486 | 0.495274570 |
| CD_s0 | 10 | 0.525646509 | 0.541113432 | 0.494641890 |
| CD_s1 | 1 | 0.550084700 | 0.565431104 | 0.519887751 |
| CD_s1 | 2 | 0.547839580 | 0.572273215 | 0.503442742 |
| CD_s1 | 3 | 0.542704546 | 0.557098202 | 0.513966810 |
| CD_s1 | 4 | 0.541990091 | 0.556529208 | 0.512868381 |
| CD_s1 | 5 | 0.537239268 | 0.558457709 | 0.496565796 |
| CD_s1 | 6 | 0.537968111 | 0.558795835 | 0.498079791 |
| CD_s1 | 7 | 0.535481635 | 0.557431446 | 0.493187838 |
| CD_s1 | 8 | 0.531858890 | 0.551615807 | 0.493268622 |
| CD_s1 | 9 | 0.530004220 | 0.548579317 | 0.493389859 |
| CD_s1 | 10 | 0.529607324 | 0.549099496 | 0.491315942 |
| CD_s2 | 1 | 0.549214379 | 0.560064478 | 0.527363733 |
| CD_s2 | 2 | 0.550606048 | 0.563998479 | 0.523724415 |
| CD_s2 | 3 | 0.547823659 | 0.565435875 | 0.514047067 |
| CD_s2 | 4 | 0.535169954 | 0.544722280 | 0.514668855 |
| CD_s2 | 5 | 0.541816420 | 0.555944358 | 0.513486876 |
| CD_s2 | 6 | 0.541165476 | 0.557664152 | 0.508682284 |
| CD_s2 | 7 | 0.533013703 | 0.547376675 | 0.503896697 |
| CD_s2 | 8 | 0.528581328 | 0.546261944 | 0.493945332 |
| CD_s2 | 9 | 0.528011076 | 0.544763062 | 0.494732522 |
| CD_s2 | 10 | 0.526612254 | 0.542308335 | 0.495158761 |
| BS_s0 | 1 | 0.549718003 | 0.563426425 | 0.522372690 |
| BS_s0 | 2 | 0.547472596 | 0.558644012 | 0.523942790 |
| BS_s0 | 3 | 0.545089447 | 0.560906392 | 0.514216779 |
| BS_s0 | 4 | 0.545935136 | 0.564106941 | 0.510784483 |
| BS_s0 | 5 | 0.540764937 | 0.552441134 | 0.516583211 |
| BS_s0 | 6 | 0.543568609 | 0.560613223 | 0.510106167 |
| BS_s0 | 7 | 0.539306821 | 0.555942096 | 0.506532135 |
| BS_s0 | 8 | 0.535235777 | 0.552981871 | 0.500363499 |
| BS_s0 | 9 | 0.533422249 | 0.549947605 | 0.500681482 |
| BS_s0 | 10 | 0.532985348 | 0.549258202 | 0.500670800 |
| BS_s1 | 1 | 0.548582685 | 0.566627332 | 0.513946463 |
| BS_s1 | 2 | 0.541024130 | 0.563666451 | 0.499138894 |
| BS_s1 | 3 | 0.542162027 | 0.558526590 | 0.510032550 |
| BS_s1 | 4 | 0.542085149 | 0.557968668 | 0.510572132 |
| BS_s1 | 5 | 0.535769972 | 0.555328853 | 0.497800995 |
| BS_s1 | 6 | 0.535026851 | 0.554855354 | 0.496435127 |
| BS_s1 | 7 | 0.535491733 | 0.554656358 | 0.498054171 |
| BS_s1 | 8 | 0.528424470 | 0.548464607 | 0.489300168 |
| BS_s1 | 9 | 0.531141755 | 0.548787072 | 0.496279598 |
| BS_s1 | 10 | 0.530261639 | 0.548408984 | 0.494510122 |
| BS_s2 | 1 | 0.550146960 | 0.566664584 | 0.518157708 |
| BS_s2 | 2 | 0.545823897 | 0.559969078 | 0.517632066 |
| BS_s2 | 3 | 0.544050579 | 0.561359100 | 0.510512682 |
| BS_s2 | 4 | 0.535152365 | 0.546242610 | 0.512081377 |
| BS_s2 | 5 | 0.540758180 | 0.552268868 | 0.516715771 |
| BS_s2 | 6 | 0.539530258 | 0.554939038 | 0.508831927 |
| BS_s2 | 7 | 0.536415310 | 0.551050181 | 0.506804307 |
| BS_s2 | 8 | 0.530615541 | 0.549976425 | 0.492628136 |
| BS_s2 | 9 | 0.531243235 | 0.548974540 | 0.496081035 |
| BS_s2 | 10 | 0.530702573 | 0.547537469 | 0.497118224 |

| Run | Best Overall epoch | Best Urban epoch | Best Rural epoch |
| --- | --- | --- | --- |
| CD_s0 | 1 | 1 | 1 |
| CD_s1 | 1 | 2 | 1 |
| CD_s2 | 2 | 3 | 1 |
| BS_s0 | 1 | 4 | 2 |
| BS_s1 | 1 | 1 | 1 |
| BS_s2 | 1 | 1 | 1 |

Observed the component and domain curves above are descriptive; adoption follows only CB-R1. The full component scores and arithmetic inputs are retained in metrics.json. No scientific prediction is amended.


Observed CD* Rural A rises at every recorded K, while its Urban A rises through K4 and then falls at K8. Observed E0, RS-D4* and BS* Rural A falls throughout; their Urban A rises through K4, with only E0 still rising at K8.

Observed the three-seed CD K4 difference has its largest raw component change in clDice (+0.004831510 Overall and +0.014508867 Rural); Urban IoU/clDice/BIoU all decrease on average. Observed BS Overall clDice decreases by 0.001288194 while IoU and BIoU rise by 0.000351365 and 0.000296379. These are component differences, not a causal decomposition.

Observed the common SHA is saved by each final evaluation report. Training configurations all record the same cache path; they do not save separate startup fingerprints.
