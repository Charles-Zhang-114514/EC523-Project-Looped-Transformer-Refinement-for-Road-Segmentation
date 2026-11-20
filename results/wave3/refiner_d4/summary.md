# Wave 3 refiner D4 acceptance

| Gate | Status | Evidence |
| --- | --- | --- |
| Reference SG1 | PASS | Observed 48 E0 archive comparisons, maximum difference 0. |
| Stress membership | PASS | Observed all six reports contain the same 835 stress IDs matching Val-rep. |
| Configuration/cache | PASS | Observed six expected configurations and identical manifest SHA 06d97973fcec14cc113b7c3468d768e1e0a6d9b3d27998fd6aa14cf07c13dc90; existing cache gates pass. |

| Prediction / rule | Status | Evidence |
| --- | --- | --- |
| R1-P1 | Not met | Observed RS-D4 mean gain=0.002029875102 in [0.001,0.004], but best epoch >=3 count is 0/3, below 2/3. |
| R1-P2 | Not met | Observed RW-D4 mean gain=-0.000367864844, outside [0,0.002]. |
| R1-P3 | Not met | Observed selected-seed gains: RS-D4* +0.000218563790; RW-D4* -0.000683598026, neither >=0.001. |
| R1-R1 | Neither group passes | Observed RS-D4* fails the Overall gain bound; RW-D4* fails Overall and Urban bounds. |
| R1-R2 | Met | Observed RS-D4 mean gain=0.002029875102 >=0.001 and paired positive seeds=3/3 >=2/3. |

Observed R1-R1 retains E0 (e6 seed 0); R1-R2 specifies D4 for subsequent refiner training directions 2/3/4.

| Job | Run | GPU / mode | Wall seconds | Peak allocated GiB | Best epoch | Normal end |
| --- | --- | --- | --- | --- | --- | --- |
| 8804723 | RS_D4_s0 | L40S / Exclusive_Process | 3587 | 6.219810 | 1 | PASS |
| 8804724 | RS_D4_s1 | L40S / Exclusive_Process | 3572 | 6.219810 | 1 | PASS |
| 8804725 | RS_D4_s2 | L40S / Exclusive_Process | 3422 | 6.219810 | 2 | PASS |
| 8804726 | RW_D4_s0 | L40S / Exclusive_Process | 1503 | 6.219810 | 1 | PASS |
| 8804727 | RW_D4_s1 | L40S / Exclusive_Process | 1498 | 6.219810 | 1 | PASS |
| 8804728 | RW_D4_s2 | L40S / Exclusive_Process | 1507 | 6.219810 | 1 | PASS |

## Configuration verification

| Arm | LR | Warmup | Epochs | Batch | Initialization | Selection |
| --- | --- | --- | --- | --- | --- | --- |
| RS-D4 | 0.0003 | 500 | 10 | 4 | Random | 834 fixed Val-sel images |
| RW-D4 | 0.0001 | 100 | 3 | 4 | e6 seed 0 | 834 fixed Val-sel images |

Observed d4_aug=true, BCE pos_weight=4, K_train=4, AdamW wd=0.01, clip=1, assigned seeds and L weights=(0.2,0.2,0.6) in all six runs. Observed complete epochs, finite batch losses, both best.pt/last.pt, correct split membership and identical reporting fingerprints. The initial-head SHA matches A1-i s0. All accounting records have failed=0/exit_status=0 and durations below h_rt; logs have completion markers and no Traceback/OOM/killed/timeout.

## Per-seed scores

| Run | Best epoch | Val-sel Overall L | Domain | Val-rep L | Val-rep LB |
| --- | --- | --- | --- | --- | --- |
| RS_D4_s0 | 1 | 0.549930937560 | Overall | 0.545918857375 | 0.545387155376 |
| RS_D4_s0 | 1 | 0.549930937560 | Urban | 0.575319119180 | 0.574753189733 |
| RS_D4_s0 | 1 | 0.549930937560 | Rural | 0.490072437850 | 0.489705066358 |
| RS_D4_s1 | 1 | 0.547158591336 | Overall | 0.543982765608 | 0.543418221191 |
| RS_D4_s1 | 1 | 0.547158591336 | Urban | 0.575532323556 | 0.574930741216 |
| RS_D4_s1 | 1 | 0.547158591336 | Rural | 0.484523331433 | 0.484139231236 |
| RS_D4_s2 | 2 | 0.551498866531 | Overall | 0.547354002322 | 0.546840222123 |
| RS_D4_s2 | 2 | 0.551498866531 | Urban | 0.579663918768 | 0.579083736966 |
| RS_D4_s2 | 2 | 0.551498866531 | Rural | 0.487033508504 | 0.486758354071 |
| RW_D4_s0 | 1 | 0.549159833916 | Overall | 0.546451840506 | 0.545953570326 |
| RW_D4_s0 | 1 | 0.549159833916 | Urban | 0.573791894197 | 0.573266547498 |
| RW_D4_s0 | 1 | 0.549159833916 | Rural | 0.493932328308 | 0.493576610310 |
| RW_D4_s1 | 1 | 0.548917869707 | Overall | 0.545519999600 | 0.545023062540 |
| RW_D4_s1 | 1 | 0.548917869707 | Urban | 0.576070605127 | 0.575488302849 |
| RW_D4_s1 | 1 | 0.548917869707 | Rural | 0.487637433342 | 0.487382549117 |
| RW_D4_s2 | 1 | 0.545666496936 | Overall | 0.544573565364 | 0.544073605461 |
| RW_D4_s2 | 1 | 0.545666496936 | Urban | 0.568235250437 | 0.567749828371 |
| RW_D4_s2 | 1 | 0.545666496936 | Rural | 0.498104853864 | 0.497679994679 |

Observed per-domain Val-sel L is not recorded; no inference is rerun to fill it. Val-rep L/LB domain values are available above.

## Three-seed means and sample standard deviations

| Arm | Metric | Domain | Mean | Sample SD |
| --- | --- | --- | --- | --- |
| RS | Val_rep_L | Overall | 0.545751875102 | 0.001691810142 |
| RS | Val_rep_L | Urban | 0.576838453834 | 0.002449245413 |
| RS | Val_rep_L | Rural | 0.487209759262 | 0.002778748595 |
| RS | Val_rep_LB | Overall | 0.545215199564 | 0.001717468834 |
| RS | Val_rep_LB | Urban | 0.576255889305 | 0.002450596443 |
| RS | Val_rep_LB | Rural | 0.486867550555 | 0.002784523847 |
| RW | Val_rep_L | Overall | 0.545515135156 | 0.000939147020 |
| RW | Val_rep_L | Urban | 0.572699249920 | 0.004030334902 |
| RW | Val_rep_L | Rural | 0.493224871838 | 0.005269449126 |
| RW | Val_rep_LB | Overall | 0.545016746109 | 0.000939998349 |
| RW | Val_rep_LB | Urban | 0.572168226240 | 0.003984435845 |
| RW | Val_rep_LB | Rural | 0.492879718035 | 0.005183974383 |
| RS | Val_sel_L | Overall | 0.549529465142 | 0.002197813065 |
| RW | Val_sel_L | Overall | 0.547914733519 | 0.001950785091 |

Observed mean=sum(seed values)/3; sample SD=sqrt(sum((seed-mean)^2)/2). Predictions use registered rounded old means RS=0.543722 and RW=0.545883. Exact archived means are retained separately for sensitivity and seed pairing; the verdicts agree.

## Seed-paired D4 minus old runs

| Run | Old L | D4 L | Paired delta | Old best epoch | D4 best epoch |
| --- | --- | --- | --- | --- | --- |
| RS_D4_s0 | 0.544581149641 | 0.545918857375 | 0.001337707734 | 2 | 1 |
| RS_D4_s1 | 0.540456568216 | 0.543982765608 | 0.003526197393 | 1 | 1 |
| RS_D4_s2 | 0.546127594714 | 0.547354002322 | 0.001226407608 | 1 | 2 |
| RW_D4_s0 | 0.546661282903 | 0.546451840506 | -0.000209442397 | 1 | 1 |
| RW_D4_s1 | 0.546231042674 | 0.545519999600 | -0.000711043074 | 1 | 1 |
| RW_D4_s2 | 0.544756653986 | 0.544573565364 | -0.000183088622 | 1 | 1 |

| Arm | Mean gain vs registered baseline | Mean gain vs exact archive | Positive pairs | Val-sel selected run |
| --- | --- | --- | --- | --- |
| RS | 0.002029875102 | 0.002030104245 | 3/3 | RS_D4_s2 |
| RW | -0.000367864844 | -0.000367858031 | 0/3 | RW_D4_s0 |

## Selected-candidate R1-R1 checks

| Candidate | Domain | Report L | Delta vs E0 |
| --- | --- | --- | --- |
| RS_D4_s2 | Overall | 0.547354002322 | 0.000218563790 |
| RS_D4_s2 | Urban | 0.579663918768 | 0.001169609641 |
| RS_D4_s2 | Rural | 0.487033508504 | -0.001129823149 |
| RW_D4_s0 | Overall | 0.546451840506 | -0.000683598026 |
| RW_D4_s0 | Urban | 0.573791894197 | -0.004702414930 |
| RW_D4_s0 | Rural | 0.493932328308 | 0.005768996655 |

Observed adoption requires Overall delta >=0.001 and each domain delta >=-0.003. Observed RS-D4* seed 2 has Overall +0.000218563790 and passes both domain bounds; RW-D4* seed 0 has Overall -0.000683598026 and Urban -0.004702414930.

## Selected-run K curves and components

| Run | Domain | K | IoU | clDice | BIoU | A |
| --- | --- | --- | --- | --- | --- | --- |
| RS_D4_s2 | Overall | 1 | 0.632710265663 | 0.700823714702 | 0.095015453990 | 0.545605338040 |
| RS_D4_s2 | Overall | 2 | 0.635605305110 | 0.700055209256 | 0.096009907353 | 0.547021196803 |
| RS_D4_s2 | Overall | 4 | 0.638555378518 | 0.698011137096 | 0.096833976007 | 0.548047825589 |
| RS_D4_s2 | Overall | 8 | 0.630134130574 | 0.689833943714 | 0.094773416673 | 0.540971931736 |
| RS_D4_s2 | Urban | 1 | 0.676914188562 | 0.726358108682 | 0.087440345389 | 0.573852595964 |
| RS_D4_s2 | Urban | 2 | 0.681670938853 | 0.728542609660 | 0.088846642288 | 0.577167580782 |
| RS_D4_s2 | Urban | 4 | 0.688129946333 | 0.734194690077 | 0.090548794209 | 0.582433139031 |
| RS_D4_s2 | Urban | 8 | 0.683428355078 | 0.741235003862 | 0.090974323097 | 0.582279543317 |
| RS_D4_s2 | Rural | 1 | 0.539441174972 | 0.668870106210 | 0.104112056203 | 0.491204030590 |
| RS_D4_s2 | Rural | 2 | 0.538974176215 | 0.664551198303 | 0.104586930676 | 0.489769833734 |
| RS_D4_s2 | Rural | 4 | 0.535783083520 | 0.653251265368 | 0.104321523477 | 0.484731226066 |
| RS_D4_s2 | Rural | 8 | 0.521638743493 | 0.627347021018 | 0.099260690054 | 0.468875616062 |
| RW_D4_s0 | Overall | 1 | 0.633714120559 | 0.701753152816 | 0.096440116389 | 0.546671029402 |
| RW_D4_s0 | Overall | 2 | 0.632671994064 | 0.703191471082 | 0.096805283662 | 0.546654495089 |
| RW_D4_s0 | Overall | 4 | 0.630873196248 | 0.705496389129 | 0.096128555749 | 0.546311226012 |
| RW_D4_s0 | Overall | 8 | 0.627465005304 | 0.709147733269 | 0.095564845385 | 0.545589791710 |
| RW_D4_s0 | Urban | 1 | 0.677794115623 | 0.726718812915 | 0.088218467220 | 0.574556395130 |
| RW_D4_s0 | Urban | 2 | 0.676012346047 | 0.728194624427 | 0.088822461834 | 0.574229052718 |
| RW_D4_s0 | Urban | 4 | 0.674099494910 | 0.728882894167 | 0.088383626704 | 0.573391341046 |
| RW_D4_s0 | Urban | 8 | 0.670387750889 | 0.729293005282 | 0.086949507941 | 0.571371678617 |
| RW_D4_s0 | Rural | 1 | 0.540759065976 | 0.671097146477 | 0.106307494565 | 0.492970175844 |
| RW_D4_s0 | Rural | 2 | 0.541067835706 | 0.673015053296 | 0.106360974060 | 0.493710628654 |
| RW_D4_s0 | Rural | 4 | 0.539781958382 | 0.677897793262 | 0.105333142555 | 0.494326945681 |
| RW_D4_s0 | Rural | 8 | 0.538085731339 | 0.686031014924 | 0.105663845494 | 0.495984939245 |

| Run | Domain | K1 to K2 | K2 to K4 | K4 to K8 |
| --- | --- | --- | --- | --- |
| RS_D4_s2 | Overall | up | up | down |
| RS_D4_s2 | Urban | up | up | down |
| RS_D4_s2 | Rural | down | down | down |
| RW_D4_s0 | Overall | down | down | down |
| RW_D4_s0 | Urban | down | down | down |
| RW_D4_s0 | Rural | up | up | up |

## Every epoch Val-sel L

| Run | Epoch | L |
| --- | --- | --- |
| RS_D4_s0 | 1 | 0.549930937560 |
| RS_D4_s0 | 2 | 0.548247484704 |
| RS_D4_s0 | 3 | 0.545055840986 |
| RS_D4_s0 | 4 | 0.543125384432 |
| RS_D4_s0 | 5 | 0.537233978593 |
| RS_D4_s0 | 6 | 0.541464429764 |
| RS_D4_s0 | 7 | 0.536717621620 |
| RS_D4_s0 | 8 | 0.531484564976 |
| RS_D4_s0 | 9 | 0.530735083069 |
| RS_D4_s0 | 10 | 0.530910941039 |
| RS_D4_s1 | 1 | 0.547158591336 |
| RS_D4_s1 | 2 | 0.539360305376 |
| RS_D4_s1 | 3 | 0.541815474810 |
| RS_D4_s1 | 4 | 0.541419008441 |
| RS_D4_s1 | 5 | 0.537218836137 |
| RS_D4_s1 | 6 | 0.534060942745 |
| RS_D4_s1 | 7 | 0.535391366355 |
| RS_D4_s1 | 8 | 0.531300447499 |
| RS_D4_s1 | 9 | 0.531196359581 |
| RS_D4_s1 | 10 | 0.530509220936 |
| RS_D4_s2 | 1 | 0.550422908265 |
| RS_D4_s2 | 2 | 0.551498866531 |
| RS_D4_s2 | 3 | 0.543812166366 |
| RS_D4_s2 | 4 | 0.544786980814 |
| RS_D4_s2 | 5 | 0.543564621877 |
| RS_D4_s2 | 6 | 0.543495467989 |
| RS_D4_s2 | 7 | 0.542178007516 |
| RS_D4_s2 | 8 | 0.538537267558 |
| RS_D4_s2 | 9 | 0.539671137027 |
| RS_D4_s2 | 10 | 0.539326807379 |
| RW_D4_s0 | 1 | 0.549159833916 |
| RW_D4_s0 | 2 | 0.547008137893 |
| RW_D4_s0 | 3 | 0.545751110594 |
| RW_D4_s1 | 1 | 0.548917869707 |
| RW_D4_s1 | 2 | 0.542391120338 |
| RW_D4_s1 | 3 | 0.543998483749 |
| RW_D4_s2 | 1 | 0.545666496936 |
| RW_D4_s2 | 2 | 0.545534711741 |
| RW_D4_s2 | 3 | 0.545567940835 |

## Stress by corruption type

Not recorded: per-image Counts are absent; only aggregate Counts and per-image timing/core metadata were saved, so shadow/occlusion/blur scores cannot be reconstructed without new inference.

Observed no selected-refiner plus S5 combination is evaluated here. The condition requiring a later combination is not triggered because both adoption rules fail. Observed subsequent training uses D4 under R1-R2 despite retaining the current checkpoint.
