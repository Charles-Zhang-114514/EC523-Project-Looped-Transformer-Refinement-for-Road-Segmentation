# Wave 3 S5 acceptance

| Gate | Status | Evidence |
| --- | --- | --- |
| SG1 | PASS | Observed 48 components against fe10d7b E0; max absolute difference 0, tolerance 1e-5. |
| SG2 | PASS | Observed 15,022 stored cumulative hook-count checks over 5,008 view-batched trajectories; zero mismatches. Actual core invocations total 33,384, summing maximum K once per trajectory. |
| SG3 | PASS | Observed 835 unique seed523 stress IDs exactly match Val-rep; one shared image stream contains all three modes. |

| Prediction / rule | Status | Evidence |
| --- | --- | --- |
| S5-P1 | Not met | Observed S5a minus E0 report L=0.000301744198, outside [0.002,0.006]. |
| S5-P2 | Met | Observed absolute report L difference=0.000003066164 < 0.002. |
| S5-P3 | Time qualitative; memory bound met | Observed S5a/E0 refiner time ratios K4=7.532020, K8=7.919207; peak=3.105933 GiB <=8. No numeric tolerance was registered for approximately eightfold. |
| S5-R1 | Not met | Observed Val-sel selects S5b; report LB gain=0.000303237861 <0.002; domain L bounds pass. |

Observed S5-R1 retains E0 and does not adopt S5.

| Job | Run | GPU / mode | Wall seconds | Peak allocated GiB | Best epoch | Normal end |
| --- | --- | --- | --- | --- | --- | --- |
| 8804722 | s5_eval | L40S / Exclusive_Process | 2015 | 3.105933 | None | PASS |

## Gate details

| Split | Images | Cumulative endpoint checks | Actual core invocations |
| --- | --- | --- | --- |
| Val-sel | 834 | 6672 | 13344 |
| Val-rep | 835 | 6680 | 13360 |
| stress | 835 | 1670 | 6680 |

Observed each S5a/S5b image has K1/2/4/8 counts on clean splits and K4 on stress. Endpoint counts are cumulative and are not summed as independent invocations. Observed SG1 is independently recomputed from archived official scores without model inference.

## L and LB by domain

| Mode | Domain | Val-sel L | Val-rep L | Val-rep LB | Stress A(K4) |
| --- | --- | --- | --- | --- | --- |
| E0 | Overall | 0.551907986584 | 0.547135438532 | 0.546635973274 | 0.545264121396 |
| E0 | Urban | 0.569210639022 | 0.578494309127 | 0.577975655777 | 0.578049264235 |
| E0 | Rural | 0.518669554494 | 0.488163331653 | 0.487821525487 | 0.484847748092 |
| S5a | Overall | 0.552063124829 | 0.547437182730 | 0.546936299447 | 0.545683463384 |
| S5a | Urban | 0.569134772703 | 0.578923142912 | 0.578401736948 | 0.578647994985 |
| S5a | Rural | 0.519190783026 | 0.488254177139 | 0.487912824886 | 0.484978503731 |
| S5b | Overall | 0.552064101030 | 0.547440248894 | 0.546939211136 | 0.545687602162 |
| S5b | Urban | 0.569138151341 | 0.578927830116 | 0.578407378646 | 0.578660421111 |
| S5b | Rural | 0.519188082763 | 0.488254040898 | 0.487911977336 | 0.484974532774 |

Observed L=0.2*A1+0.2*A2+0.6*A4 and LB=0.2*A1+0.2*A2+0.4*A4+0.2*A_stress4, with each A formed from saved official aggregate Counts. Observed Val-sel S5b L=0.552064101030 exceeds S5a L=0.552063124829; the report data are not used to select the mode.

| Selected mode | Domain | Report L minus E0 |
| --- | --- | --- |
| S5b | Urban | 0.000433520989 |
| S5b | Rural | 0.000090709245 |

## Resource observations

Mean cumulative refiner time per image at K; excludes view encoders/TTA draft construction. Peak allocated CUDA bytes include resident models and shared view features; ratios are within this evaluation, not isolated incremental memory.

| Split | Mode | K | Mean refiner seconds/image | Time / E0 | Peak GiB | Peak / E0 |
| --- | --- | --- | --- | --- | --- | --- |
| Val-sel | E0 | 1 | 0.003198935841 | 1.000000 | 1.562947 | 1.000000 |
| Val-sel | E0 | 2 | 0.006410527105 | 1.000000 | 1.566853 | 1.000000 |
| Val-sel | E0 | 4 | 0.011945229597 | 1.000000 | 1.566853 | 1.000000 |
| Val-sel | E0 | 8 | 0.022283484112 | 1.000000 | 1.566853 | 1.000000 |
| Val-rep | E0 | 1 | 0.003210627256 | 1.000000 | 1.562947 | 1.000000 |
| Val-rep | E0 | 2 | 0.006402708324 | 1.000000 | 1.566853 | 1.000000 |
| Val-rep | E0 | 4 | 0.011962092636 | 1.000000 | 1.566853 | 1.000000 |
| Val-rep | E0 | 8 | 0.022351687597 | 1.000000 | 1.566853 | 1.000000 |
| stress | E0 | 4 | 0.010530286859 | 1.000000 | 1.566853 | 1.000000 |
| Val-sel | S5a | 1 | 0.024121022090 | 7.540327 | 3.039527 | 1.944741 |
| Val-sel | S5a | 2 | 0.046348018494 | 7.229986 | 3.105933 | 1.982275 |
| Val-sel | S5a | 4 | 0.090053572484 | 7.538873 | 3.105933 | 1.982275 |
| Val-sel | S5a | 8 | 0.176979291051 | 7.942173 | 3.105933 | 1.982275 |
| Val-rep | S5a | 1 | 0.024117040116 | 7.511629 | 3.039527 | 1.944741 |
| Val-rep | S5a | 2 | 0.046372615836 | 7.242656 | 3.105933 | 1.982275 |
| Val-rep | S5a | 4 | 0.090098716470 | 7.532020 | 3.105933 | 1.982275 |
| Val-rep | S5a | 8 | 0.177007651052 | 7.919207 | 3.105933 | 1.982275 |
| stress | S5a | 4 | 0.090303030912 | 8.575553 | 3.105933 | 1.982275 |
| Val-sel | S5b | 1 | 0.023646544914 | 7.392003 | 3.039527 | 1.944741 |
| Val-sel | S5b | 2 | 0.045653293216 | 7.121613 | 3.070777 | 1.959837 |
| Val-sel | S5b | 4 | 0.088903923121 | 7.442630 | 3.070777 | 1.959837 |
| Val-sel | S5b | 8 | 0.175066013269 | 7.856312 | 3.070777 | 1.959837 |
| Val-rep | S5b | 1 | 0.023636867586 | 7.362072 | 3.039527 | 1.944741 |
| Val-rep | S5b | 2 | 0.045667443107 | 7.132520 | 3.070777 | 1.959837 |
| Val-rep | S5b | 4 | 0.088941783888 | 7.435303 | 3.070777 | 1.959837 |
| Val-rep | S5b | 8 | 0.175065557221 | 7.832319 | 3.070777 | 1.959837 |
| stress | S5b | 4 | 0.087827831680 | 8.340498 | 3.070777 | 1.959837 |

Observed timing excludes encoder view construction and TTA draft construction. Observed allocation peaks include resident models/shared features. The approximately-eightfold prediction remains qualitative; no new pass threshold or K convention is introduced.

## Val-rep K curves and components

| Mode | Domain | K | IoU | clDice | BIoU | A |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Overall | 1 | 0.632569959578 | 0.701023991248 | 0.094705416910 | 0.545533260545 |
| E0 | Overall | 2 | 0.635184777277 | 0.700630483143 | 0.095390277405 | 0.546859589062 |
| E0 | Overall | 4 | 0.637934026543 | 0.698818483915 | 0.095744446188 | 0.547761447684 |
| E0 | Overall | 8 | 0.635590688046 | 0.694943152334 | 0.094718446220 | 0.545221978967 |
| E0 | Urban | 1 | 0.676759656370 | 0.726542540688 | 0.087155654698 | 0.573773721331 |
| E0 | Urban | 2 | 0.680996970055 | 0.728770749714 | 0.088202607015 | 0.576770231345 |
| E0 | Urban | 4 | 0.686372174977 | 0.732170362211 | 0.089026674172 | 0.580642530986 |
| E0 | Urban | 8 | 0.686954369404 | 0.738324578687 | 0.089643675330 | 0.582903293374 |
| E0 | Rural | 1 | 0.539361291798 | 0.669063754298 | 0.103769710060 | 0.491153714200 |
| E0 | Rural | 2 | 0.539068570829 | 0.665527205627 | 0.104000800997 | 0.489992607302 |
| E0 | Rural | 4 | 0.537198763914 | 0.657337471825 | 0.103780777087 | 0.486556778922 |
| E0 | Rural | 8 | 0.530021411957 | 0.641589074349 | 0.100775018440 | 0.477642431971 |
| S5a | Overall | 1 | 0.632629577133 | 0.701233469548 | 0.094801873536 | 0.545645204138 |
| S5a | Overall | 2 | 0.635319922489 | 0.700716963564 | 0.095510099018 | 0.546977070117 |
| S5a | Overall | 4 | 0.638437632836 | 0.699104923032 | 0.096187932354 | 0.548187879798 |
| S5a | Overall | 8 | 0.636830711204 | 0.695746975144 | 0.095256696981 | 0.546190787542 |
| S5a | Urban | 1 | 0.676836104639 | 0.726809690565 | 0.087302703924 | 0.573921500274 |
| S5a | Urban | 2 | 0.681173987805 | 0.728873252396 | 0.088400851247 | 0.576929139871 |
| S5a | Urban | 4 | 0.687025289948 | 0.732678474148 | 0.089694187934 | 0.581255024805 |
| S5a | Urban | 8 | 0.688609708379 | 0.740117552470 | 0.090270509486 | 0.584394221828 |
| S5a | Rural | 1 | 0.539376050378 | 0.669181955267 | 0.103806714233 | 0.491203954616 |
| S5a | Rural | 2 | 0.539100105370 | 0.665518281581 | 0.104027994607 | 0.490011136081 |
| S5a | Rural | 4 | 0.537369871998 | 0.657364400033 | 0.103955044954 | 0.486685265000 |
| S5a | Rural | 8 | 0.530406234670 | 0.641045378233 | 0.101216730241 | 0.477760076854 |
| S5b | Overall | 1 | 0.632629577133 | 0.701233469548 | 0.094801873536 | 0.545645204138 |
| S5b | Overall | 2 | 0.635320296257 | 0.700718449063 | 0.095509923121 | 0.546977667471 |
| S5b | Overall | 4 | 0.638440462828 | 0.699115599477 | 0.096189398479 | 0.548192790953 |
| S5b | Overall | 8 | 0.636835020360 | 0.695763724126 | 0.095253986699 | 0.546197424757 |
| S5b | Urban | 1 | 0.676836104639 | 0.726809690565 | 0.087302703924 | 0.573921500274 |
| S5b | Urban | 2 | 0.681174670560 | 0.728874179071 | 0.088400129607 | 0.576929614923 |
| S5b | Urban | 4 | 0.687030598743 | 0.732694882532 | 0.089694571651 | 0.581262678461 |
| S5b | Urban | 8 | 0.688624476264 | 0.740162634904 | 0.090270697532 | 0.584415168110 |
| S5b | Rural | 1 | 0.539376050378 | 0.669181955267 | 0.103806714233 | 0.491203954616 |
| S5b | Rural | 2 | 0.539099868324 | 0.665520223619 | 0.104028484373 | 0.490011698122 |
| S5b | Rural | 4 | 0.537368265663 | 0.657363879866 | 0.103957768966 | 0.486684850584 |
| S5b | Rural | 8 | 0.530392906351 | 0.641029219372 | 0.101210261252 | 0.477747271238 |

## Observed curve transitions

| Mode | Domain | K1 to K2 | K2 to K4 | K4 to K8 |
| --- | --- | --- | --- | --- |
| E0 | Overall | up | up | down |
| E0 | Urban | up | up | up |
| E0 | Rural | down | down | down |
| S5a | Overall | up | up | down |
| S5a | Urban | up | up | up |
| S5a | Rural | down | down | down |
| S5b | Overall | up | up | down |
| S5b | Urban | up | up | up |
| S5b | Rural | down | down | down |

## Stress by corruption type

Not recorded: per-image Counts are absent; only aggregate Counts and per-image timing/core metadata were saved, so shadow/occlusion/blur scores cannot be reconstructed without new inference.

Observed any future S5 adoption remains pending TA confirmation of whether batched views satisfy the one-core-call requirement. Observed S5-R1 and R1-R1 do not both pass, so no combination evaluation is requested by the registered rule in this round.
