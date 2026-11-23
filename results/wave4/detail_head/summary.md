# Wave 4 detail-head acceptance

| Gate | Status | Evidence |
| --- | --- | --- |
| DRG1 | PASS | Observed pre-update DR-W seed0 evaluates 835 images: 48 domain/component/K comparisons, maximum difference 0, tolerance 1e-5. |
| DRG2 | PASS | Observed 25050 saved cumulative core-call checks, no mismatch; 60120 actual calls after counting maximum K once per image trajectory. |
| DRG3 | PASS | Observed all six selected best.pt gates have nonzero weight norms; 256/256 gate weights are nonzero in every checkpoint. |

| Prediction / rule | Status | Evidence |
| --- | --- | --- |
| DR-P1 | Not met | Observed DR-S mean L delta=-0.000418171462, outside [0.002,0.008]; positive pairs=1/3, below 2/3. |
| DR-P2 | Not met | Observed mean K4 BIoU delta=-0.000430925366 <0.005; IoU delta=-0.000697347669 <0. |
| DR-P3 | Met by specified mean-difference comparison | Observed Rural L delta=-0.000092980527 > Urban=-0.000697675014; both are negative, so this is smaller decline, not positive improvement. |
| DR-P4 | Not met | Observed DR-W mean L delta=0.000006592376, outside [0.001,0.006]. |
| DR-P5 | Not met | Observed selected DR-S*/DR-W* LB gains are -0.000000183609/-0.000676703868; neither reaches +0.002. |
| DR-R1 | Neither group passes | Observed both selected Overall LB values fail E0+0.002; DR-W* Urban additionally fails the -0.003 domain bound. |
| DR-R2 | Not met | Observed DR-S mean L delta=-0.000418171462 <0.001 and positive pairs=1/3. |

Observed DR-R1 retains E0; DR-R2 does not add detail heads by default for directions 3/4. The previously established D4 training decision is unchanged.

| Job | Run | GPU / mode | Wall seconds | Recorded peak allocated GiB | Best epoch | Normal end |
| --- | --- | --- | --- | --- | --- | --- |
| 8815009 | DR_S_s0 | L40S / Exclusive_Process | 3568 | 7.462248 | 1 | PASS |
| 8815010 | DR_S_s1 | L40S / Exclusive_Process | 4574 | 7.462248 | 1 | PASS |
| 8815011 | DR_S_s2 | L40S / Exclusive_Process | 3639 | 7.462248 | 1 | PASS |
| 8815012 | DR_W_s0 | L40S / Exclusive_Process | 1578 | 7.462126 | 1 | PASS |
| 8815013 | DR_W_s1 | L40S / Exclusive_Process | 1473 | 7.462248 | 1 | PASS |
| 8815014 | DR_W_s2 | L40S / Exclusive_Process | 1505 | 7.462248 | 1 | PASS |

## Configuration and provenance

| Arm | LR | Warmup | Epochs | Batch | Initialization | Selection |
| --- | --- | --- | --- | --- | --- | --- |
| DR-S | 0.0003 | 500 | 10 | 4 | Random refiner; zero new gate | 834 |
| DR-W | 0.0001 | 100 | 3 | 4 | Strict e6 s0 legacy keys; new branch initialized as designed | 834 |

Observed detail_head=true and d4_aug=true, exact assigned seeds, K_train=4, BCE pos_weight=4, AdamW wd=0.01, clipping=1.0 and L weights=(0.2,0.2,0.6). All six selection/report memberships match the fixed 834/835 split and every epoch completes with finite batch losses. All best.pt/last.pt files exist. Observed identical passed manifest SHA 06d97973fcec14cc113b7c3468d768e1e0a6d9b3d27998fd6aa14cf07c13dc90, head SHA matching A1-i s0 and all cache files predating every job.

Observed DRG1 at job 8815012 log line 849 precedes first training update at line 853. The archive referenced by the gate is byte-identical to fe10d7b's wave1/ensemble_bias metrics.json. Recorded allocation peaks are maxima from saved training/final-evaluation phase records, not independent GPU accounting maxima.

## Trained gate tensors

L2/Frobenius norm of the 16x16x1x1 gate weight at each selected best.pt, CPU double precision; no model forward performed.

| Run | Checkpoint epoch | Gate weight L2 norm | Nonzero weights | Gate bias L2 norm |
| --- | --- | --- | --- | --- |
| DR_S_s0 | 1 | 0.058635480531 | 256/256 | 0.030753293475 |
| DR_S_s1 | 1 | 0.061968998763 | 256/256 | 0.030802714968 |
| DR_S_s2 | 1 | 0.045944571314 | 256/256 | 0.018267757847 |
| DR_W_s0 | 1 | 0.056756777297 | 256/256 | 0.011217890420 |
| DR_W_s1 | 1 | 0.066673565853 | 256/256 | 0.010494950677 |
| DR_W_s2 | 1 | 0.071716776495 | 256/256 | 0.011821052947 |

Observed enabled detail paths and trained nonzero gates satisfy DRG3. No gate-zero ablation or inference was performed in this acceptance, so no isolated causal effect is claimed.

## Per-seed scores and K4 components

| Run | Val-sel Overall L | Domain | Val-rep L | Val-rep LB | K4 IoU | K4 clDice | K4 BIoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DR_S_s0 | 0.551307425640 | Overall | 0.547145922060 | 0.546635789666 | 0.634404987760 | 0.701921875418 | 0.096120748913 |
| DR_S_s0 | 0.551307425640 | Urban | 0.576924476130 | 0.576397631062 | 0.679989452220 | 0.730824158961 | 0.089167348381 |
| DR_S_s0 | 0.551307425640 | Rural | 0.490849717073 | 0.490485160046 | 0.538837092009 | 0.666981542953 | 0.104481192299 |
| DR_S_s1 | 0.546542551561 | Overall | 0.543255836073 | 0.542809334984 | 0.629195515949 | 0.697105931601 | 0.094987176291 |
| DR_S_s1 | 0.546542551561 | Urban | 0.572200128399 | 0.571720254958 | 0.673272647216 | 0.725249015203 | 0.088946472560 |
| DR_S_s1 | 0.546542551561 | Rural | 0.487723283371 | 0.487435999390 | 0.534990609922 | 0.662618461231 | 0.102303189132 |
| DR_S_s2 | 0.550376498926 | Overall | 0.545599352787 | 0.545094486646 | 0.637772431374 | 0.691820888477 | 0.096254151188 |
| DR_S_s2 | 0.550376498926 | Urban | 0.579297731933 | 0.578717033067 | 0.688752733704 | 0.731799802230 | 0.090919134237 |
| DR_S_s2 | 0.550376498926 | Rural | 0.482777335761 | 0.482524458360 | 0.531439797000 | 0.641604970663 | 0.102684165097 |
| DR_W_s0 | 0.549184807690 | Overall | 0.546462789217 | 0.545959269407 | 0.630910547175 | 0.705495889678 | 0.096139603098 |
| DR_W_s0 | 0.549184807690 | Urban | 0.573800914006 | 0.573267641221 | 0.674148401772 | 0.728867392540 | 0.088396512383 |
| DR_W_s0 | 0.549184807690 | Rural | 0.493945359696 | 0.493587266233 | 0.539800570778 | 0.677906467241 | 0.105340647361 |
| DR_W_s1 | 0.548870770234 | Overall | 0.545490428328 | 0.544990797473 | 0.632067003799 | 0.700057220039 | 0.096065126728 |
| DR_W_s1 | 0.548870770234 | Urban | 0.576157008449 | 0.575571722903 | 0.677824087885 | 0.729421072889 | 0.089655800882 |
| DR_W_s1 | 0.548870770234 | Rural | 0.487435920069 | 0.487179122487 | 0.535306791852 | 0.664277012836 | 0.103801971947 |
| DR_W_s2 | 0.545695063408 | Overall | 0.544611965053 | 0.544119317529 | 0.623349709772 | 0.709487669914 | 0.094789899971 |
| DR_W_s2 | 0.545695063408 | Urban | 0.568329034669 | 0.567851833903 | 0.662650154392 | 0.725937326184 | 0.085509822284 |
| DR_W_s2 | 0.545695063408 | Rural | 0.498057809103 | 0.497638689175 | 0.540297703847 | 0.691203910366 | 0.105730894448 |

Observed only Overall Val-sel L is saved by the training loop; no domain selection score is inferred. All report-domain scores above come from archived official aggregates.

## Three-seed means and sample standard deviations

| Arm | Metric | Domain | Mean | Sample SD |
| --- | --- | --- | --- | --- |
| S | Val_rep_L | Overall | 0.545333703640 | 0.001958601376 |
| S | Val_rep_L | Urban | 0.576140778821 | 0.003613119163 |
| S | Val_rep_L | Rural | 0.487116778735 | 0.004070223694 |
| S | Val_rep_LB | Overall | 0.544846537099 | 0.001925239749 |
| S | Val_rep_LB | Urban | 0.575611639696 | 0.003563995342 |
| S | Val_rep_LB | Rural | 0.486815205932 | 0.004016494892 |
| W | Val_rep_L | Overall | 0.545521727533 | 0.000925808971 |
| W | Val_rep_L | Urban | 0.572762319042 | 0.004016005852 |
| W | Val_rep_L | Rural | 0.493146362956 | 0.005355831262 |
| W | Val_rep_LB | Overall | 0.545023128136 | 0.000920401913 |
| W | Val_rep_LB | Urban | 0.572230399342 | 0.003963089020 |
| W | Val_rep_LB | Rural | 0.492801692631 | 0.005273848523 |
| S | Val_sel_L | Overall | 0.549408825375 | 0.002525529711 |
| W | Val_sel_L | Overall | 0.547916880444 | 0.001930546059 |

| Arm | Domain | K4 component | Mean | Sample SD |
| --- | --- | --- | --- | --- |
| S | Overall | road_iou | 0.633790978361 | 0.004321299019 |
| S | Overall | cldice | 0.696949565166 | 0.005052308595 |
| S | Overall | boundary_iou | 0.095787358797 | 0.000696181065 |
| S | Urban | road_iou | 0.680671611047 | 0.007762555953 |
| S | Urban | cldice | 0.729290992131 | 0.003534282539 |
| S | Urban | boundary_iou | 0.089677651726 | 0.001080812496 |
| S | Rural | road_iou | 0.535089166311 | 0.003699632196 |
| S | Rural | cldice | 0.657068324949 | 0.013568183919 |
| S | Rural | boundary_iou | 0.103156182176 | 0.001163195831 |
| W | Overall | road_iou | 0.628775753582 | 0.004734533932 |
| W | Overall | cldice | 0.705013593210 | 0.004733688163 |
| W | Overall | boundary_iou | 0.095664876599 | 0.000758666434 |
| W | Urban | road_iou | 0.671540881349 | 0.007915897705 |
| W | Urban | cldice | 0.728075263871 | 0.001872090716 |
| W | Urban | boundary_iou | 0.087854045183 | 0.002125555840 |
| W | Rural | road_iou | 0.538468355492 | 0.002749254228 |
| W | Rural | cldice | 0.677795796814 | 0.013463789905 |
| W | Rural | boundary_iou | 0.104957837918 | 0.001019849403 |

Observed mean=sum(seed values)/3 and sample SD uses denominator 2. Comparisons use exact archived same-seed RS-D4/RW-D4 values, not rounded baseline means.

## Same-seed differences: detail minus prior D4

| Run | Baseline | Domain | Delta L | Delta LB | Delta K4 IoU | Delta K4 clDice | Delta K4 BIoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DR_S_s0 | RS_D4_s0 | Overall | 0.001227064685 | 0.001248634290 | 0.001426841688 | 0.002426037550 | 0.000056612266 |
| DR_S_s0 | RS_D4_s0 | Urban | 0.001605356951 | 0.001644441328 | 0.002412777628 | 0.001952018821 | -0.000241668221 |
| DR_S_s0 | RS_D4_s0 | Rural | 0.000777279224 | 0.000780093688 | 0.000247207328 | 0.003145469096 | 0.000371957706 |
| DR_S_s1 | RS_D4_s1 | Overall | -0.000726929535 | -0.000608886208 | -0.002735937551 | 0.002055993220 | -0.000769563547 |
| DR_S_s1 | RS_D4_s1 | Urban | -0.003332195157 | -0.003210486258 | -0.007102470745 | -0.005096714119 | -0.001104350796 |
| DR_S_s1 | RS_D4_s1 | Rural | 0.003199951938 | 0.003296768153 | 0.005107359953 | 0.010860288120 | -0.000386866979 |
| DR_S_s2 | RS_D4_s2 | Overall | -0.001754649535 | -0.001745735477 | -0.000782947144 | -0.006190248619 | -0.000579824818 |
| DR_S_s2 | RS_D4_s2 | Urban | -0.000366186835 | -0.000366703899 | 0.000622787372 | -0.002394887847 | 0.000370340027 |
| DR_S_s2 | RS_D4_s2 | Rural | -0.004256172744 | -0.004233895710 | -0.004343286520 | -0.011646294706 | -0.001637358381 |
| DR_W_s0 | RW_D4_s0 | Overall | 0.000010948712 | 0.000005699081 | 0.000037350926 | -0.000000499450 | 0.000011047349 |
| DR_W_s0 | RW_D4_s0 | Urban | 0.000009019809 | 0.000001093722 | 0.000048906861 | -0.000015501627 | 0.000012885680 |
| DR_W_s0 | RW_D4_s0 | Rural | 0.000013031388 | 0.000010655923 | 0.000018612396 | 0.000008673978 | 0.000007504805 |
| DR_W_s1 | RW_D4_s1 | Overall | -0.000029571272 | -0.000032265068 | -0.000000572862 | -0.000074390096 | 0.000005118281 |
| DR_W_s1 | RW_D4_s1 | Urban | 0.000086403322 | 0.000083420054 | 0.000102871190 | 0.000193097927 | 0.000038365175 |
| DR_W_s1 | RW_D4_s1 | Rural | -0.000201513274 | -0.000203426631 | -0.000200424939 | -0.000388394607 | -0.000034794702 |
| DR_W_s2 | RW_D4_s2 | Overall | 0.000038399690 | 0.000045712068 | 0.000074317724 | 0.000087921840 | 0.000014321965 |
| DR_W_s2 | RW_D4_s2 | Urban | 0.000093784233 | 0.000102005532 | 0.000182838474 | 0.000178510399 | 0.000051235841 |
| DR_W_s2 | RW_D4_s2 | Rural | -0.000047044761 | -0.000041305504 | -0.000121694743 | -0.000017851358 | -0.000035569723 |

| Arm | Domain | Mean delta L | Mean delta LB | Mean delta K4 IoU | Mean delta K4 clDice | Mean delta K4 BIoU |
| --- | --- | --- | --- | --- | --- | --- |
| S | Overall | -0.000418171462 | -0.000368662465 | -0.000697347669 | -0.000569405950 | -0.000430925366 |
| S | Urban | -0.000697675014 | -0.000644249610 | -0.001355635248 | -0.001846527715 | -0.000325226330 |
| S | Rural | -0.000092980527 | -0.000052344623 | 0.000337093587 | 0.000786487504 | -0.000550755885 |
| W | Overall | 0.000006592376 | 0.000006382027 | 0.000037031929 | 0.000004344098 | 0.000010162532 |
| W | Urban | 0.000063069121 | 0.000062173103 | 0.000111538842 | 0.000118702233 | 0.000034162232 |
| W | Rural | -0.000078508882 | -0.000078025404 | -0.000101169095 | -0.000132523996 | -0.000020953206 |

## Selected candidates

| Candidate | Val-sel L | Domain | Report LB | LB minus E0 |
| --- | --- | --- | --- | --- |
| DR_S_s0 | 0.551307425640 | Overall | 0.546635789666 | -0.000000183609 |
| DR_S_s0 | 0.551307425640 | Urban | 0.576397631062 | -0.001578024715 |
| DR_S_s0 | 0.551307425640 | Rural | 0.490485160046 | 0.002663634558 |
| DR_W_s0 | 0.549184807690 | Overall | 0.545959269407 | -0.000676703868 |
| DR_W_s0 | 0.549184807690 | Urban | 0.573267641221 | -0.004708014556 |
| DR_W_s0 | 0.549184807690 | Rural | 0.493587266233 | 0.005765740745 |

Observed Val-sel selects DR-S seed0 and DR-W seed0. Both fail the exact E0 LB+0.002 and registered rounded 0.548636 thresholds. DR-S* passes both domain bounds; DR-W* Urban LB delta=-0.004708014556 fails -0.003.

## Descriptive component decomposition of L changes

Apply official linear quality weights to saved per-K component differences and L weights. This is score-difference arithmetic, not a new image metric implementation or causal ablation. Observed official quality weights are 0.5 IoU, 0.3 clDice and 0.2 BIoU ($EC523_STARTER_DIR/metrics.py:69).

| Arm | Domain | IoU contribution | clDice contribution | BIoU contribution | Sum = mean delta L |
| --- | --- | --- | --- | --- | --- |
| S | Overall | -0.000240828799 | -0.000111594076 | -0.000065748587 | -0.000418171462 |
| S | Urban | -0.000392343271 | -0.000271842938 | -0.000033488805 | -0.000697675014 |
| S | Rural | -0.000023224958 | 0.000034302982 | -0.000104058551 | -0.000092980527 |
| W | Overall | 0.000010433616 | -0.000005194006 | 0.000001352766 | 0.000006592376 |
| W | Urban | 0.000036652908 | 0.000020898645 | 0.000005517568 | 0.000063069121 |
| W | Rural | -0.000038899637 | -0.000035625256 | -0.000003983990 | -0.000078508882 |

Observed these are algebraic differences of already saved official scores across seeds/K, not recomputed metrics or causal attributions. All contribution sums match the corresponding saved mean delta L.

## K curves: E0, prior selected D4 and selected DR

| Configuration | Domain | K1 A | K2 A | K4 A | K8 A |
| --- | --- | --- | --- | --- | --- |
| E0 | Overall | 0.545533260545 | 0.546859589062 | 0.547761447684 | 0.545221978967 |
| E0 | Urban | 0.573773721331 | 0.576770231345 | 0.580642530986 | 0.582903293374 |
| E0 | Rural | 0.491153714200 | 0.489992607302 | 0.486556778922 | 0.477642431971 |
| RS_D4_s2 | Overall | 0.545605338040 | 0.547021196803 | 0.548047825589 | 0.540971931736 |
| RS_D4_s2 | Urban | 0.573852595964 | 0.577167580782 | 0.582433139031 | 0.582279543317 |
| RS_D4_s2 | Rural | 0.491204030590 | 0.489769833734 | 0.484731226066 | 0.468875616062 |
| RW_D4_s0 | Overall | 0.546671029402 | 0.546654495089 | 0.546311226012 | 0.545589791710 |
| RW_D4_s0 | Urban | 0.574556395130 | 0.574229052718 | 0.573391341046 | 0.571371678617 |
| RW_D4_s0 | Rural | 0.492970175844 | 0.493710628654 | 0.494326945681 | 0.495984939245 |
| DR_S_s0 | Overall | 0.547170251803 | 0.547549739632 | 0.547003206288 | 0.542604881630 |
| DR_S_s0 | Urban | 0.576275979945 | 0.577120070283 | 0.577075443474 | 0.574062298914 |
| DR_S_s0 | Rural | 0.491579228984 | 0.491441614331 | 0.490409247350 | 0.484427034392 |
| DR_W_s0 | Overall | 0.546667829706 | 0.546650233050 | 0.546331961110 | 0.545659286574 |
| DR_W_s0 | Urban | 0.574544620869 | 0.574218785790 | 0.573413721124 | 0.571465425446 |
| DR_W_s0 | Rural | 0.492986865040 | 0.493718868341 | 0.494340355033 | 0.496022121260 |

| Configuration | Domain | K1 to K2 | K2 to K4 | K4 to K8 |
| --- | --- | --- | --- | --- |
| E0 | Overall | up | up | down |
| E0 | Urban | up | up | up |
| E0 | Rural | down | down | down |
| RS_D4_s2 | Overall | up | up | down |
| RS_D4_s2 | Urban | up | up | down |
| RS_D4_s2 | Rural | down | down | down |
| RW_D4_s0 | Overall | down | down | down |
| RW_D4_s0 | Urban | down | down | down |
| RW_D4_s0 | Rural | up | up | up |
| DR_S_s0 | Overall | up | down | down |
| DR_S_s0 | Urban | up | down | down |
| DR_S_s0 | Rural | down | down | down |
| DR_W_s0 | Overall | down | down | down |
| DR_W_s0 | Urban | down | down | down |
| DR_W_s0 | Rural | up | up | up |

Observed E0/RS-D4* Overall peaks at K4; DR-S* Overall peaks at K2 and then declines. DR-S* Urban rises at K2 then declines, and Rural declines throughout. RW-D4*/DR-W* Overall and Urban decline throughout while Rural rises throughout.

## Best epochs and selection curves

| Run | Prior D4 best epoch | DR best epoch |
| --- | --- | --- |
| DR_S_s0 | 1 | 1 |
| DR_S_s1 | 1 | 1 |
| DR_S_s2 | 2 | 1 |
| DR_W_s0 | 1 | 1 |
| DR_W_s1 | 1 | 1 |
| DR_W_s2 | 1 | 1 |

| Run | Epoch | Val-sel L |
| --- | --- | --- |
| DR_S_s0 | 1 | 0.551307425640 |
| DR_S_s0 | 2 | 0.547533008070 |
| DR_S_s0 | 3 | 0.542233398652 |
| DR_S_s0 | 4 | 0.542070206220 |
| DR_S_s0 | 5 | 0.539222762442 |
| DR_S_s0 | 6 | 0.541962808459 |
| DR_S_s0 | 7 | 0.537673555522 |
| DR_S_s0 | 8 | 0.533597828563 |
| DR_S_s0 | 9 | 0.532419830465 |
| DR_S_s0 | 10 | 0.532013115407 |
| DR_S_s1 | 1 | 0.546542551561 |
| DR_S_s1 | 2 | 0.545154411883 |
| DR_S_s1 | 3 | 0.540815500534 |
| DR_S_s1 | 4 | 0.541335161220 |
| DR_S_s1 | 5 | 0.536613654515 |
| DR_S_s1 | 6 | 0.535956866255 |
| DR_S_s1 | 7 | 0.537111460498 |
| DR_S_s1 | 8 | 0.533690385827 |
| DR_S_s1 | 9 | 0.533638202687 |
| DR_S_s1 | 10 | 0.532908598682 |
| DR_S_s2 | 1 | 0.550376498926 |
| DR_S_s2 | 2 | 0.549921659970 |
| DR_S_s2 | 3 | 0.544838145551 |
| DR_S_s2 | 4 | 0.542232997825 |
| DR_S_s2 | 5 | 0.542011359817 |
| DR_S_s2 | 6 | 0.542985550587 |
| DR_S_s2 | 7 | 0.542475161746 |
| DR_S_s2 | 8 | 0.538390945958 |
| DR_S_s2 | 9 | 0.539426288868 |
| DR_S_s2 | 10 | 0.539161936245 |
| DR_W_s0 | 1 | 0.549184807690 |
| DR_W_s0 | 2 | 0.547039191281 |
| DR_W_s0 | 3 | 0.545776022113 |
| DR_W_s1 | 1 | 0.548870770234 |
| DR_W_s1 | 2 | 0.542381419072 |
| DR_W_s1 | 3 | 0.543953554618 |
| DR_W_s2 | 1 | 0.545695063408 |
| DR_W_s2 | 2 | 0.545495477647 |
| DR_W_s2 | 3 | 0.545572523085 |

## Interpretation limits

Observed DR-P3 meets the explicitly specified comparison of mean L differences because Rural declines less than Urban; neither domain has a positive mean gain. This verdict does not assert positive improvement. Observed gate norms confirm the requested trained-weight condition, with no extra ablation. Observed no per-image table, image, weight or cache is included in this report.
