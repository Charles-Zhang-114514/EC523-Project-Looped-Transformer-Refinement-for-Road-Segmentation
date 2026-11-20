# Wave 2 refiner acceptance

| Gate | Status | Evidence |
| --- | --- | --- |
| RG1 | PASS | Observed 12 comparisons, max absolute difference 0, tolerance 1e-5. |
| RG2 | PASS | Observed 48 comparisons, max absolute difference 0, tolerance 1e-5. |
| RG3 | PASS | Observed all three user-revised checks pass; based on mtime, not per-run fingerprints. |

| Prediction / rule | Status | Evidence |
| --- | --- | --- |
| R-P1 | Not met | Observed RS mean 0.543721770857 minus E0 0.547135438532 = -0.003413667675, outside [-0.002,0.003]. |
| R-P2 | Not met | Observed RW mean delta -0.001252445344, outside [0,0.003]. Observed lower dispersion: population SD RW 0.000815580227 < RS 0.002393607362; sample SD RW 0.000998877701 < RS 0.002931558341. |
| R-P3 | Not met | Observed selected candidate gains RS*=-0.001007843818 and RW*=-0.000474155629; neither reaches +0.001. |
| R-R1 | Neither group passes | Observed RS* L=0.546127594714 and RW* L=0.546661282903, both below 0.548135; RW* Urban delta=-0.003979108866 also falls below -0.003. |

Observed R-R1 retains E0 (e6 seed 0).

| Job | Run | GPU | Wall seconds | Peak GiB | Best epoch / step | Normal end |
| --- | --- | --- | --- | --- | --- | --- |
| 8779164 | cache | L40S | 1486 | Not recorded | None | PASS |
| 8779167 | RS_s0 | L40S | 3057 | 6.173218 | 2 | PASS |
| 8779168 | RS_s1 | L40S | 3173 | 6.173218 | 1 | PASS |
| 8779169 | RS_s2 | L40S | 3011 | 6.173218 | 1 | PASS |
| 8779170 | RW_s0 | L40S | 1240 | 6.173218 | 1 | PASS |
| 8779171 | RW_s1 | L40S | 1103 | 6.173218 | 1 | PASS |
| 8779172 | RW_s2 | L40S | 1149 | 6.173218 | 1 | PASS |

## Cache provenance

Observed 4191 float32 cached drafts and zero flipped pixels in every Train/Val-sel/Val-rep domain. Current manifest SHA-256: 06d97973fcec14cc113b7c3468d768e1e0a6d9b3d27998fd6aa14cf07c13dc90.

Observed head SHA-256 matches A1-i s0: 9429dacae36969948eddd5f4551c3fed9567a1acfee6526164fe1839bc3b568f. All six configs record runs/cache/tta_draft_a1i_s0. The loader checks head and split fingerprints before use (scripts/train_refiner.py:858; src/looproad/data/draft_cache.py:45). It does not save the manifest fingerprint in the run output. Observed RG3 now passes under the user-authorized mtime replacement; historical per-run hashes remain unrecorded. Observed this provenance-rule revision was authorized after viewing results and is not blind. The original e58dfa0 predictions and thresholds are unchanged.

Observed the complete current cache contains 4192 files: 4191 NPZ drafts and one manifest. Observed the manifest and latest file mtime is 2026-11-20T00:15:45.278569-05:00; the latest NPZ mtime is 2026-11-20T00:14:10.806519-05:00. Observed earliest refiner start is 2026-11-20T00:16:53-05:00 (jobs 8779167/8779168); the latest refiner end is 2026-11-20T01:26:13-05:00. Observed zero files at or after the earliest start, zero files after each job end, and zero files after the latest end. Observed the matching head SHA above satisfies the third check. These timestamps use America/New_York (UTC-05:00); exact epoch and UTC values are retained in metrics.json.

| Run | Start epoch | Cache end epoch | Seconds after cache end |
| --- | --- | --- | --- |
| RS_s0 | 1795151813 | 1795151745 | 68 |
| RS_s1 | 1795151813 | 1795151745 | 68 |
| RS_s2 | 1795152248 | 1795151745 | 503 |
| RW_s0 | 1795152318 | 1795151745 | 573 |
| RW_s1 | 1795153659 | 1795151745 | 1914 |
| RW_s2 | 1795154824 | 1795151745 | 3079 |

## Configuration verification

| Arm | LR | Warmup | Epochs | Batch | Initialization | Selection images |
| --- | --- | --- | --- | --- | --- | --- |
| RS | 0.0003 | 500 | 10 | 4 | Random refiner | 834 |
| RW | 0.0001 | 100 | 3 | 4 | e6 seed 0 best.pt | 834 |

Observed all seeds/configurations match their arm; K_train=4, per-step BCE pos_weight=4, AdamW wd=0.01, clip=1.0, L weights=(0.2,0.2,0.6), fixed selection 834 and report 835 names. All best.pt and last.pt files exist. Every job has accounting failed=0, exit_status=0, and a completion marker below its h_rt; no error or timeout was observed. Cache peak CUDA allocation is not recorded.

## Seed scores and registered adjudication

| Run | Best epoch | Val-sel L | Val-rep Overall L | Urban L | Rural L |
| --- | --- | --- | --- | --- | --- |
| RS_s0 | 2 | 0.548244378879 | 0.544581149641 | 0.571343585191 | 0.492283405106 |
| RS_s1 | 1 | 0.544589777151 | 0.540456568216 | 0.574501412335 | 0.476554347814 |
| RS_s2 | 1 | 0.550468413814 | 0.546127594714 | 0.576834408309 | 0.488190782087 |
| RW_s0 | 1 | 0.549283610072 | 0.546661282903 | 0.574515200261 | 0.493308583936 |
| RW_s1 | 1 | 0.549203356801 | 0.546231042674 | 0.576052872062 | 0.489626635128 |
| RW_s2 | 1 | 0.546165235698 | 0.544756653986 | 0.569151486998 | 0.496965667862 |
| E0 | Archived | 0.551907986584 | 0.547135438532 | 0.578494309127 | 0.488163331653 |

| Arm | Three-seed mean | Population SD | Sample SD | Mean minus E0 | Val-sel maximum seed |
| --- | --- | --- | --- | --- | --- |
| RS | 0.543721770857 | 0.002393607362 | 0.002931558341 | -0.003413667675 | RS_s2 |
| RW | 0.545882993187 | 0.000815580227 | 0.000998877701 | -0.001252445344 | RW_s0 |

Observed mean is sum(L_seed)/3; population SD uses divisor 3 and sample SD uses divisor 2. SD convention was not preregistered, so both are reported. Observed RW has smaller seed dispersion by both conventions. Observed Val-sel maxima select RS*=RS_s2 and RW*=RW_s0. Observed neither candidate passes R-R1.

| Candidate | Domain | Val-rep L minus E0 |
| --- | --- | --- |
| RS_s2 | Overall | -0.001007843818 |
| RS_s2 | Urban | -0.001659900818 |
| RS_s2 | Rural | 0.000027450433 |
| RW_s0 | Overall | -0.000474155629 |
| RW_s0 | Urban | -0.003979108866 |
| RW_s0 | Rural | 0.005145252282 |

## Descriptive K=1/2/4/8 comparisons

| Run | Domain | K | IoU | clDice | BIoU | A |
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
| RS_s2 | Overall | 1 | 0.631272623387 | 0.700476845529 | 0.094310445105 | 0.544641454374 |
| RS_s2 | Overall | 2 | 0.632993101548 | 0.699541763389 | 0.094671334947 | 0.545293346780 |
| RS_s2 | Overall | 4 | 0.635989668477 | 0.699072309224 | 0.095922652328 | 0.546901057472 |
| RS_s2 | Overall | 8 | 0.627055244259 | 0.696562047116 | 0.094409581959 | 0.541378152656 |
| RS_s2 | Urban | 1 | 0.674973675361 | 0.725785182548 | 0.086745421378 | 0.572571476721 |
| RS_s2 | Urban | 2 | 0.677546606383 | 0.727033468808 | 0.087451961620 | 0.574373736158 |
| RS_s2 | Urban | 4 | 0.683260338323 | 0.732026961069 | 0.089186760369 | 0.579075609556 |
| RS_s2 | Urban | 8 | 0.674912463642 | 0.738028249903 | 0.088979681054 | 0.576660643003 |
| RS_s2 | Rural | 1 | 0.538730595967 | 0.668660230274 | 0.103415049804 | 0.490646377027 |
| RS_s2 | Rural | 2 | 0.538703330773 | 0.665158224097 | 0.103372361328 | 0.489573604881 |
| RS_s2 | Rural | 4 | 0.537088425603 | 0.658532658378 | 0.104036495967 | 0.486911309509 |
| RS_s2 | Rural | 8 | 0.527903357103 | 0.646420016641 | 0.100982921028 | 0.478074267750 |
| RW_s0 | Overall | 1 | 0.634131302239 | 0.701921088628 | 0.096696186795 | 0.546981215067 |
| RW_s0 | Overall | 2 | 0.633159863924 | 0.702871220220 | 0.097174978741 | 0.546876293777 |
| RW_s0 | Overall | 4 | 0.631365551451 | 0.704778274354 | 0.096833552625 | 0.546482968556 |
| RW_s0 | Overall | 8 | 0.627966648182 | 0.707455349789 | 0.096463162312 | 0.545512561490 |
| RW_s0 | Urban | 1 | 0.678684032410 | 0.727800143598 | 0.088567683888 | 0.575395596062 |
| RW_s0 | Urban | 2 | 0.676992837541 | 0.728808451715 | 0.089249256414 | 0.574988805568 |
| RW_s0 | Urban | 4 | 0.675023686744 | 0.729093931670 | 0.089119218429 | 0.574063866559 |
| RW_s0 | Urban | 8 | 0.671325196928 | 0.729135827672 | 0.088202417086 | 0.572043830183 |
| RW_s0 | Rural | 1 | 0.540291395088 | 0.670094101924 | 0.106454450070 | 0.492464818135 |
| RW_s0 | Rural | 2 | 0.540599987806 | 0.671522758096 | 0.106670706869 | 0.493090962706 |
| RW_s0 | Rural | 4 | 0.539370280264 | 0.675908431294 | 0.106023550459 | 0.493662379612 |
| RW_s0 | Rural | 8 | 0.537424627715 | 0.682361233884 | 0.106212354310 | 0.494663154885 |

## Every epoch Val-sel L

| Run | Epoch | Val-sel L |
| --- | --- | --- |
| RS_s0 | 1 | 0.548184491625 |
| RS_s0 | 2 | 0.548244378879 |
| RS_s0 | 3 | 0.546712081772 |
| RS_s0 | 4 | 0.545183781568 |
| RS_s0 | 5 | 0.539635993769 |
| RS_s0 | 6 | 0.542889765501 |
| RS_s0 | 7 | 0.538700486144 |
| RS_s0 | 8 | 0.535721336508 |
| RS_s0 | 9 | 0.531816811340 |
| RS_s0 | 10 | 0.531591219436 |
| RS_s1 | 1 | 0.544589777151 |
| RS_s1 | 2 | 0.537165523251 |
| RS_s1 | 3 | 0.541697732953 |
| RS_s1 | 4 | 0.537918642564 |
| RS_s1 | 5 | 0.533659694595 |
| RS_s1 | 6 | 0.525998171797 |
| RS_s1 | 7 | 0.531706539349 |
| RS_s1 | 8 | 0.525305124532 |
| RS_s1 | 9 | 0.523186988360 |
| RS_s1 | 10 | 0.522191409115 |
| RS_s2 | 1 | 0.550468413814 |
| RS_s2 | 2 | 0.545705762368 |
| RS_s2 | 3 | 0.540575486490 |
| RS_s2 | 4 | 0.538900003889 |
| RS_s2 | 5 | 0.541453191854 |
| RS_s2 | 6 | 0.541242038233 |
| RS_s2 | 7 | 0.537334724175 |
| RS_s2 | 8 | 0.531493833593 |
| RS_s2 | 9 | 0.531388913100 |
| RS_s2 | 10 | 0.530244246762 |
| RW_s0 | 1 | 0.549283610072 |
| RW_s0 | 2 | 0.545513187646 |
| RW_s0 | 3 | 0.543856572902 |
| RW_s1 | 1 | 0.549203356801 |
| RW_s1 | 2 | 0.542474549623 |
| RW_s1 | 3 | 0.543373970971 |
| RW_s2 | 1 | 0.546165235698 |
| RW_s2 | 2 | 0.543833184883 |
| RW_s2 | 3 | 0.543534410733 |

Observed only archived official metrics were aggregated; no images, inference, official Counts recomputation, per-image table or cache is included. Observed the revised RG3 passes on mtime evidence, all R predictions are not met, and R-R1 retains E0.
