# Observed LoRA smoke acceptance

| Gate | Observed result | Evidence |
|---|---|---|
| L0 | PASS | Maximum pixel difference 0; comparison covers 2097152 output pixels |
| M / B-P2 | Met in smoke | Batch-4 peak 5798432256 bytes = 5.400210857 GiB, bound 40 GiB |

| Rule | Observed outcome |
|---|---|
| B-P2 | Memory bound met in this batch-4 smoke; full-run peak remains unmeasured |
| B-P1 / B-R1 | Pending full training and Val-rep evaluation |

Observed L0 and the batch-4 memory bound permit proceeding to full LoRA training.

Observed job 8765080 used NVIDIA L40S for 114 wallclock seconds with failed=0 and exit_status=0. Actual temporary disk use was not recorded.
The L0 pixel count is derived from the saved source's batch of two Urban images and its 1024x1024 one-channel output; the log records the maximum difference directly.

| Batch | Updates | Peak bytes | Peak GiB | Minimum seconds | Mean seconds | Maximum seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 20 | 3563956736 | 3.319193363 | 0.323406352 | 0.437561918 | 2.538273494 |
| 4 | 20 | 5798432256 | 5.400210857 | 0.749302594 | 0.750878350 | 0.752158375 |

| Update (zero-based) | Batch-2 seconds | Batch-4 seconds |
|---:|---:|---:|
| 0 | 2.538273494 | 0.751077246 |
| 1 | 0.323753900 | 0.750168000 |
| 2 | 0.323406352 | 0.749302594 |
| 3 | 0.323797537 | 0.750684800 |
| 4 | 0.324078208 | 0.750600386 |
| 5 | 0.376660574 | 0.749837354 |
| 6 | 0.323655882 | 0.751176764 |
| 7 | 0.324099313 | 0.750195368 |
| 8 | 0.324148076 | 0.750363453 |
| 9 | 0.324391527 | 0.750759228 |
| 10 | 0.324176903 | 0.750647923 |
| 11 | 0.324543155 | 0.751234884 |
| 12 | 0.324019620 | 0.751123246 |
| 13 | 0.324641151 | 0.751905539 |
| 14 | 0.324345547 | 0.751089719 |
| 15 | 0.324657159 | 0.750980430 |
| 16 | 0.324525418 | 0.752158375 |
| 17 | 0.324590576 | 0.750784576 |
| 18 | 0.324453809 | 0.751662424 |
| 19 | 0.325020157 | 0.751814686 |

## Observed full-training runtime estimate

Observed T=0.752158374991 s/update and I=0.062203874506 s/image.
Observed training term 3155*T=2373.059673097450 s and evaluation term 11685*(I+0.125)=2187.477273601908 s.
Estimated requested seconds ceil(1.5*(3155*T+11685*(I+0.125))+900)=7741; h_rt=02:09:01.
No full LoRA training log, completed accounting record or output directory was found. Submission status without a receipt remains unverified; this estimate applies if full training has not yet been submitted.
