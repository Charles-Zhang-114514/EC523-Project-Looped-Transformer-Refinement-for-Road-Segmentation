# Observed full LoRA training acceptance

| Gate | Observed result | Evidence |
|---|---|---|
| LG1 | Not recorded; no failure assigned | Epoch-0 selection score absent; L0 max pixel difference 0 |
| LG2 | PASS | Val-sel 834; Val-rep 835; exact split membership; SHA-256 bdd7d7a89237c96305f0ad404a5c9ae1254ed8cbc0b6e0e4fcc7b514bd323c68 |

| Rule | Observed outcome | Evidence |
|---|---|---|
| B-P1 | Not met | A=0.525091501373; gain vs 0.538246=-0.013154498627, required [0.005,0.020]; IoU gain=-0.006830010017 > clDice gain=-0.032646283701 |
| B-R1 | Not met | A=0.525091501373 < 0.543246; Urban delta=-0.004688914325; Rural delta=-0.030337230803; each allowed >=-0.002 |

Observed registered B-R1 selects the frozen encoder for wave 2.

## Observed execution and configuration

Observed job 8766226 on NVIDIA L40S at source fe10d7b completed all five epochs and 3155 updates, with LORA_COMPLETE, failed=0 and exit_status=0. Total wallclock was 2988 s (49 min 48 s), below the 7741 s request.
Observed training CUDA allocation peak 5799431680 bytes (5.401141644 GiB). Observed best.pt and last.pt exist, each 15279789 bytes; neither is included in these results.
Observed config rank=16, alpha=16, q/k/v/o adapters in 24 blocks, AdamW lr=0.0001 and weight_decay=0, 300 warmup updates then cosine to zero, five epochs, batch=4, D4 augmentation, seed=0, native non-reentrant checkpointing.
Observed 3145728 adapter parameters and 656865 head parameters. The logged source initializes the head from runs/draft_head/A1-i/best.pt and uses all 834 Val-sel images. Projection names and head initialization are verified from the logged source; they are not separate fields in the raw report.
The raw report did not save a split-file fingerprint. The SHA above is calculated from the fixed split file; all saved selection and report image memberships agree exactly with that file.

## Observed epoch selection and timing

Whole-epoch wallclock was not recorded. The phase sum below adds synchronized GPU-update durations and selection wallclock; data loading, transfers and checkpoint writes are outside that sum.

| Epoch | Updates | Train-update seconds | Selection wall seconds | Recorded phase sum seconds | Val-sel A |
|---:|---:|---:|---:|---:|---:|
| 1 | 631 | 486.928719462 | 46.695056638 | 533.623776100 | 0.525868792451 |
| 2 | 631 | 484.582269392 | 46.753950415 | 531.336219807 | 0.523011804142 |
| 3 | 631 | 484.452280657 | 47.084750785 | 531.537031442 | 0.515501443021 |
| 4 | 631 | 484.284802591 | 47.405096517 | 531.689899108 | 0.519689869123 |
| 5 | 631 | 484.022282633 | 47.095503933 | 531.117786566 | 0.517315842749 |

Observed best epoch 1, best Val-sel A=0.525868792451. Observed last-minus-best Val-sel A=-0.008552949702.

## Observed Val-rep components

Each score cell lists IoU / clDice / BIoU / A. Delta uses the exact archived A1-i seed-0 score. The B-P1 A threshold above uses the preregistered rounded reference 0.538246.

| TTA | Domain | A1-i seed 0 | LoRA seed 0 | Paired delta |
|---|---|---|---|---|
| none | Overall | 0.622292764 / 0.695215840 / 0.092674934 / 0.538246121 | 0.615462754 / 0.662569556 / 0.092946287 / 0.525091501 | -0.006830010 / -0.032646284 / 0.000271353 / -0.013154620 |
| none | Urban | 0.665863264 / 0.721756819 / 0.086455522 / 0.566749782 | 0.666635539 / 0.700377490 / 0.093149259 / 0.562060868 | 0.000772274 / -0.021379330 / 0.006693737 / -0.004688914 |
| none | Rural | 0.530191817 / 0.662104901 / 0.100134166 / 0.483754212 | 0.502815521 / 0.611580119 / 0.092675923 / 0.453416981 | -0.027376295 / -0.050524782 / -0.007458243 / -0.030337231 |
| d4 | Overall | 0.629340648 / 0.701297970 / 0.093759915 / 0.543811698 | 0.620404769 / 0.666377749 / 0.093165057 / 0.528748721 | -0.008935878 / -0.034920221 / -0.000594858 / -0.015062977 |
| d4 | Urban | 0.671789277 / 0.724176384 / 0.085774398 / 0.570302433 | 0.671914475 / 0.703738537 / 0.093292793 / 0.565737358 | 0.000125199 / -0.020437847 / 0.007518395 / -0.004565076 |
| d4 | Rural | 0.539294897 / 0.672548599 / 0.103379628 / 0.492087953 | 0.506559175 / 0.615708286 / 0.092995083 / 0.456591090 | -0.032735722 / -0.056840313 / -0.010384545 / -0.035496864 |

## Observed ensemble draft-only record

Not recorded. The existing ensemble report contains K=1/2/4/8 after refinement, with no y0_draft scores for E0/E1/E2 on either split. No new inference was performed.
