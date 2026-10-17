# dinov3_vitl16_sat Linear Probe

Device: `cuda`; pos_weight: `4`.
Head: `bn_linear`; trainable parameters: `3073`.
Checkpoint encoder: `dinov3_vitl16_sat`; checkpoint pos_weight: `4.0`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.636309 | 0.674681 | 0.077386 | 0.536036 |
| Rural | 0.493505 | 0.592914 | 0.074168 | 0.439460 |
| Overall | 0.589464 | 0.638496 | 0.076015 | 0.501484 |

Predicted road fraction among valid pixels at threshold 0.5:

| Group | Predicted road / valid pixels |
|---|---:|
| Urban | 0.08922241 |
| Rural | 0.02357191 |
| Overall | 0.04985662 |

## Validation at best Overall A threshold (0.55)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.643488 | 0.678708 | 0.079693 | 0.541295 |
| Rural | 0.488847 | 0.579988 | 0.073169 | 0.433054 |
| Overall | 0.592622 | 0.635398 | 0.076905 | 0.502311 |

Overall A by threshold: 0.05=0.333398, 0.1=0.399296, 0.15=0.433746, 0.2=0.455358, 0.25=0.470386, 0.3=0.481197, 0.35=0.489210, 0.4=0.494992, 0.45=0.499068, 0.5=0.501484, 0.55=0.502311, 0.6=0.501842, 0.65=0.499976, 0.7=0.496417, 0.75=0.490605, 0.8=0.481201, 0.85=0.466117, 0.9=0.439836, 0.95=0.384549

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.036122169190768876`.
Val forward seconds: `60.28790037939325`; metric worker CPU seconds: `3225.035525277839`; metric wall seconds: `606.6622446450638`; pipeline wall seconds: `612.2583128649276`.
Checkpoint: `runs/linear_probe/checkpoints/dinov3_vitl16_sat_bn_posw4.pt`.
