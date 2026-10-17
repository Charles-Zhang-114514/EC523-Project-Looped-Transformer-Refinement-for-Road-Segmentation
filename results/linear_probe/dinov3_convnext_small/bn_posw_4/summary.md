# dinov3_convnext_small Linear Probe

Device: `cuda`; pos_weight: `4`.
Head: `bn_linear`; trainable parameters: `1153`.
Checkpoint encoder: `dinov3_convnext_small`; checkpoint pos_weight: `4.0`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.547853 | 0.599462 | 0.059139 | 0.465593 |
| Rural | 0.443907 | 0.521486 | 0.072052 | 0.392810 |
| Overall | 0.515034 | 0.564367 | 0.064114 | 0.439650 |

Predicted road fraction among valid pixels at threshold 0.5:

| Group | Predicted road / valid pixels |
|---|---:|
| Urban | 0.08849415 |
| Rural | 0.02167674 |
| Overall | 0.04842866 |

## Validation at best Overall A threshold (0.55)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.554442 | 0.608675 | 0.060840 | 0.471992 |
| Rural | 0.439652 | 0.509256 | 0.071272 | 0.386857 |
| Overall | 0.518063 | 0.564954 | 0.064864 | 0.441491 |

Overall A by threshold: 0.05=0.186197, 0.1=0.262189, 0.15=0.313295, 0.2=0.350673, 0.25=0.378675, 0.3=0.399563, 0.35=0.415400, 0.4=0.426753, 0.45=0.434785, 0.5=0.439650, 0.55=0.441491, 0.6=0.440371, 0.65=0.436348, 0.7=0.428840, 0.75=0.417178, 0.8=0.400183, 0.85=0.375538, 0.9=0.337488, 0.95=0.269248

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.010002061004129363`.
Val forward seconds: `16.693439815891907`; metric worker CPU seconds: `2971.3251371694496`; metric wall seconds: `528.4682930570561`; pipeline wall seconds: `537.2041324829916`.
Checkpoint: `runs/linear_probe/checkpoints/dinov3_convnext_small_bn_posw4.pt`.
