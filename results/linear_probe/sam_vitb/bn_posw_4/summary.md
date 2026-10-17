# sam_vitb Linear Probe

Device: `cuda`; pos_weight: `4`.
Head: `bn_linear`; trainable parameters: `769`.
Checkpoint encoder: `sam_vitb`; checkpoint pos_weight: `4.0`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.418491 | 0.528836 | 0.045729 | 0.377042 |
| Rural | 0.370882 | 0.496854 | 0.061967 | 0.346891 |
| Overall | 0.402900 | 0.512935 | 0.052118 | 0.365754 |

Predicted road fraction among valid pixels at threshold 0.5:

| Group | Predicted road / valid pixels |
|---|---:|
| Urban | 0.09999796 |
| Rural | 0.02974377 |
| Overall | 0.05787168 |

## Validation at best Overall A threshold (0.6)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.426142 | 0.548234 | 0.049956 | 0.387532 |
| Rural | 0.372402 | 0.491737 | 0.063841 | 0.346490 |
| Overall | 0.408630 | 0.522068 | 0.055439 | 0.372023 |

Overall A by threshold: 0.05=0.129326, 0.1=0.182467, 0.15=0.224065, 0.2=0.257905, 0.25=0.286079, 0.3=0.309588, 0.35=0.329122, 0.4=0.345021, 0.45=0.357076, 0.5=0.365754, 0.55=0.370892, 0.6=0.372023, 0.65=0.368822, 0.7=0.360708, 0.75=0.346767, 0.8=0.325227, 0.85=0.293595, 0.9=0.244967, 0.95=0.164680

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.024957376771225382`.
Val forward seconds: `41.65386183117516`; metric worker CPU seconds: `2918.115949406405`; metric wall seconds: `509.1118917519925`; pipeline wall seconds: `514.205625758972`.
Checkpoint: `runs/linear_probe/checkpoints/sam_vitb_bn_posw4.pt`.
