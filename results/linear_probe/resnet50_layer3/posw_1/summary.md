# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `1`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | Quality |
|---|---:|---:|---:|---:|
| Urban | 0.430623 | 0.522689 | 0.040511 | 0.380221 |
| Rural | 0.272939 | 0.290987 | 0.037857 | 0.231337 |
| Overall | 0.379050 | 0.424055 | 0.039458 | 0.324633 |

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.005068459394672487`.
Val forward seconds: `8.459258729708381`; metric worker CPU seconds: `391.16259481891757`; metric wall seconds: `91.89650360902306`; pipeline wall seconds: `97.68639759300277`.
Checkpoint: `runs/linear_probe/checkpoints/posw_1.pt`.
