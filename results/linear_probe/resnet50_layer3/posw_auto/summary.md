# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `17.974647`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | Quality |
|---|---:|---:|---:|---:|
| Urban | 0.300404 | 0.359280 | 0.020445 | 0.262075 |
| Rural | 0.238631 | 0.347929 | 0.032592 | 0.230212 |
| Overall | 0.279836 | 0.353949 | 0.025432 | 0.251189 |

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.005271446078837447`.
Val forward seconds: `8.798043505579699`; metric worker CPU seconds: `557.369629099092`; metric wall seconds: `152.06890152598498`; pipeline wall seconds: `162.24736337299692`.
Checkpoint: `runs/linear_probe/checkpoints/posw_auto.pt`.
