# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `2`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.432576 | 0.525257 | 0.042197 | 0.382305 |
| Rural | 0.270282 | 0.294422 | 0.039112 | 0.231290 |
| Overall | 0.379680 | 0.427236 | 0.040984 | 0.326208 |

## Validation at best Overall A threshold (0.25)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.481204 | 0.551418 | 0.046987 | 0.415425 |
| Rural | 0.327269 | 0.389410 | 0.050425 | 0.290543 |
| Overall | 0.432337 | 0.484857 | 0.048302 | 0.371286 |

Overall A by threshold: 0.05=0.248112, 0.1=0.315540, 0.15=0.349552, 0.2=0.365831, 0.25=0.371286, 0.3=0.369660, 0.35=0.363223, 0.4=0.353489, 0.45=0.340903, 0.5=0.326208, 0.55=0.309617, 0.6=0.291195, 0.65=0.270591, 0.7=0.247980, 0.75=0.222549, 0.8=0.193928, 0.85=0.160669, 0.9=0.119236, 0.95=0.064336

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.00522919408720346`.
Val forward seconds: `8.727524931542575`; metric worker CPU seconds: `2767.915985337575`; metric wall seconds: `495.1584983849898`; pipeline wall seconds: `500.1039636519272`.
Checkpoint: `runs/linear_probe/checkpoints/posw_2.pt`.
