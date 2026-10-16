# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `None`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.300404 | 0.359280 | 0.020445 | 0.262075 |
| Rural | 0.238631 | 0.347929 | 0.032592 | 0.230212 |
| Overall | 0.279836 | 0.353949 | 0.025432 | 0.251189 |

## Validation at best Overall A threshold (0.9)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.477476 | 0.563806 | 0.048284 | 0.417537 |
| Rural | 0.323532 | 0.395245 | 0.053515 | 0.291042 |
| Overall | 0.429071 | 0.494299 | 0.050248 | 0.372875 |

Overall A by threshold: 0.05=0.090380, 0.1=0.114008, 0.15=0.133578, 0.2=0.151339, 0.25=0.168243, 0.3=0.184871, 0.35=0.201243, 0.4=0.217587, 0.45=0.234131, 0.5=0.251189, 0.55=0.268530, 0.6=0.286263, 0.65=0.304477, 0.7=0.322716, 0.75=0.340936, 0.8=0.358136, 0.85=0.371195, 0.9=0.372875, 0.95=0.337288

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.006364998641190074`.
Val forward seconds: `10.623182732146233`; metric worker CPU seconds: `3933.4638623498613`; metric wall seconds: `688.5355545929633`; pipeline wall seconds: `693.2880641800584`.
Checkpoint: `runs/linear_probe/checkpoints/posw_auto.pt`.
