# resnet50_layer3 Linear Probe

Device: `cuda`; pos_weight: `4`.
Head: `bn_linear`; trainable parameters: `3073`.
Checkpoint encoder: `resnet50_layer3`; checkpoint pos_weight: `4.0`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.471287 | 0.537943 | 0.045476 | 0.406122 |
| Rural | 0.326666 | 0.391589 | 0.050727 | 0.290955 |
| Overall | 0.426583 | 0.478674 | 0.047427 | 0.366379 |

Predicted road fraction among valid pixels at threshold 0.5:

| Group | Predicted road / valid pixels |
|---|---:|
| Urban | 0.08952270 |
| Rural | 0.01895882 |
| Overall | 0.04721072 |

## Validation at best Overall A threshold (0.55)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.477809 | 0.547574 | 0.046694 | 0.412515 |
| Rural | 0.323331 | 0.382581 | 0.050572 | 0.286554 |
| Overall | 0.429889 | 0.480566 | 0.048131 | 0.368740 |

Overall A by threshold: 0.05=0.152695, 0.1=0.206043, 0.15=0.245475, 0.2=0.276522, 0.25=0.302053, 0.3=0.322455, 0.35=0.338908, 0.4=0.351574, 0.45=0.360668, 0.5=0.366379, 0.55=0.368740, 0.6=0.367452, 0.65=0.362330, 0.7=0.352934, 0.75=0.338472, 0.8=0.317598, 0.85=0.287992, 0.9=0.243815, 0.95=0.171223

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.005400984735594178`.
Val forward seconds: `9.014243523706682`; metric worker CPU seconds: `2841.7675519444747`; metric wall seconds: `504.838868472958`; pipeline wall seconds: `511.5568776259897`.
Checkpoint: `runs/linear_probe/checkpoints/resnet50_layer3_bn_posw4.pt`.
