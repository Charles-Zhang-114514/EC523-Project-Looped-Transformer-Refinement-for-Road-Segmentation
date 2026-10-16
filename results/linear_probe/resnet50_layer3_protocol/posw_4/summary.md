# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `4`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.479103 | 0.559804 | 0.048318 | 0.417156 |
| Rural | 0.312115 | 0.364898 | 0.048985 | 0.275324 |
| Overall | 0.425939 | 0.479089 | 0.048572 | 0.366410 |

## Validation at best Overall A threshold (0.4)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.481385 | 0.554408 | 0.047307 | 0.416476 |
| Rural | 0.327198 | 0.394432 | 0.051484 | 0.292226 |
| Overall | 0.432793 | 0.488785 | 0.048888 | 0.372810 |

Overall A by threshold: 0.05=0.196260, 0.1=0.259049, 0.15=0.300624, 0.2=0.329524, 0.25=0.349428, 0.3=0.362555, 0.35=0.369963, 0.4=0.372810, 0.45=0.371390, 0.5=0.366410, 0.55=0.358096, 0.6=0.346318, 0.65=0.331050, 0.7=0.312239, 0.75=0.289005, 0.8=0.260165, 0.85=0.223861, 0.9=0.176209, 0.95=0.107499

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.005232265589612555`.
Val forward seconds: `8.732651269063354`; metric worker CPU seconds: `2905.2652259744937`; metric wall seconds: `518.0397359030321`; pipeline wall seconds: `523.1428681149846`.
Checkpoint: `runs/linear_probe/checkpoints/posw_4.pt`.
