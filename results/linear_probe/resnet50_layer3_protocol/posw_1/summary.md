# ResNet-50 Layer3 Linear Probe

Device: `cuda`; pos_weight: `None`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.430623 | 0.522689 | 0.040511 | 0.380221 |
| Rural | 0.272939 | 0.290987 | 0.037857 | 0.231337 |
| Overall | 0.379050 | 0.424055 | 0.039458 | 0.324633 |

## Validation at best Overall A threshold (0.25)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.480965 | 0.550929 | 0.046788 | 0.415119 |
| Rural | 0.327715 | 0.383929 | 0.049348 | 0.288906 |
| Overall | 0.431702 | 0.482019 | 0.047784 | 0.370013 |

Overall A by threshold: 0.05=0.245365, 0.1=0.313736, 0.15=0.348259, 0.2=0.364593, 0.25=0.370013, 0.3=0.368274, 0.35=0.361702, 0.4=0.351626, 0.45=0.339148, 0.5=0.324633, 0.55=0.308458, 0.6=0.290523, 0.65=0.270736, 0.7=0.248924, 0.75=0.224736, 0.8=0.197509, 0.85=0.165625, 0.9=0.125830, 0.95=0.071661

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.004962042462467978`.
Val forward seconds: `8.281648869859055`; metric worker CPU seconds: `2704.4315709389048`; metric wall seconds: `484.511664162972`; pipeline wall seconds: `487.5129061189946`.
Checkpoint: `runs/linear_probe/checkpoints/posw_1.pt`.
