# dinov3_vitl16_web Linear Probe

Device: `cuda`; pos_weight: `4`.
Head: `bn_linear`; trainable parameters: `3073`.
Checkpoint encoder: `dinov3_vitl16_web`; checkpoint pos_weight: `4.0`.

Official metrics are computed by summing per-image `Counts` before `summarize`.

## Validation at threshold 0.5

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.664708 | 0.705534 | 0.084496 | 0.560913 |
| Rural | 0.522136 | 0.619433 | 0.095428 | 0.465984 |
| Overall | 0.617080 | 0.666980 | 0.089365 | 0.526507 |

Predicted road fraction among valid pixels at threshold 0.5:

| Group | Predicted road / valid pixels |
|---|---:|
| Urban | 0.07911309 |
| Rural | 0.02193572 |
| Overall | 0.04482802 |

## Validation at best Overall A threshold (0.45)

Threshold selected on Val; results are optimistically biased.

| Group | Road IoU | clDice | Boundary IoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.660550 | 0.702564 | 0.083060 | 0.557656 |
| Rural | 0.527599 | 0.634482 | 0.096880 | 0.473520 |
| Overall | 0.616120 | 0.671760 | 0.089235 | 0.527435 |

Overall A by threshold: 0.05=0.361430, 0.1=0.435604, 0.15=0.472494, 0.2=0.494426, 0.25=0.508564, 0.3=0.517643, 0.35=0.523384, 0.4=0.526518, 0.45=0.527435, 0.5=0.526507, 0.55=0.523761, 0.6=0.519213, 0.65=0.512618, 0.7=0.503865, 0.75=0.491673, 0.8=0.474965, 0.85=0.451668, 0.9=0.416254, 0.95=0.351265

GT self-score sanity check (5 images): `{'road_iou': 1.0, 'cldice': 1.0, 'boundary_iou': 1.0, 'skeleton_precision': 1.0, 'skeleton_recall': 1.0, 'quality': 1.0, 'counts': {'valid': 4684861, 'intersection': 327362, 'union': 327362, 'predicted_skeleton': 5544, 'true_skeleton': 5544, 'skeleton_precision_hits': 5544, 'skeleton_recall_hits': 5544, 'boundary_intersection': 31272, 'boundary_union': 31272}}`.
Mean validation seconds per image: `0.03622359147327721`.
Val forward seconds: `60.457174168899655`; metric worker CPU seconds: `3005.1773001884576`; metric wall seconds: `553.9085036420729`; pipeline wall seconds: `559.4865627189865`.
Checkpoint: `runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt`.
