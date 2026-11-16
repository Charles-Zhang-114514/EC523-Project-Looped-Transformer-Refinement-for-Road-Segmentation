# Observed draft threshold calibration

Thresholds are selected only by Val-sel Overall A; ties choose the smaller tau. Val-rep per-domain optima are post-hoc, optimistic diagnostics. Train and Val-rep are never used to choose the reported global threshold.

| Head | Set | Domain | Selected tau | A at 0.5 | A at selected tau | Gain |
|---|---|---|---:|---:|---:|---:|
| linear | Train | Overall | 0.45 | 0.566751465 | 0.556686164 | -0.010065301 |
| linear | Train | Urban | 0.45 | 0.568768334 | 0.558849358 | -0.009918976 |
| linear | Train | Rural | 0.45 | 0.549087603 | 0.538600627 | -0.010486975 |
| linear | Val-sel | Overall | 0.45 | 0.536019110 | 0.536453840 | +0.000434730 |
| linear | Val-sel | Urban | 0.45 | 0.565774331 | 0.562127498 | -0.003646832 |
| linear | Val-sel | Rural | 0.45 | 0.455319372 | 0.465778743 | +0.010459371 |
| linear | Val-rep | Overall | 0.45 | 0.526050033 | 0.527689285 | +0.001639252 |
| linear | Val-rep | Urban | 0.45 | 0.566607394 | 0.563745858 | -0.002861537 |
| linear | Val-rep | Rural | 0.45 | 0.454603115 | 0.463334460 | +0.008731345 |
| A1-i | Train | Overall | 0.60 | 0.546353896 | 0.576254019 | +0.029900123 |
| A1-i | Train | Urban | 0.60 | 0.537624959 | 0.567524489 | +0.029899530 |
| A1-i | Train | Rural | 0.60 | 0.559503043 | 0.588688141 | +0.029185098 |
| A1-i | Val-sel | Overall | 0.60 | 0.543394425 | 0.546853474 | +0.003459049 |
| A1-i | Val-sel | Urban | 0.60 | 0.560576848 | 0.572497847 | +0.011920998 |
| A1-i | Val-sel | Rural | 0.60 | 0.491196073 | 0.473890784 | -0.017305289 |
| A1-i | Val-rep | Overall | 0.60 | 0.538246121 | 0.540108604 | +0.001862484 |
| A1-i | Val-rep | Urban | 0.60 | 0.566749782 | 0.578250427 | +0.011500645 |
| A1-i | Val-rep | Rural | 0.60 | 0.483754212 | 0.470764409 | -0.012989803 |

## Post-hoc Val-rep optima (optimistic)

| Head | Domain | tau | A |
|---|---|---:|---:|
| linear | Overall | 0.45 | 0.527689285 |
| linear | Urban | 0.55 | 0.567697168 |
| linear | Rural | 0.30 | 0.476364566 |
| A1-i | Overall | 0.60 | 0.540108604 |
| A1-i | Urban | 0.70 | 0.581233457 |
| A1-i | Rural | 0.35 | 0.492274252 |

## Registered predictions

Observed G: True.
observed calibration comparison; direction B unchanged
Observed P3: False; calibration gain=+0.001639252, 70% of recorded refiner gain=0.004750584.
Observed P4: False; calibration gain=+0.001862484, 70% of recorded refiner gain=0.003397822.
