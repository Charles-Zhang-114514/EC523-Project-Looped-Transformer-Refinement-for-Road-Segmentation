# Wave 5 Train holdout acceptance

## Gates

| Gate | Result | Observed evidence |
| --- | --- | --- |
| HG1 | PASS | 200 held-out Train IDs (100/domain), 2322 remaining Train IDs; intersection=0. The held-out list is copied into metrics.json. |
| KG | PASS | 4175 endpoints equal K across 835 clean and 835 stress images. |

## Preregistered decisions

| Prediction/rule | Result | Calculation |
| --- | --- | --- |
| HO-P1 | holds | held-out Train best epoch 7 >=5; Val-sel best epoch 1 <=2 |
| HO-R1 | Train/Val distribution difference | 7 - 1 = 6 >=3 |

Observed HO-R1 selects the preregistered interpretation: **early stopping is mainly associated with Train/Val distribution differences**.

Observed this is the wording required by the preregistered rule; a single holdout experiment does not independently identify a causal mechanism.

## Job and configuration

| Job | GPU | Compute mode | Wall time | Peak GiB | Selected epoch |
| --- | --- | --- | --- | --- | --- |
| 8821489 | NVIDIA L40S | Exclusive_Process | 00:58:29 | 6.219810 | 1 |

Observed accounting failed=0/exit_status=0, all ten epochs completed, no error/timeout markers, and FOCUS_EVAL_COMPLETE. Observed D4 is on, detail head is off, seed=0, lr=3e-4, warmup=500 + cosine, ten epochs, batch=4 and Val-sel=834. Checkpoint selection remains Val-sel.

## Epoch curves

| Epoch | Held Train Overall | Held Train Urban | Held Train Rural | Val-sel Overall | Val-sel Urban | Val-sel Rural |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.556681954 | 0.550118150 | 0.562761452 | 0.548062380 | 0.560244299 | 0.523003384 |
| 2 | 0.557772175 | 0.550709110 | 0.565205247 | 0.545370502 | 0.557854218 | 0.519740173 |
| 3 | 0.557414318 | 0.552402582 | 0.558539653 | 0.545584641 | 0.558742324 | 0.518747279 |
| 4 | 0.558509495 | 0.552755612 | 0.561785335 | 0.543416452 | 0.556850145 | 0.516044246 |
| 5 | 0.565353367 | 0.563135082 | 0.557595793 | 0.535031935 | 0.553403430 | 0.499149591 |
| 6 | 0.564101517 | 0.562924636 | 0.553427256 | 0.534895943 | 0.552785370 | 0.499719036 |
| 7 | 0.567667330 | 0.563259576 | 0.566156596 | 0.529622982 | 0.549733545 | 0.490025251 |
| 8 | 0.560454331 | 0.557343123 | 0.555566203 | 0.523242382 | 0.542064913 | 0.485437966 |
| 9 | 0.558034048 | 0.556168461 | 0.549736092 | 0.522118123 | 0.539385604 | 0.487217204 |
| 10 | 0.561346334 | 0.559071734 | 0.554119018 | 0.522649871 | 0.541480760 | 0.484941841 |

| Split | Overall best epoch | Urban best epoch | Rural best epoch |
| --- | --- | --- | --- |
| Held-out Train | 7 | 7 | 7 |
| Val-sel | 1 | 1 | 1 |

Observed held-out Train Overall L rises from 0.556681954 at epoch 1 to its maximum 0.567667330 at epoch 7, with intervening reversals. Observed Val-sel Overall L has its maximum at epoch 1, then has a lower endpoint 0.522649871 at epoch 10.

## Selected-checkpoint descriptive results

| Domain | Val-sel L | Val-rep L | Val-rep LB | A(K1) | A(K2) | A(K4) | A(K8) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.548062380 | 0.544536677 | 0.544132133 | 0.544272535 | 0.544579464 | 0.544610462 | 0.542487384 |
| Urban | 0.560244299 | 0.571502055 | 0.571119456 | 0.570981136 | 0.571393877 | 0.571711754 | 0.570218631 |
| Rural | 0.523003384 | 0.492012990 | 0.491674268 | 0.492168265 | 0.492285494 | 0.491870397 | 0.488983894 |

Observed held-out IDs are retained in metrics.json under gates.HG1.heldout_ids; original training and held-out lists remain in the run report. No inference, training, official metric recomputation or hyperparameter selection was performed during this acceptance task.
