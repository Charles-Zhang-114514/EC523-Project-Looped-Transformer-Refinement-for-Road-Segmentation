# Inference probes

Observed report status: stress_only.

Observed scope: device=cpu; limit per domain=None; requested checkpoints=6.

## Confidence gating

Observed selection protocol: one tau per checkpoint, selected only on Val-sel by 0.33*A(K1)+0.67*A(K4). Exact ties keep the lowest tau. Val-rep curves are diagnostic only.

Observed gating evaluation pending manual GPU execution.

## Fixed y0 answer with current residual

Observed new-mode semantics: answer tokens use y0; hidden memory and the current-logit residual base accumulate normally. Historical no-mask-feedback instead uses y0 as both answer and residual base.

Observed new-mode evaluation pending manual GPU execution.

## Stored stress supplement

Observed data only from existing stress results. Stress seeds 523/524/525 are Counts-pooled before official summarize in that source. Category drops compare the matched clean tile assignments, not an unmatched clean subset. Positive drop means clean minus stress. y0 Counts are identical across model seeds and are counted once.

| Condition | Domain | y0 IoU | y0 clDice | y0 BIoU | y0 A | IoU drop | clDice drop | BIoU drop | A drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary | Urban | 0.671418 | 0.714745 | 0.082375 | 0.566607 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ordinary | Rural | 0.506060 | 0.610700 | 0.091815 | 0.454603 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ordinary | Overall | 0.617080 | 0.667221 | 0.086718 | 0.526050 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| shadow | Urban | 0.673793 | 0.715695 | 0.081322 | 0.567869 | 0.002124 | 0.003446 | 0.000522 | 0.002200 |
| shadow | Rural | 0.514019 | 0.626973 | 0.091579 | 0.463417 | -0.003083 | -0.000808 | 0.000198 | -0.001744 |
| shadow | Overall | 0.622658 | 0.674501 | 0.086030 | 0.530886 | 0.000517 | 0.001557 | 0.000369 | 0.000799 |
| occlusion | Urban | 0.665187 | 0.707645 | 0.082399 | 0.561367 | 0.006580 | 0.006414 | 0.001115 | 0.005437 |
| occlusion | Rural | 0.496764 | 0.584950 | 0.091854 | 0.442238 | 0.004424 | 0.009554 | 0.001171 | 0.005312 |
| occlusion | Overall | 0.608784 | 0.651920 | 0.086751 | 0.517318 | 0.005681 | 0.007785 | 0.001160 | 0.005408 |
| blur | Urban | 0.666276 | 0.708449 | 0.080070 | 0.561687 | 0.000120 | 0.002829 | 0.001763 | 0.001261 |
| blur | Rural | 0.512050 | 0.613267 | 0.092675 | 0.458540 | -0.006178 | -0.002229 | -0.001967 | -0.004151 |
| blur | Overall | 0.614784 | 0.664846 | 0.085917 | 0.524029 | -0.001502 | 0.000895 | -0.000009 | -0.000484 |

| Condition | Domain | IoU break-even precision (%) | v2 K4 added precision mean +/- sample std (%) | Relation | Seeds above |
| --- | --- | --- | --- | --- | --- |
| ordinary | Urban | 40.170546 | 34.190206 +/- 2.872451 | below | 0/3 |
| ordinary | Rural | 33.601602 | 47.900708 +/- 0.648008 | above | 3/3 |
| ordinary | Overall | 38.160156 | 40.231175 +/- 2.391282 | above | 2/3 |
| shadow | Urban | 40.255461 | 33.276354 +/- 2.588805 | below | 0/3 |
| shadow | Rural | 33.950611 | 48.406491 +/- 0.766759 | above | 3/3 |
| shadow | Overall | 38.372725 | 40.004754 +/- 2.528254 | above | 2/3 |
| occlusion | Urban | 39.946687 | 32.865501 +/- 2.948089 | below | 0/3 |
| occlusion | Rural | 33.189213 | 47.577194 +/- 0.704798 | above | 3/3 |
| occlusion | Overall | 37.841240 | 39.115923 +/- 2.555560 | above | 2/3 |
| blur | Urban | 39.985942 | 38.597614 +/- 3.251062 | below | 1/3 |
| blur | Rural | 33.864618 | 46.938027 +/- 0.463376 | above | 3/3 |
| blur | Overall | 38.072198 | 42.384415 +/- 2.168727 | above | 3/3 |

Observed break-even formula: IoU_y0/(1+IoU_y0), an add-only IoU comparison, not an A-score guarantee. Undefined zero-denominator precisions are --.

Observed provenance: stress rows copy datasets.ordinary.models.0.groups.{domain}.y0_draft.scores or combined_stress.models.0.groups.{category/domain}.y0_draft.scores. Drops use paired_clean.stress_pooled for the same group. Added precision uses the corresponding three models at 4.diagnostics.added_accuracy.
