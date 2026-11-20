# Wave 2 LoRA retry acceptance

| Gate | Status | Evidence |
| --- | --- | --- |
| LG | PASS | Observed frozen_head_unchanged=true and FROZEN_HEAD_PASS at job 8779165 log line 1278; exact head-state comparison includes parameters and BN statistics/counters. |

| Prediction / rule | Status | Evidence |
| --- | --- | --- |
| LA-P1 | Met | Observed 0.538729171519 - 0.538246 = 0.000483171519, in [-0.003,0.008]. |
| LA-R1 | Not met | Observed A=0.538729171519 < 0.543246; Urban delta=0.005562225015, Rural delta=-0.007665285519. |
| LC-P1 | Not met; curve assessment qualitative | Observed best epoch=2, not 1; Val-rep A=0.528911746203 <= 0.533. |
| LC-R1 | Qualitative else branch | Observed LC-P1 is not met; registered wording: degradation is related to LoRA. |

Observed LA-R1 specifies abandoning LoRA without another retry.

| Job | Run | GPU | Wall seconds | Peak GiB | Best epoch / step | Normal end |
| --- | --- | --- | --- | --- | --- | --- |
| 8779165 | LA | L40S | 2001 | 5.394516 | 1 / 200 | PASS |
| 8779166 | LC | L40S | 1403 | 3.768252 | 2 | PASS |

## Configuration verification

| Run | LR | Warmup | Epochs | Batch | Freeze head | LoRA | D4 | Val-sel images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LA | 0.00002 | 100 | 2 | 4 | Yes; BN eval | rank/alpha 16/16 | Yes | 834 |
| LC | 0.0001 | 300 | 5 | 4 | No | Disabled | Yes | 834 |

Observed both heads initialize from A1-i s0 in scripts/train_lora_draft.py:131 at source 71822ec; both recorded L0 maximum differences are zero. Selection names match all 834 fixed names and report names match all 835 fixed names. LA completes 1262 updates; LC completes 3155. Both best.pt and last.pt exist. LA trains 3,145,728 adapter parameters and zero head parameters; LC trains 656,865 head parameters and no adapters. Accounting failed=0 and exit_status=0; no error or timeout was observed.

## LA Val-sel curve

| Completed updates | Epoch | A |
| --- | --- | --- |
| 200 | 1 | 0.538982479791 |
| 400 | 1 | 0.534881303141 |
| 600 | 1 | 0.536025852116 |
| 631 | 1 | 0.535673171388 |
| 800 | 2 | 0.535115269595 |
| 1000 | 2 | 0.534856616167 |
| 1200 | 2 | 0.534849740227 |
| 1262 | 2 | 0.534870526709 |

Observed best checkpoint at step 200, epoch 1, A=0.538982479791. Endpoint evaluations at 631 and 1262 supplement the 200-step interval. Last-minus-best=-0.004111953082.

## LC versus first LoRA Val-sel curve

| Epoch | LC A | First LoRA A |
| --- | --- | --- |
| 1 | 0.527155385725 | 0.525868792451 |
| 2 | 0.528338687826 | 0.523011804142 |
| 3 | 0.522775986683 | 0.515501443021 |
| 4 | 0.526162463303 | 0.519689869123 |
| 5 | 0.524775803214 | 0.517315842749 |

Observed LC peaks at epoch 2, declines at epoch 3, rebounds at epoch 4 and declines at epoch 5; every post-peak score stays below its peak. Observed first LoRA peaks at epoch 1 with a later rebound at epoch 4. Curve similarity and subsequent decline remain qualitative; no threshold is added. LC last-minus-best=-0.003562884612.

Observed LC epoch-1 Val-sel A=0.527155385725 versus first LoRA A=0.525868792451, a difference of 0.001286593273 (about 0.0013). Observed that continuing to train only the head without LoRA also produces a decline of almost the same magnitude at the first epoch. Observed that, with the LC report-A bound already satisfied, the literal LC-R1 branch in this comparison depends only on whether the best epoch is 1. Observed this condition does not represent the intended question well; the registered literal verdict is retained alongside this descriptive limitation.

## Val-rep component comparisons

| Run | Mode | Domain | IoU | clDice | BIoU | A | Delta IoU | Delta clDice | Delta BIoU | Delta A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LA | none | Overall | 0.626142734778 | 0.687756300978 | 0.096654569182 | 0.538729171519 | 0.003849970571 | -0.007459538919 | 0.003979635008 | 0.000483050612 |
| LA | none | Urban | 0.674138526563 | 0.722231793156 | 0.092866030751 | 0.572312007378 | 0.008275262261 | 0.000474973733 | 0.006410508824 | 0.005562225015 |
| LA | none | Rural | 0.524711326844 | 0.644912479105 | 0.101297596293 | 0.476088926412 | -0.005480489920 | -0.017192422327 | 0.001163430695 | -0.007665285519 |
| LA | d4 | Overall | 0.632816347176 | 0.692574841966 | 0.096993017762 | 0.543579229730 | 0.003475699665 | -0.008723127813 | 0.003233103225 | -0.000232467866 |
| LA | d4 | Urban | 0.679602117807 | 0.723374196863 | 0.091247227392 | 0.575062763441 | 0.007812841155 | -0.000802186929 | 0.005472828955 | 0.004760330290 |
| LA | d4 | Rural | 0.533295245209 | 0.653600089534 | 0.104055465376 | 0.483538742540 | -0.005999651461 | -0.018948509007 | 0.000675837407 | -0.008549210951 |
| LC | none | Overall | 0.616086523891 | 0.676094270910 | 0.090201014921 | 0.528911746203 | -0.006206240316 | -0.019121568987 | -0.002473919253 | -0.009334374705 |
| LC | none | Urban | 0.666566023506 | 0.710730332495 | 0.090145973586 | 0.564531306219 | 0.000702759204 | -0.011026486928 | 0.003690451659 | -0.002218476145 |
| LC | none | Rural | 0.507292328360 | 0.631111069320 | 0.090271025576 | 0.461033690091 | -0.022899488404 | -0.030993832113 | -0.009863140022 | -0.022720521840 |
| LC | d4 | Overall | 0.621721090417 | 0.680256734187 | 0.090643327087 | 0.533066230882 | -0.007619557093 | -0.021041235592 | -0.003116587450 | -0.010745466714 |
| LC | d4 | Urban | 0.671008147701 | 0.713616796166 | 0.089906100442 | 0.567570332789 | -0.000781128951 | -0.010559587626 | 0.004131702005 | -0.002732100362 |
| LC | d4 | Rural | 0.514648818148 | 0.636340005189 | 0.091587502961 | 0.466543911223 | -0.024646078521 | -0.036208593352 | -0.011792125009 | -0.025544042268 |

## A1-i s0 paired baseline

| Mode | Domain | IoU | clDice | BIoU | A |
| --- | --- | --- | --- | --- | --- |
| none | Overall | 0.622292764207 | 0.695215839897 | 0.092674934174 | 0.538246120907 |
| none | Urban | 0.665863264302 | 0.721756819423 | 0.086455521927 | 0.566749782363 |
| none | Rural | 0.530191816764 | 0.662104901432 | 0.100134165598 | 0.483754211931 |
| d4 | Overall | 0.629340647510 | 0.701297969779 | 0.093759914537 | 0.543811697596 |
| d4 | Urban | 0.671789276652 | 0.724176383792 | 0.085774398436 | 0.570302433151 |
| d4 | Rural | 0.539294896669 | 0.672548598540 | 0.103379627969 | 0.492087953491 |

Observed LC-R1 wording follows the preregistered qualitative branch and is not independent evidence establishing a sole causal source of degradation. Observed LC also falls below A1-i s0.
