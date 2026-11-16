# Observed convolutional draft-head results

Variant: A1-i; layers: last; trainable parameters: 656865; best epoch: 6.

Observed selection uses only fixed seed-0 Val-sel samples and A at threshold 0.5. The best head is evaluated once on Val-rep. Overall sums Urban and Rural Counts before official summarize. CPU smoke limits apply independently to Train, Val-sel and Val-rep; they do not adjudicate preregistered predictions.

| Domain | IoU | clDice | BIoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.665863264 | 0.721756819 | 0.086455522 | 0.566749782 |
| Rural | 0.530191817 | 0.662104901 | 0.100134166 | 0.483754212 |
| Overall | 0.622292764 | 0.695215840 | 0.092674934 | 0.538246121 |

| Epoch | Train loss | Train seconds | Val-sel A | New best |
|---|---:|---:|---:|---|
| 1 | 0.301798398 | 162.486770 | 0.541088028 | True |
| 2 | 0.159016102 | 161.706558 | 0.532993089 | False |
| 3 | 0.135914346 | 161.692179 | 0.529287784 | False |
| 4 | 0.127168269 | 162.428337 | 0.537157808 | False |
| 5 | 0.125668067 | 161.391066 | 0.529072459 | False |
| 6 | 0.120260023 | 160.948048 | 0.543394425 | True |
| 7 | 0.116338869 | 160.941660 | 0.530161436 | False |
| 8 | 0.115247263 | 162.171494 | 0.529126820 | False |
| 9 | 0.111645558 | 162.470847 | 0.532422987 | False |
| 10 | 0.109710850 | 160.724569 | 0.521990667 | False |
