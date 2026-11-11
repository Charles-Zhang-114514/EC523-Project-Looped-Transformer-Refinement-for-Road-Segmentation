# Observed convolutional draft-head results

Variant: A1-ii; layers: 6,12,18,24; trainable parameters: 1449441; best epoch: 4.

Observed selection uses only fixed seed-0 Val-sel samples and A at threshold 0.5. The best head is evaluated once on Val-rep. Overall sums Urban and Rural Counts before official summarize. CPU smoke limits apply independently to Train, Val-sel and Val-rep; they do not adjudicate preregistered predictions.

| Domain | IoU | clDice | BIoU | A |
|---|---:|---:|---:|---:|
| Urban | 0.644908475 | 0.697852954 | 0.078761239 | 0.547562372 |
| Rural | 0.498076714 | 0.596839896 | 0.082653173 | 0.444620961 |
| Overall | 0.600481130 | 0.656030911 | 0.080381023 | 0.513126043 |

| Epoch | Train loss | Train seconds | Val-sel A | New best |
|---|---:|---:|---:|---|
| 1 | 0.288201711 | 176.518338 | 0.496176507 | True |
| 2 | 0.152113171 | 164.931676 | 0.500138585 | True |
| 3 | 0.128368082 | 170.158519 | 0.509289607 | True |
| 4 | 0.117409635 | 169.539981 | 0.519146664 | True |
| 5 | 0.119257820 | 169.134154 | 0.503462067 | False |
| 6 | 0.110594729 | 169.309666 | 0.515018373 | False |
| 7 | 0.105585123 | 168.266414 | 0.503829569 | False |
| 8 | 0.103424935 | 167.699952 | 0.505187907 | False |
| 9 | 0.100522834 | 166.212337 | 0.489010031 | False |
| 10 | 0.096416595 | 167.449625 | 0.494685686 | False |
