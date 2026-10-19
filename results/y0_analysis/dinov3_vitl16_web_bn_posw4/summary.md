# Frozen-probe y0 error analysis

Encoder: `dinov3_vitl16_web`; head: `bn_linear`; checkpoint: `dinov3_vitl16_web_bn_posw4.pt`; threshold: 0.5.
Images: 1669; seed: 0; device: cuda.
Full-Val regression against A=0.526507: passed.

A limited run is a workflow check, not evidence about the full Val split.

## Official pooled scores

| Group | Images | IoU | clDice | BIoU | A |
| --- | --- | --- | --- | --- | --- |
| Urban | 677 | 0.664708 | 0.705534 | 0.084496 | 0.560913 |
| Rural | 992 | 0.522136 | 0.619433 | 0.095428 | 0.465984 |
| Overall | 1669 | 0.617080 | 0.666980 | 0.089365 | 0.526507 |

Per-image scores and pooled scores use official image_counts/summarize. Overall Counts = Urban + Rural.
The official boundary metric additionally excludes the image frame and 3-pixel neighborhoods of ignored regions
(metrics.py:44-49); the descriptive geometry statistics below include all valid pixels.

## Pixel errors by Euclidean distance

Fractions use FP+FN in the same group as denominator. Bin labels describe ceil(distance):
(0,2], (2,4], (4,8], (8,16], (16,infinity). No rounding is applied to EDT values.
With no reference road, all relevant error pixels enter >16. Ignored pixels are excluded
as sources and observations; straight-line EDT distances can cross ignored regions.

| Group | Distance | FP pixels | FP / all errors | FN pixels | FN / all errors |
| --- | --- | --- | --- | --- | --- |
| Urban | 1-2 | 1363340 | 0.063296 | 1340788 | 0.062249 |
| Urban | 3-4 | 1125093 | 0.052235 | 1133905 | 0.052644 |
| Urban | 5-8 | 1499924 | 0.069637 | 1620038 | 0.075214 |
| Urban | 9-16 | 1345985 | 0.062490 | 1545674 | 0.071761 |
| Urban | >16 | 5948309 | 0.276164 | 4616005 | 0.214309 |
| Rural | 1-2 | 1031543 | 0.066987 | 1001350 | 0.065027 |
| Rural | 3-4 | 817273 | 0.053073 | 873588 | 0.056730 |
| Rural | 5-8 | 955533 | 0.062051 | 1267551 | 0.082313 |
| Rural | 9-16 | 598103 | 0.038840 | 1209756 | 0.078560 |
| Rural | >16 | 2188824 | 0.142140 | 5455551 | 0.354278 |
| Overall | 1-2 | 2394883 | 0.064835 | 2342138 | 0.063407 |
| Overall | 3-4 | 1942366 | 0.052584 | 2007493 | 0.054347 |
| Overall | 5-8 | 2455457 | 0.066475 | 2887589 | 0.078174 |
| Overall | 9-16 | 1944088 | 0.052631 | 2755430 | 0.074596 |
| Overall | >16 | 8137133 | 0.220291 | 10071556 | 0.272660 |

## Uncovered skeleton segments

Zhang skeletons and 8-connectivity are used. Dilate each uncovered segment by a 3x3 square;
count 8-connected contact regions on covered skeleton: >=2 = gap, 1 = terminal_missing, 0 = whole_missed.
Lengths count skeleton pixels, not Euclidean arc length. These are geometric proxies, not verified road identities.

| Group | Type | Length | Segments | Skeleton pixels |
| --- | --- | --- | --- | --- |
| Urban | gap | <=8 | 893 | 3012 |
| Urban | gap | 9-16 | 638 | 7880 |
| Urban | gap | 17-32 | 851 | 19672 |
| Urban | gap | 33-64 | 483 | 21391 |
| Urban | gap | >64 | 507 | 110069 |
| Urban | terminal_missing | <=8 | 674 | 2897 |
| Urban | terminal_missing | 9-16 | 362 | 4507 |
| Urban | terminal_missing | 17-32 | 322 | 7464 |
| Urban | terminal_missing | 33-64 | 220 | 9912 |
| Urban | terminal_missing | >64 | 420 | 100335 |
| Urban | whole_missed | <=8 | 73 | 139 |
| Urban | whole_missed | 9-16 | 22 | 281 |
| Urban | whole_missed | 17-32 | 33 | 764 |
| Urban | whole_missed | 33-64 | 40 | 1874 |
| Urban | whole_missed | >64 | 128 | 36874 |
| Rural | gap | <=8 | 1466 | 5262 |
| Rural | gap | 9-16 | 1095 | 13366 |
| Rural | gap | 17-32 | 1217 | 28525 |
| Rural | gap | 33-64 | 799 | 36053 |
| Rural | gap | >64 | 906 | 186267 |
| Rural | terminal_missing | <=8 | 420 | 1751 |
| Rural | terminal_missing | 9-16 | 291 | 3635 |
| Rural | terminal_missing | 17-32 | 277 | 6437 |
| Rural | terminal_missing | 33-64 | 198 | 9184 |
| Rural | terminal_missing | >64 | 498 | 121013 |
| Rural | whole_missed | <=8 | 29 | 71 |
| Rural | whole_missed | 9-16 | 10 | 114 |
| Rural | whole_missed | 17-32 | 12 | 302 |
| Rural | whole_missed | 33-64 | 28 | 1407 |
| Rural | whole_missed | >64 | 166 | 67426 |
| Overall | gap | <=8 | 2359 | 8274 |
| Overall | gap | 9-16 | 1733 | 21246 |
| Overall | gap | 17-32 | 2068 | 48197 |
| Overall | gap | 33-64 | 1282 | 57444 |
| Overall | gap | >64 | 1413 | 296336 |
| Overall | terminal_missing | <=8 | 1094 | 4648 |
| Overall | terminal_missing | 9-16 | 653 | 8142 |
| Overall | terminal_missing | 17-32 | 599 | 13901 |
| Overall | terminal_missing | 33-64 | 418 | 19096 |
| Overall | terminal_missing | >64 | 918 | 221348 |
| Overall | whole_missed | <=8 | 102 | 210 |
| Overall | whole_missed | 9-16 | 32 | 395 |
| Overall | whole_missed | 17-32 | 45 | 1066 |
| Overall | whole_missed | 33-64 | 68 | 3281 |
| Overall | whole_missed | >64 | 294 | 104300 |

| Group | GT skeleton pixels | Uncovered pixels | Uncovered fraction |
| --- | --- | --- | --- |
| Urban | 1066749 | 327071 | 0.306605 |
| Rural | 988077 | 480813 | 0.486615 |
| Overall | 2054826 | 807884 | 0.393164 |

## False-positive fragments

A predicted 8-connected component is a fragment only if it never touches GT dilated by a radius-2 Euclidean disk.
Dilation outputs are clipped to valid pixels. Area quantiles pool fragment components, not per-image averages.

| Group | Fragments | Area | Area / FP | Min | P10 | P25 | P50 | P75 | P90 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 5630 | 3496124 | 0.309867 | 1.000000 | 7.000000 | 22.000000 | 78.000000 | 278.000000 | 969.000000 | 62266.000000 |
| Rural | 3898 | 1754885 | 0.313861 | 1.000000 | 5.000000 | 17.000000 | 55.500000 | 146.000000 | 361.000000 | 74273.000000 |
| Overall | 9528 | 5251009 | 0.311191 | 1.000000 | 6.000000 | 20.000000 | 67.000000 | 206.250000 | 691.300000 | 74273.000000 |

## Component counts

Counts sum tile-local components; they are not a stitched geographic graph.

| Group | GT components | Predicted components |
| --- | --- | --- |
| Urban | 1196 | 9388 |
| Rural | 1108 | 9777 |
| Overall | 2304 | 19165 |

## Per-image A distribution

| Group | Min | P10 | P25 | P50 | P75 | P90 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | 0.000000 | 0.000000 | 0.314182 | 0.539446 | 0.669975 | 0.734305 | 1.000000 |
| Rural | 0.000000 | 0.000000 | 0.052660 | 0.413999 | 0.654691 | 1.000000 | 1.000000 |
| Overall | 0.000000 | 0.000000 | 0.132606 | 0.488358 | 0.664719 | 1.000000 | 1.000000 |

## Empty-road cases

| Group | No GT road | No predicted road | No GT but predicted road | FP pixels in those images |
| --- | --- | --- | --- | --- |
| Urban | 96 | 44 | 61 | 859368 |
| Rural | 314 | 202 | 141 | 765949 |
| Overall | 410 | 246 | 202 | 1625317 |

## Lowest-A images (up to 10 per domain)

| Group | Filename | IoU | clDice | BIoU | A |
| --- | --- | --- | --- | --- | --- |
| Urban | 3521.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3529.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3570.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3577.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3580.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3592.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3613.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3618.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3625.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Urban | 3668.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2524.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2526.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2530.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2531.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2533.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2538.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2541.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2545.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2551.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Rural | 2552.png | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Timing and self-checks

| Quantity | Seconds |
| --- | --- |
| forward_total | 64.839843 |
| forward_mean_per_image | 0.038850 |
| analysis_worker_total | 360.932061 |
| analysis_mean_per_image | 0.216256 |
| official_metric_worker_total | 139.252350 |
| wall_including_loading_and_visualization | 165.382098 |

All per-image distance-bin and uncovered-skeleton partition checks passed.
NA/null denotes an empty denominator or distribution. Empty-road official scores follow the official definitions.
Visualizations remain under ignored runs/; no probabilities or features are saved.
Panels show the original on the left and an overlay on the right: TP green, FP red, FN blue, ignored gray.
