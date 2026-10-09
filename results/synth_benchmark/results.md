# Structured synthetic road benchmark

Date: 2026-10-05. Seed: 523. Images: 20. Size: 1024 x 1024. CPU elapsed: 133.6 seconds.

Official image_counts and summarize are imported through sys.path. Integer Counts are summed before summarization. All self-scores passed exactly 1.0 before experiments. Drops are clean score minus condition score. No image-score averaging is used.

## Design and assumptions

- Roads are axis-aligned intersecting networks with mixed nominal widths 4-8 and 12-20 pixels, plus three isolated parallel pairs. Seeded geometry and widths vary across images. Ignore rectangles occur every third seed and may intersect roads. Purple denotes ignore in visualizations. Raw labels and detailed metadata are excluded from this repository; they can be regenerated locally.
- Zhang centerlines, 8-neighbor endpoint/junction degrees, and twice the foreground Euclidean distance estimate local width. Junction neighborhoods are removed to summarize skeleton segments. Raster width estimates can differ from nominal drawing widths. Locally generated metadata records endpoints, junction pixels, segment width summaries, drawing coordinates, all error locations and repair outcomes.
- Three nested gaps per image reuse centers and PCA tangents across lengths. Centers are away from junctions. Cuts span the full estimated width; anchors lie four pixels beyond each gap. Asserted disconnected anchors before closing prevent alternate-route false repair. Connectivity uses 8-neighbor components after masking ignore pixels.
- Bridges connect designated pairs in separate original components, using their nominal drawing width. Closing uses Euclidean disk footprints. New clean bridges are reported both for designated pairs and as mergers among all original connected components.
- Proportional thickening adds an outward radius equal to fraction times estimated full local width, taken from the nearest original centerline pixel. This increases full width by approximately twice that radius; it is not a fraction of half-width.
- Clean ignore invariance is verified by official self-scoring; every prediction is masked by the official evaluator. Closing also masks ignore pixels before connectivity checks.

The synthetic masks contain straight, uniform-width, high-contrast geometry without image evidence, curved roads, occlusions, grade-separated crossings, complex junctions, or real segmentation noise. Observations may change with those structures, pixel scale, road density, or errors on cyclic networks. Gap component checks here intentionally use terminal branches.

## Reproduction

Set `EC523_STARTER_DIR="<YOUR_STARTER_DIR>/cli"`, activate your environment, and run `python analysis/synth_benchmark.py` from the repository root. Raw outputs go to ignored `tmp/synth_benchmark/`. Configure `--n`, `--seed`, `--size`, `--lengths`, `--radii`, and `--output` as needed. Default parameters match the requested scan. Results are descriptive pooled scores, not statistical estimates of performance on real roads.

## Clean self-score

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Gap sensitivity

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap_4 | 0.998537 | 0.999326 | 0.997417 | 0.998550 | 0.001463 | 0.000674 | 0.002583 | 0.001450 |
| gap_8 | 0.997074 | 0.998651 | 0.996089 | 0.997350 | 0.002926 | 0.001349 | 0.003911 | 0.002650 |
| gap_16 | 0.994148 | 0.997299 | 0.993435 | 0.994951 | 0.005852 | 0.002701 | 0.006565 | 0.005049 |
| gap_32 | 0.988297 | 0.994583 | 0.988125 | 0.990149 | 0.011703 | 0.005417 | 0.011875 | 0.009851 |
| gap_64 | 0.976594 | 0.989107 | 0.977507 | 0.980531 | 0.023406 | 0.010893 | 0.022493 | 0.019469 |

## Bridge sensitivity

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bridge_1 | 0.999162 | 0.999414 | 0.998804 | 0.999166 | 0.000838 | 0.000586 | 0.001196 | 0.000834 |
| bridge_3 | 0.997531 | 0.998356 | 0.996489 | 0.997570 | 0.002469 | 0.001644 | 0.003511 | 0.002430 |

## Uniform thickening

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thicken_k_1 | 0.803023 | 1.000000 | 0.533667 | 0.808245 | 0.196977 | 0.000000 | 0.466333 | 0.191755 |
| thicken_k_2 | 0.671014 | 0.999882 | 0.212258 | 0.677923 | 0.328986 | 0.000118 | 0.787742 | 0.322077 |
| thicken_k_3 | 0.576161 | 0.988710 | 0.002265 | 0.585146 | 0.423839 | 0.011290 | 0.997735 | 0.414854 |

## Proportional thickening

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thicken_fraction_0.25 | 0.679925 | 1.000000 | 0.277891 | 0.695541 | 0.320075 | 0.000000 | 0.722109 | 0.304459 |
| thicken_fraction_0.5 | 0.493522 | 0.967029 | 0.031302 | 0.543130 | 0.506478 | 0.032971 | 0.968698 | 0.456870 |

## Closing damaged roads

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap_4_close_1 | 0.998054 | 0.999326 | 0.996241 | 0.998073 | 0.001946 | 0.000674 | 0.003759 | 0.001927 |
| gap_4_close_2 | 0.998053 | 0.999933 | 0.994539 | 0.997914 | 0.001947 | 0.000067 | 0.005461 | 0.002086 |
| gap_4_close_4 | 0.919334 | 0.944726 | 0.907603 | 0.924605 | 0.080666 | 0.055274 | 0.092397 | 0.075395 |
| gap_4_close_8 | 0.753637 | 0.788057 | 0.740145 | 0.761265 | 0.246363 | 0.211943 | 0.259855 | 0.238735 |
| gap_8_close_1 | 0.996591 | 0.998651 | 0.994915 | 0.996874 | 0.003409 | 0.001349 | 0.005085 | 0.003126 |
| gap_8_close_2 | 0.995627 | 0.998651 | 0.992330 | 0.995875 | 0.004373 | 0.001349 | 0.007670 | 0.004125 |
| gap_8_close_4 | 0.918018 | 0.944064 | 0.904322 | 0.923093 | 0.081982 | 0.055936 | 0.095678 | 0.076907 |
| gap_8_close_8 | 0.753263 | 0.787969 | 0.739004 | 0.760823 | 0.246737 | 0.212031 | 0.260996 | 0.239177 |
| gap_16_close_1 | 0.993667 | 0.997299 | 0.992262 | 0.994476 | 0.006333 | 0.002701 | 0.007738 | 0.005524 |
| gap_16_close_2 | 0.992706 | 0.997299 | 0.989681 | 0.993479 | 0.007294 | 0.002701 | 0.010319 | 0.006521 |
| gap_16_close_4 | 0.914257 | 0.941827 | 0.902256 | 0.920128 | 0.085743 | 0.058173 | 0.097744 | 0.079872 |
| gap_16_close_8 | 0.750039 | 0.785008 | 0.734307 | 0.757383 | 0.249961 | 0.214992 | 0.265693 | 0.242617 |
| gap_32_close_1 | 0.987818 | 0.994583 | 0.986957 | 0.989675 | 0.012182 | 0.005417 | 0.013043 | 0.010325 |
| gap_32_close_2 | 0.986863 | 0.994583 | 0.984383 | 0.988683 | 0.013137 | 0.005417 | 0.015617 | 0.011317 |
| gap_32_close_4 | 0.908876 | 0.939053 | 0.896983 | 0.915551 | 0.091124 | 0.060947 | 0.103017 | 0.084449 |
| gap_32_close_8 | 0.745064 | 0.780173 | 0.729773 | 0.752539 | 0.254936 | 0.219827 | 0.270227 | 0.247461 |
| gap_64_close_1 | 0.976121 | 0.989107 | 0.976345 | 0.980062 | 0.023879 | 0.010893 | 0.023655 | 0.019938 |
| gap_64_close_2 | 0.975177 | 0.989107 | 0.973787 | 0.979078 | 0.024823 | 0.010893 | 0.026213 | 0.020922 |
| gap_64_close_4 | 0.898114 | 0.933461 | 0.886438 | 0.906383 | 0.101886 | 0.066539 | 0.113562 | 0.093617 |
| gap_64_close_8 | 0.736241 | 0.772920 | 0.719373 | 0.743871 | 0.263759 | 0.227080 | 0.280627 | 0.256129 |

## Per-gap connectivity repair

| length | radius | repaired | total | repair_rate |
| --- | --- | --- | --- | --- |
| 4 | 1 | 0 | 60 | 0.000000 |
| 4 | 2 | 48 | 60 | 0.800000 |
| 4 | 4 | 60 | 60 | 1.000000 |
| 4 | 8 | 60 | 60 | 1.000000 |
| 8 | 1 | 0 | 60 | 0.000000 |
| 8 | 2 | 0 | 60 | 0.000000 |
| 8 | 4 | 20 | 60 | 0.333333 |
| 8 | 8 | 48 | 60 | 0.800000 |
| 16 | 1 | 0 | 60 | 0.000000 |
| 16 | 2 | 0 | 60 | 0.000000 |
| 16 | 4 | 0 | 60 | 0.000000 |
| 16 | 8 | 5 | 60 | 0.083333 |
| 32 | 1 | 0 | 60 | 0.000000 |
| 32 | 2 | 0 | 60 | 0.000000 |
| 32 | 4 | 0 | 60 | 0.000000 |
| 32 | 8 | 0 | 60 | 0.000000 |
| 64 | 1 | 0 | 60 | 0.000000 |
| 64 | 2 | 0 | 60 | 0.000000 |
| 64 | 4 | 0 | 60 | 0.000000 |
| 64 | 8 | 0 | 60 | 0.000000 |

## Closing clean roads

| condition | road_iou | cldice | boundary_iou | quality | drop_road_iou | drop_cldice | drop_boundary_iou | drop_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_close_1 | 0.999516 | 1.000000 | 0.998821 | 0.999522 | 0.000484 | 0.000000 | 0.001179 | 0.000478 |
| clean_close_2 | 0.998549 | 1.000000 | 0.996228 | 0.998520 | 0.001451 | 0.000000 | 0.003772 | 0.001480 |
| clean_close_4 | 0.919639 | 0.944726 | 0.908661 | 0.924969 | 0.080361 | 0.055274 | 0.091339 | 0.075031 |
| clean_close_8 | 0.753887 | 0.788057 | 0.741085 | 0.761578 | 0.246113 | 0.211943 | 0.258915 | 0.238422 |

## New bridges on clean roads

| radius | connected_pairs | total_pairs | pair_rate | images_with_new_bridges | total_images | component_merges |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 60 | 0.000000 | 0 | 20 | 0 |
| 2 | 0 | 60 | 0.000000 | 0 | 20 | 0 |
| 4 | 22 | 60 | 0.366667 | 16 | 20 | 22 |
| 8 | 60 | 60 | 1.000000 | 20 | 20 | 60 |

Verification: same-seed labels match exactly; changing predictions only inside ignore pixels leaves official Counts unchanged. Proportional conditions were recomputed using centerline widths after correcting a boundary-width lookup. Recompute CPU time: 10.8 seconds.
