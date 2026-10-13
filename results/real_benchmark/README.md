# Real-label structured benchmark

The committed pilot used N=10 road-positive **Train** masks per domain, seed 523. It reads the shared masks through EC523_DATA_DIR without copying them. It never reads Val. Official metrics are imported through EC523_STARTER_DIR, and image Counts are added before summarize.

## Reproduce and extend

Activate the project's ignored local env.sh, which should load the installed environment and set EC523_DATA_DIR and EC523_STARTER_DIR. No package installation or download is needed.

```bash
source env.sh
python analysis/real_benchmark.py --n 10 --seed 523 \
  --output results/real_benchmark/pilot_rerun
```

Choose a new output directory for every run; nonempty directories are rejected. Optional derived PNG examples must be under ignored tmp/:

```bash
python analysis/real_benchmark.py --n 10 --seed 523 \
  --output results/real_benchmark/pilot_with_examples \
  --examples tmp/real_benchmark_examples_rerun
```

The pilot took **310.45 seconds** for 20 masks on a single CPU process. A linear projection for N=100 per domain is about **3104.5 seconds (51.7 minutes)**. Dataset composition and node load can change runtime, so jobs/real_benchmark.qsub requests one CPU slot, 4G per core, **h_rt=02:00:00**, and no GPU. It sources env.sh as its first executable command and writes to a fresh results/real_benchmark/n100_jobJOBID directory. The script has not been submitted.

Submit from the repository root yourself:

```bash
qsub jobs/real_benchmark.qsub
```

The N=10 numeric tables and report are in pilot_n10/. Local example PNGs and runtime logs are under ignored tmp/, and are not committed. Do not add dataset-derived PNGs to this repository.

## Preliminary observations

| Condition | Real pilot: pooled All quality drop | Previous synthetic N=20 quality drop |
| --- | ---: | ---: |
| One-pixel outward offset | 0.103481 | 0.191755 |
| One-pixel inward offset | 0.104246 | Not measured |
| Gap L=8 | 0.011662 | 0.002650 |
| Gap L=16 | 0.018936 | 0.005049 |
| Gap L=32 | 0.033507 | 0.009851 |
| Gap L=64 | 0.062686 | 0.019469 |

Observed the same ordering as the prior synthetic scan: the one-pixel outward boundary offset reduced the pooled quality more than the tested gaps. The inward one-pixel offset had a similar effect in this pilot. This does not imply that offsets are more harmful to road-network use: global boundary offsets modify many more pixels than at most three localized gaps, and the official quality does not directly measure route connectivity. The synthetic geometry and its road density/widths also differ from the selected real masks.

For All, observed quality drops per 1000 changed valid pixels were 0.001330 for outward offset, 0.001337 for inward offset, and 0.001094/0.000888/0.000785/0.000735 for L=8/16/32/64. These pooled normalizations are descriptive, nonlinear and sample-size dependent. They are not per-pixel causal contributions. The changed-area ordering differs from the original synthetic benchmark: its N=20 outward-offset drop per 1000 changed valid pixels was 0.000539, compared with 0.000625/0.000595/0.000581/0.000574 for gaps L=8/16/32/64. In this real pilot the outward offset has the larger normalized drop. Thus the absolute-drop ordering agrees, but the normalized ordering does not.

See synthetic_comparison.csv. The synthetic comparison uses the previously committed N=20 quality drops and regenerates the same seed-523, 1024x1024 synthetic masks in memory with the shared helpers to count valid changed pixels. No dataset masks were used for this regeneration. Differences in road density, widths and placement eligibility prevent a causal cross-dataset interpretation.

Observed 52 candidate sites per length across 18 of 20 masks. L=8 and 16 had 52 valid events under both neighborhood definitions; L=32 and 64 had 51 valid plus one missing-anchor event. No bypassed event occurred in this small, conservatively selected pilot. Unit tests explicitly cover both a global cycle outside the local window (eligible) and a local bypass inside it (excluded).

Observed closing r=8 restored local eight-neighbor connectivity for all 52 L=8 gaps, but only 1/52 L=16 gaps and 1/51 L=32 or 64 gaps. Closing on clean masks also introduced two component merges at r=8. A restored local path can be a new detour rather than faithful reconstruction of the original removed road.

Observed the direction-filtered endpoint control repaired few pilot gaps (e.g. 1/52 at L=16,D=32 under eight-neighbor connectivity), and introduced no inter-component merges on these clean samples. Centerline endpoint tangents on real cut shapes can be complex; this is a deliberately simple geometric control, not a learned model. The false-connection count excludes shortcuts within an already connected component.

Bridge search found five artificial bridges in only two of the 20 images; 18 images were skipped. Many sampled masks have only one road component, or no accepted nearby pair under the distance/width/ignore/third-component guards. Skips are retained as unchanged predictions in headline scores and explicitly recorded. Likewise, two images had no eligible gap sites.

The pilot is a feasibility check without uncertainty intervals. N=100 has not been run; no claim is made about Val performance or final model selection.

## Refactor verification

The original and final shared-code versions were run at N=2, seed 523, 1024x1024. metrics.csv, gap_repair.csv and clean_false_bridges.csv matched byte-for-byte; all accumulated official Counts, seeded geometry, error events and repair outcomes matched exactly. Timing/output-directory fields naturally differed. The final synthetic rerun took approximately 14 seconds. Seven analytical shared-geometry tests passed, including local/global cycles, four/eight connectivity, ignored-pixel protection, component bridging and nested offsets.
