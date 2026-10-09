# EC523 Project — Looped Transformer Refinement for Road Segmentation

## Overview

We study road segmentation refinement by repeatedly applying one shared transformer update rule. We ask when additional iterations repair errors, when they fail, and why.

## Status

Early-stage project. The official encoder will be released after the midterm; each group currently selects its own encoder. Historical notes describe the earlier starter-package release.

## Setup

On BU SCC, copy `env.sh.example` to ignored `env.sh` and replace its placeholders. Obtain the course starter package and course data from the course staff; neither is included here. `EC523_STARTER_DIR` must point to the starter package's `cli/` directory.

```bash
module load python3/3.12.4
python -m venv env/venv
source env/venv/bin/activate
pip install -r requirements.txt
source env.sh
python analysis/synth_benchmark.py --n 2 --output tmp/synth_smoke
```

Alternatively, set `export EC523_STARTER_DIR="<YOUR_STARTER_DIR>/cli"` in your existing environment. Benchmark execution uses CPU only. Raw outputs stay in ignored `tmp/`.

## Repository structure

```text
src/looproad/
  data/          Data loading and preprocessing.
  encoders/      Encoder adapters.
  models/        Shared refinement models.
  train/         Training routines.
  eval/          Evaluation routines.
submission/      Course submission integration.
analysis/        Analysis and benchmark scripts.
scripts/         Command-line workflow helpers.
configs/         Experiment configurations.
jobs/            SCC job templates.
tests/           Automated checks.
results/         Selected small public result summaries and figures.
notes/           Project notes and work log.
```

## Current work

The structured synthetic benchmark measures the effects of road gaps, false bridges, and thickening, with morphological closing as a non-learning control. It uses official metrics from the external starter package; see [results and examples](results/synth_benchmark/results.md).

## Collaboration workflow

Merge into `main` through pull requests only. Develop on individual branches such as `name/topic`; each PR needs at least one teammate's review. Never commit datasets, features, weights, course starter code, PDFs, environments, caches, logs, or credentials. Keep only selected small summaries and figures in `results/`.

## Team

- Member 1: TBD
- Member 2: TBD
- Member 3: TBD

## AI usage disclosure

AI coding assistants helped write code and organize documentation. Team members are responsible for research decisions and review.

## Acknowledgments

Boston University · EC523 Deep Learning · Fall 2026
