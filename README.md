# EC523 Project — Looped Transformer Refinement for Road Segmentation

## Overview

We study road segmentation refinement by repeatedly applying one shared transformer update rule. We ask when additional iterations repair errors, when they fail, and why.

## Status

The repository includes data audits, structured-error benchmarks, and frozen-encoder linear-probe training/evaluation for ResNet-50, DINOv3, and SAM. Looped-refinement training remains future work. The official encoder is expected after the midterm; historical notes describe the earlier starter-package release.

## Setup on BU SCC

Use BU SCC's shared `fall-2026-pyt` Conda environment from the
[`academic-ml/fall-2026` module](https://www.bu.edu/tech/support/research/software-and-programming/common-languages/python/python-ml/academic-machine-learning-environment/).
Start in a fresh SCC Bash shell and run from the repository root:

```bash
source scripts/activate_scc.sh
python -m unittest discover -s tests -v
```

The tracked activation script loads `miniconda/25.3.1` and
`academic-ml/fall-2026`, then activates `fall-2026-pyt`. Interactive sessions
and all batch jobs use this same script. Commit changes to this script when
the team agrees to switch SCC environments.

This shared environment provides Python 3.13.5, PyTorch 2.13.0+cu132,
NumPy 2.5.2, Pillow 12.3.0, SciPy 1.18.1, scikit-image 0.26.0, and
safetensors 0.8.0 (checked on 2026-10-15). CPU validation on that date passed
the seven then-existing repository tests, all 27 course-starter tests, the complete four-run starter
demo/evaluation workflow, a two-image synthetic benchmark, and a real-label
benchmark with one Urban and one Rural Train mask. These are compatibility
checks, not training or final course performance results. GPU training has
not been validated in this shared environment.

The shared environment also includes torchvision 0.28.0+cu132,
Transformers 5.16.1, and opencv-python 5.0.0.93 (`cv2.__version__` is 5.0.0),
checked on 2026-10-19. OpenCV imported successfully, loaded an actual image
and mask identically to Pillow after BGR-to-RGB conversion, and preserved
raw label IDs 0 through 7 in a PNG round trip. The project's dataset reader
uses Pillow to retain raw labels; OpenCV availability does not require
changing that reader. Earlier notes about missing OpenCV do not apply to this shared
environment.

No private virtual environment or package installation is needed.
`requirements.txt` records the earlier course-starter package pins; those
versions differ from the shared SCC environment. Do not install that file
into the shared environment. Do not load a separate `python3` or `pytorch`
module alongside this Conda environment. Inspect the installed packages
with `conda list` when recording an experiment.

### Configure course inputs and local paths

The course archive `EC523_Project_A_Student.zip` contains a top-level
`project_a/` directory, including `cli/metrics.py` and `cli/common.py`.
The repository deliberately keeps this starter package external. If the ZIP
is stored beside your repository, extract it from the repository root with:

```bash
unzip -n ../EC523_Project_A_Student.zip -d ..
```

The activation script detects the repository root automatically, sets
`EC523_STARTER_DIR` to `project_a/cli` beside the repository, and defaults
`EC523_DATA_DIR` to `/projectnb/ec523kb/projects/RoadSeg/dataset`. This layout
works without creating an `env.sh` file. Temporary files and model caches
stay under ignored repository directories.

If your starter or dataset is elsewhere, copy `env.sh.example` to ignored
`env.sh`, uncomment the relevant `export` lines, and set your paths. Then run
`source scripts/activate_scc.sh` again; it reads these overrides automatically.
The script also accepts existing `EC523_DATA_DIR` and `EC523_STARTER_DIR`
environment variables. `EC523_PROJECT_DIR` always identifies this checkout.

An older `env.sh` may contain module loading and private virtual-environment
activation. Replace that setup with path-only overrides using the new template;
activate the project with `source scripts/activate_scc.sh` from now on.

Commit `scripts/activate_scc.sh`, `env.sh.example`, and the corresponding
README/job changes to share the setup. Keep personal `env.sh` files ignored.
Unit tests and the dataset audit do not need the starter package; both
benchmark scripts need it.

### Pretrained encoder assets

The new Hugging Face tests use small random-weight models but still require
cached `config.json` and `preprocessor_config.json` files for the four encoder
repositories named in `src/looproad/encoders/huggingface.py`. Training also
requires cached pretrained weights. ResNet training expects the ImageNet
checkpoint under `$TORCH_HOME/checkpoints`; Hugging Face assets are looked
up through `$HF_HOME`. Installing Python packages alone does not provide
these model assets. The encoders use local caches and do not download files
automatically. Keep configs, model weights, and generated checkpoints in
ignored caches or `runs/`, outside Git.

### Run CPU checks

After syncing PRs #3-#5 on 2026-10-19, 11 of 14 test methods passed in the
shared environment. The three Hugging Face test methods failed because this
checkout had no cached model/processor configs (four subtest failures and
five errors). This does not establish full encoder compatibility yet; rerun
the suite once the required metadata is cached. No models were downloaded
for this check.

```bash
python -m unittest discover -s tests -v

# Audit image/mask integrity and sample geometry from the shared dataset.
# This scans all images and masks; sample-size only limits geometry analysis.
python scripts/data_audit.py --sample-size 2 --output tmp/data_audit_smoke

# Check the extracted starter's interface, metrics, and demo workflow:
(
    cd "$EC523_STARTER_DIR" || exit 1
    python -m unittest discover -s tests -v &&
    python smoke_test.py --output "$EC523_PROJECT_DIR/tmp/starter_smoke"
)

# Requires EC523_STARTER_DIR to point to the course cli/ directory:
python analysis/synth_benchmark.py --n 2 --output tmp/synth_smoke
python analysis/real_benchmark.py --n 1 --output tmp/real_smoke
```

Use a fresh output directory for each starter demo and real benchmark run. Keep generated
outputs in ignored `tmp/`; publish only selected small summaries and figures.
For the full real benchmark, `jobs/real_benchmark.qsub` sources the tracked
`scripts/activate_scc.sh` and runs on CPU. Submit it from the repository root with
`qsub jobs/real_benchmark.qsub`.

The unit tests, data audit, and structured-error benchmarks run on CPU.
The linear-probe pipeline also supports small CPU pilots; the `probe_*.qsub`
files request one GPU for full training and validation. These jobs now use
the same shared activation script as the CPU workflow. For GPU training,
request a GPU through
SCC's scheduler with `-l gpus=1 -l gpu_c=8.0`, as required by
`module help academic-ml/fall-2026`. See
[BU's GPU guidance](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/).

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

PRs #3-#5 add frozen-encoder linear probes and comparisons of ResNet-50,
DINOv3 ViT-L (web and satellite), DINOv3 ConvNeXt-Small, and SAM ViT-B.
See `scripts/train_linear_probe.py` and `results/linear_probe/` for the
pipeline and recorded results. The DINOv3 ViT-L web model has the highest
recorded Overall A at threshold 0.5 (0.526507); these are single-seed Val
observations from the teammate's environment, not reruns in the shared SCC
environment.

## Collaboration workflow

Merge into `main` through pull requests only. Develop on individual branches such as `name/topic`; each PR needs at least one teammate's review. Never commit datasets, features, weights, course starter code, PDFs, environments, caches, logs, or credentials. Keep only selected small summaries and figures in `results/`.

## Team

- Member 1: Haoran Zhang zhr114@bu.edu
- Member 2: 
- Member 3: 

## AI usage disclosure

AI coding assistants helped write code and organize documentation. Team members are responsible for research decisions and review.

## Acknowledgments

Boston University · EC523 Deep Learning · Fall 2026
