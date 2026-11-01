# EC523 midway submission: frozen DINOv3 and looped road refiner

This package performs image-only, offline inference. It never opens annotation
files. Submit one checkpoint for every loop count; do not ensemble the seeds.
The selected model is v2 seed 0, best epoch 7, trained with four supervised
updates. Input images must be RGB-compatible 1024 x 1024 tiles. They are not
resized, cropped or augmented.

## Environment

Use Python 3.12 (development used 3.12.4). Prepare the environment before
disconnecting the machine from the network. For CUDA 12.4:

```bash
mkdir -p .cache/pip .cache/tmp .cache/xdg .cache/torch .cache/hf
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.cache/tmp"
export XDG_CACHE_HOME="$PWD/.cache/xdg" TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/hf"
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For CPU, install the CPU-compatible torch/torchvision wheels of the same
versions instead. The fixed requirements include transformers 4.57.6, which
provides DINOv3 support. No package or weight downloads occur at inference.
Run commands from the package root; no SCC modules, personal paths or tokens
are needed. The scripts locate `src/` relative to their own files.

## Required files

Distribute the following layout, preserving the Python source tree. Do not
include the virtual environment, caches, datasets, training checkpoints or
course-owned starter files.

```text
requirements.txt
src/looproad/                 # project Python sources; exclude __pycache__
submission/
  __init__.py
  model.py
  baseline.py
  predict.py
  README.md
  model.safetensors          # exported refiner parameters and model buffers
  baseline.safetensors       # exported BN probe parameters and running stats
  encoder/
    config.json
    preprocessor_config.json
    model.safetensors        # frozen facebook/dinov3-vitl16-pretrain-lvd1689m
```

The encoder files must be actual readable files, not dangling symlinks into a
developer's HF cache. The input argument `--encoder-dir` is an explicit local
directory: it is never interpreted as a Hugging Face repository ID. Loading
uses `local_files_only=True`. Normalization mean/std are read from the supplied
preprocessor configuration; the image processor is never applied to images.

The author prepares the two small files using trusted local training artifacts:

```bash
python scripts/export_submission.py
```

This exports `runs/exp/e0_v2_s0/best.pt` and
`runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt`, with architecture
metadata, to `submission/`. Optimizer, scheduler, histories and RNG states are
excluded. Deterministic position encoding and BN buffers remain because strict
model loading requires them. Exports are ignored by git and are distributed
separately. The preparation tool refuses to overwrite existing exports.

## Inference commands

Set the offline environment for every invocation:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Directory input, preserving subdirectories in prediction IDs:

```bash
python submission/predict.py --checkpoint submission/model.safetensors \
  --baseline submission/baseline.safetensors --encoder-dir submission/encoder \
  --input images --output predictions/k4 --steps 4 --device cuda --save-trace
```

A single image uses its filename stem as ID:

```bash
python submission/predict.py --checkpoint submission/model.safetensors \
  --baseline submission/baseline.safetensors --encoder-dir submission/encoder \
  --input images/urban/001.png --output predictions/single --steps 4 --device cpu
```

Run the same weights at K=1,2,4; each output directory must be new or empty:

```bash
for k in 1 2 4; do
  python submission/predict.py --checkpoint submission/model.safetensors \
    --baseline submission/baseline.safetensors --encoder-dir submission/encoder \
    --input images --output "predictions/k${k}" --steps "$k" --device cuda
done
```

Diagnostic interventions, without changing or training weights:

```bash
for mode in reset-state no-mask-feedback; do
  python submission/predict.py --checkpoint submission/model.safetensors \
    --baseline submission/baseline.safetensors --encoder-dir submission/encoder \
    --input images --output "predictions/${mode}" --steps 8 --device cuda \
    --save-trace --ablation "$mode"
done
```

`--steps` accepts 0 through 8. K=0 returns the unchanged frozen baseline.
Only the CLI owns K; the model methods never receive it. `--seed` defaults to
0, `--device` to cpu and `--ablation` to none. Unlike the starter demo, this
entry point requires trained weights and has no `--demo` option.

Reset-state reruns the learned initializer on the original features/logits
before every update. No-mask-feedback supplies the original logits as the
step's previous-logits argument at every update. Because this argument is
also the residual base, that intervention produces y0 + delta_t at each step;
the hidden state still carries forward. Diagnostic ablations are not official
leaderboard entries.

## Outputs and validation

For directory input `images/urban/001.png`:

```text
predictions/k4/
  probabilities/urban/001.npy  # finite float32 HxW probabilities in [0,1]
  masks/urban/001.png          # uint8 0/255, probability >= 0.5
  trace/k0/urban/001.npy       # only with --save-trace; all k0 ... k4
  run.json
```

The evaluator consumes probabilities, not PNGs or logits. The final probability
map equals trace/kK when traces are requested. Duplicate relative IDs, unsupported
or empty inputs, nonfinite logits and occupied output directories are rejected.
The core is counted and parameter identity/storage is checked at every update.

`run.json` retains the starter protocol identifier and all starter field names.
It adds encoder file/bundle hashes, image count and per-image peak allocated
CUDA bytes. `baseline_sha256` hashes the explicit probe safetensors file;
encoder hashes separately identify the remaining frozen baseline. The source
hash covers submission and reused project Python sources. CPU peak CUDA memory
is null. Inference times include encoder/probe, updates and every-step CPU trace
transfer, but exclude model loading and file writes; there is no warmup.

The course `evaluate.py` requires both a probability directory and run.json.
The full check job creates documented trace-prefix views for k1/k2/k4/k8,
retaining source K8 call counts/timings separately, then invokes the unchanged
official evaluator. Trace-prefix timing is not claimed as standalone latency.

## Model and training provenance

The baseline uses frozen DINOv3 ViT-L/16 web features [B,1024,64,64], then a
frozen BatchNorm2d and 1x1 probe with bilinear upsampling. The float32 refiner
projects image tokens and 16x16 sigmoid-answer patches to width 256, adds fixed
2D positions, and repeatedly calls one shared pre-LN eight-head transformer
block. A smooth convolutional upsampling head predicts an additive dense
correction. The 1,193,025 student parameters are reused for every update.

The selected refiner was trained at git commit
`5e931314fb6bd52fb02a81ffc7faafe0bc9bcd79` using:

```bash
PYTHONPATH="$PWD/src" python scripts/train_refiner.py \
  --head-type conv_upsample --epochs 10 --batch-size 4 --lr 3e-4 \
  --k-train 4 --seed 0 --device cuda --select-every 1 --select-per-domain 100 \
  --selection-ks 1:0.25,2:0.25,4:0.5 --val-split configs/val_split_seed0.json \
  --eval-extra --eval-k 1,2,4,6,8 \
  --out-dir results/exp/e0_v2_s0 --ckpt-dir runs/exp/e0_v2_s0
```

Other saved settings: AdamW weight decay 0.01, BCE positive weight 4,
500-step linear warmup followed by cosine decay, gradient clipping 1.0,
width 256, one transformer block, eight heads, MLP ratio 4 and patch 16.
Checkpoint selection uses Val-sel only; Val-rep is a disjoint report split.
The selected checkpoint is epoch 7, rather than the final epoch 10.

## Code sources and full acceptance job

The new submission adapters and CLI are student-written. They reuse this
project's `src/looproad/models/refiner.py`, `models/linear_probe.py` and
`encoders/huggingface.py`; PyTorch supplies SDPA/torchvision and Hugging Face
Transformers loads the explicit local DINOv3 weights. Encoder source:
`facebook/dinov3-vitl16-pretrain-lvd1689m`.

Behavior is aligned to the read-only course starter `predict.py:40-78,83-106,
129-154`, `baseline.py:1-6`, and `README.md:45-66,98-114`. No course starter
implementation is copied or imported by inference. Existing project code was
developed with AI coding assistance and reviewed through the listed tests;
team members remain responsible for understanding submitted code.

For the author's full SCC acceptance check, pass the explicit local encoder
directory through SGE's `-v` option (an exported login-shell variable alone
is not sufficient):

```bash
mkdir -p runs/logs
qsub -v EC523_ENCODER_DIR="$PWD/submission/encoder" jobs/submission_check.qsub
```

Run this manually from the repository root. The job uses 835 read-only
Val-rep image symlinks, no HF cache, two K8 trajectories (none/reset-state),
all traces and ten official evaluation reports. Reserve at least 80 GiB of
free/quota space; NPY files alone need approximately 65.3 GiB. Do not include
these check outputs or the job log in the submission package.
