CONFIRM CONTROL, LOVASZ, AND AUGMENTATION WITH SEEDS 1 AND 2

Run these commands from your SCC checkout with the completed seed-0 batch available.

Submit ONCE from an SCC Bash terminal:

cd "$(git rev-parse --show-toplevel)"
source scripts/activate_scc.sh
python scripts/repeat_refiner_batch.py submit

This submits one six-task array after input checks and tests pass:

  Task 1: control,      seed 1
  Task 2: Lovasz,       seed 1
  Task 3: augmentation, seed 1
  Task 4: control,      seed 2
  Task 5: Lovasz,       seed 2
  Task 6: augmentation, seed 2

It does not repeat the other screening methods or seed 0. Repeating the submit
command creates another six-task batch, so do not run it again to check status.

The reference is the completed seed-0 batch:
  runs/six_run/20261101_192457Z_25415913 (array job 8386879)
For another completed screening batch, pass --reference /absolute/path/to/batch.

Training starts from fresh seed-specific refiner initialization. The helper
copies the reference batch's frozen training source, validation split, and probe;
it does not load seed-0 refiner weights or use later edits to the working trainer.
It reuses each method's original training command, changing only --seed and the
two output directories. DINOv3 and the probe remain frozen. The dataset and DINO
cache are reused offline. Seed 0's reports are copied for the eventual comparison.

All runs keep ten epochs, batch size 4, lr=3e-4, K_train=4, BCE pos_weight=4,
the same shared refiner and conv_upsample head, fixed Val-sel/Val-rep membership,
and official threshold 0.5. Lovasz keeps auxiliary weight 0.1. Augmentation keeps
the same Train-only D4 transform. The split file still uses seed 0: changing
training seeds does NOT change validation membership.

Resources match the original batch: one L40S GPU, 8 CPU slots, 64 GB host RAM,
four-hour limit per task, and at most two tasks running simultaneously.
The runtime checks require the same GPU model, PyTorch/CUDA, and Python versions
as seed 0. No alternate GPU model is silently substituted.

The first runs of these methods took approximately 42-46 minutes per task,
including checks/training/evaluation. Six repeats should therefore require
roughly 4.3 GPU-hours and about 2-3 hours if two tasks can run continuously;
queue delays and runtime variation can make wall-clock time longer.
Four hours is the per-task limit, not the expected runtime.

To limit concurrency to one task, use this INSTEAD of the submission above:
  python scripts/repeat_refiner_batch.py submit --max-running 1

CHECK STATUS

qstat -u "$USER"
python scripts/repeat_refiner_batch.py status

qstat may combine queued array tasks into ranges. 'qw' means queued; 'r' means
running. The status helper lists all six methods/seeds separately. Completion
requires all six SUCCESS markers; disappearance from qstat alone is not proof.
A killed or timed-out job may remain marked STARTED, so inspect qstat and logs.

FOLLOW A LOG (task 2 is Lovasz seed 1)

batch_dir=$(cat runs/confirmation/LATEST)
tail -n 40 -F "$batch_dir/logs/task_2.log"

Replace 2 with the task number you want. The log may not exist until it starts.
Ctrl+C stops tail only; training continues. You may disconnect after submission.
For a shell/activation failure, also inspect scheduler output in the logs folder.

COMPARE AFTER ALL SIX FINISH

python scripts/repeat_refiner_batch.py compare

This combines the existing seed-0 results with the new seeds 1 and 2. It prints
and saves comparison.txt and comparison.json in the confirmation batch folder.
It reports per-seed values, three-seed mean +/- sample standard deviation (n-1),
and candidate-minus-control differences paired by seed. Incomplete methods are
listed explicitly and are not given a three-seed average.

Reported metrics include A at K=1,2,4,6,8; Urban/Rural A at K4; the weighted
K1/K2/K4 proxy; Urban total/new false fragments; Urban/Rural gap repair and new
coverage breaks. Positive A deltas favor the candidate; positive fragment/break
deltas are worse. No stress evaluation is included, so the proxy is not the full
course score. Three seeds provide repeatability evidence, not a significance
guarantee; Val-rep has already informed development.

OUTPUT LOCATIONS

  runs/confirmation/<timestamp>_<id>/
    manifest.json                         task definitions and input hashes
    reference/<method>/reports/metrics.json copied seed-0 evidence
    logs/task_1.log ... task_6.log
    source/runs/control_seed1/...
    source/runs/lovasz_seed1/...
    source/runs/augmentation_seed1/...
    source/runs/control_seed2/...
    source/runs/lovasz_seed2/...
    source/runs/augmentation_seed2/...
    comparison.txt
    comparison.json

Each run contains reports/metrics.json, reports/summary.md,
checkpoints/best.pt (selected weights), checkpoints/last.pt, and SUCCESS or
FAILED.json. Do not edit source or reference files inside a prepared batch.
Original screening outputs and runs/six_run/LATEST are preserved.

After reconnecting, repeat cd and source above, then use status or compare.
For an older confirmation batch, add --batch /absolute/SCC/path/to/batch to
the status or compare command. runs/confirmation/LATEST identifies only the
most recently submitted confirmation batch.
