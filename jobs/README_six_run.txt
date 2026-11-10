SIX-RUN V2 EXPERIMENT BATCH
Prepared November 1, 2026

Run the following commands from your SCC checkout. Submitting creates new training runs.

1. SUBMIT ON SCC (run once)

cd "$(git rev-parse --show-toplevel)"
source scripts/activate_scc.sh
python scripts/six_run_batch.py submit

The helper copies the current experiment source, validation split, and small
trained probe into a uniquely named directory under runs/six_run/. It includes
uncommitted/new source files and records their hashes and Git provenance.
The existing shared DINOv3 cache and course dataset are reused offline.
Do not modify the source copy inside a prepared batch.

CPU tests and real input/cache checks run before submission. Each allocated task
then checks its GPU, runs CUDA loss checks and a real DINO/probe forward pass,
and starts training. A failed preflight does not submit any jobs.

Submission uses jobs/refiner_six.qsub as a six-task array. It prints the job ID,
batch path, and log locations, and saves the path in runs/six_run/LATEST.
All tasks request an L40S; qgpus -s supplies the scheduler's exact spelling.
At most two tasks run together. Each task requests one GPU, 8 CPU slots,
64 GB host RAM, and a four-hour wall-time limit including evaluation.
The four-hour limit is an allocation limit, not a measured runtime estimate.
New losses, especially soft-clDice, can take longer than the earlier BCE run.

Run the submit command only once for this batch. Repeating it creates six MORE
runs. A missing L40S inventory entry is reported before submission; it does not
silently mix GPU models. Optional controls, used INSTEAD of the command above:

python scripts/six_run_batch.py submit --max-running 1

For a deliberate entire-batch hardware change, --gpu-type A40 is supported.
Never change hardware for only one task when making this comparison.

2. CHECK STATUS

qstat -u "$USER"
python scripts/six_run_batch.py status

qstat may combine queued array tasks into a range. 'qw' means queued, 'r' means
running, and 'Eqw' means a scheduler error. The separate status command shows
all six experiments. 'STARTED' alone does not prove the process is still running;
check qstat and its log. A killed/timed-out job may lack a FAILED marker.
All six must say SUCCESS to consider the batch finished successfully.

To follow task 3 (the targeted false-addition experiment):

batch_dir=$(cat runs/six_run/LATEST)
tail -n 40 -F "$batch_dir/logs/task_3.log"

Change 3 to 1..6 for another task. The log may not exist until that task starts.
Ctrl+C stops only tail, not the SCC training job. You can disconnect after
submission; the scheduler keeps the jobs running. For an early shell/startup
failure also inspect the scheduler output files in the same logs directory.

3. COMPARE THE COMPLETED RUNS

python scripts/six_run_batch.py compare

This prints the comparison and saves comparison.txt plus comparison.json in
the batch directory. It also lists missing/failed runs. It rejects mismatched
validation memberships, settings, GPU/software runtimes, or modified snapshots.
Use the NEW control as the primary comparator. The previous pos_weight=2 and
pos_weight=4 jobs remain useful context but ran on different GPU models.

Each experiment has:
  <batch>/source/runs/<experiment>/checkpoints/best.pt  selected weights
  <batch>/source/runs/<experiment>/checkpoints/last.pt  last-epoch weights/state
  <batch>/source/runs/<experiment>/reports/metrics.json
  <batch>/source/runs/<experiment>/reports/summary.md
  <batch>/source/runs/<experiment>/SUCCESS             final evaluation succeeded

After reconnecting, repeat cd and source from step 1 before using these commands.
LATEST points to the most recently submitted batch. For an older batch use:
  python scripts/six_run_batch.py status --batch /absolute/SCC/path/to/batch
  python scripts/six_run_batch.py compare --batch /absolute/SCC/path/to/batch

4. EXPERIMENT DEFINITIONS

Task  Name           Change from control
1     control        BCE, pos_weight=4, no augmentation
2     posw3          BCE, pos_weight=3
3     false_add      BCE4 + targeted false-addition penalty, calibrated on Train
4     lovasz         BCE4 + 0.1 * binary Lovasz hinge
5     soft_cldice    BCE4 + 0.1 * soft-clDice, 20 skeletonization iterations
6     augmentation   BCE4 + random D4 (90-degree rotations and reflections)

Shared settings: seed 0; ten epochs from the same refiner initialization;
batch size 4; AdamW lr=3e-4, weight decay=0.01, LR warmup=500 steps;
K_train=4 with equal loss at every loop; dimension 256, one shared block,
8 heads, conv_upsample output head; persistent state and mask feedback enabled.
The pretrained DINOv3 encoder and existing probe stay frozen for every run.
Only refiner weights are trained. No variant loads a teammate's refiner weights.

All experiments train on both Urban and Rural Train images. Augmentation applies
the same transform to image, target, valid mask, and raw labels BEFORE the frozen
encoder. It is deterministic by seed/epoch/image and never applies to validation.

The false-addition term is the mean squared probability increase relative to
the frozen draft, restricted to valid true-background pixels that the draft
also classified as background. True road pixels are exempt from this penalty.
Its first 500 optimizer updates use BCE only; the next eight also use BCE only
while measuring independent-logit gradient norms on Train. From update 509,
a fixed coefficient makes the SUM of those calibration auxiliary gradient norms
10% of the SUM of BCE gradient norms. This is an initial calibration target,
not a guarantee of that ratio throughout training or for parameter gradients.
Calibration fails visibly if it cannot measure a usable gradient. Its state and
coefficient are saved in checkpoints and metrics.json; loss components and the
active coefficient appear in each batch log. Warmup is necessary because the
zero-initialized refiner initially makes no additions to penalize.

Lovasz pools valid pixels across each minibatch. Soft-clDice uses valid-clipped
masks, zero exterior padding, batch-pooled skeleton precision/recall, and
checkpointed predicted skeletons to reduce retained memory. All-background
batches use BCE alone. These are training surrogates; official evaluation code,
threshold 0.5, and A = 0.5 IoU + 0.3 clDice + 0.2 boundary IoU stay unchanged.

Checkpoint selection: fixed 100 images/domain from Val-sel, every epoch,
maximize 0.25*A(K1) + 0.25*A(K2) + 0.5*A(K4). Report the selected checkpoint
on all 835 disjoint Val-rep images at K=1,2,4,6,8. The split is the existing
configs/val_split_seed0.json. Selection never uses the Val-rep scores.

The comparison includes overall/Urban/Rural A at K4, selection and reporting
proxies, Urban total/new false-road fragments, gap repair, and newly broken
centerline coverage. Full component metrics and K curves are in each report.
Fewer fragments alone are insufficient: check rural/overall quality, added road
pixels, gap repair, and newly lost coverage to detect another deletion-only model.
These six single-seed runs are a screening study. Confirm promising variants
with additional seeds and the existing stress evaluation before a final claim.
The proxy above is not the full stress-inclusive course score. Val-rep has
already been used for development and is not a new untouched test set.

SCC GPU option reference:
https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/
