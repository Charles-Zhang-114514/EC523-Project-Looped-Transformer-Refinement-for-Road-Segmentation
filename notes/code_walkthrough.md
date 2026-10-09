# Project A code walkthrough

Reviewed on 2026-10-05. References are relative to `project_a/`; `cli/README.md` and root `README.md` are distinguished explicitly. Line numbers refer to the unchanged distributed sources. This is a study note, not a final course report.

## Scope and file map

The package is a review draft containing a runner, random-weight example, synthetic generator, evaluator, stress generator, tests, and handout. It does not provide a trained staff baseline or released data splits (`README.md:9`, `README.md:25`, `cli/README.md:3`, `cli/README.md:6`).

| File | Responsibility and source |
| --- | --- |
| `cli/baseline.py` | Frozen baseline loading and brightness-only demo (`cli/baseline.py:13`, `cli/baseline.py:28`, `cli/baseline.py:34`). |
| `cli/submission/model.py` | Student factory, initializer, and recurrent update (`cli/submission/model.py:19`, `cli/submission/model.py:31`, `cli/submission/model.py:35`, `cli/submission/model.py:48`). |
| `cli/submission/__init__.py` | Package marker, with no model logic (`cli/submission/__init__.py:1`). |
| `cli/predict.py` | Exact-K inference, audit hooks, image I/O, traces and metadata (`cli/predict.py:40`, `cli/predict.py:81`). |
| `cli/common.py` | SHA-256, JSON, ID/manifest/output/probability guards (`cli/common.py:11`, `cli/common.py:19`, `cli/common.py:29`, `cli/common.py:38`, `cli/common.py:46`, `cli/common.py:66`). |
| `cli/metrics.py` | Integer sufficient statistics, pooled metrics, final weighted score (`cli/metrics.py:17`, `cli/metrics.py:32`, `cli/metrics.py:59`, `cli/metrics.py:73`). |
| `cli/evaluate.py` | One condition and K; validate predictions against a label manifest (`cli/evaluate.py:15`). |
| `cli/score.py` | Combine the four compatible evaluation reports (`cli/score.py:9`). |
| `cli/make_stress.py` | RGB corruptions independent of labels (`cli/make_stress.py:17`, `cli/make_stress.py:22`, `cli/make_stress.py:51`). |
| `cli/make_demo.py` | Random safetensors weights and two synthetic road images (`cli/make_demo.py:13`). |
| `cli/smoke_test.py` | Separate-process four-run workflow and demo score assertion (`cli/smoke_test.py:14`). |
| `cli/tests/test_interface.py` | 11 interface/integration tests (`cli/tests/test_interface.py:30`, `cli/tests/test_interface.py:122`). |
| `cli/tests/test_metrics.py` | 17 metric/validation tests (`cli/tests/test_metrics.py:16`, `cli/tests/test_metrics.py:98`). |
| `cli/requirements.txt` | Six exact direct dependency versions (`cli/requirements.txt:1`). |
| `handout/project_a.tex` | Twelve-section specification, equations, experiments, milestones, bibliography (`handout/project_a.tex:43`, `handout/project_a.tex:122`, `handout/project_a.tex:210`, `handout/project_a.tex:398`, `handout/project_a.tex:471`). |
| `handout/project_a.pdf`, `handout/assets/course_imagery.png` | Rendered handout and attributed illustration, not training data; image crops and credit are specified in the TeX (`handout/project_a.tex:62`, `handout/project_a.tex:66`, `handout/project_a.tex:69`). |
| `README.md`, `cli/README.md`, `VALIDATION.md` | Release status, usage/contract, and the distributor's historical validation record (`README.md:9`, `cli/README.md:9`, `VALIDATION.md:3`, `VALIDATION.md:33`). |

All text/code files were read; the PDF's extracted text was checked, and the raster illustration was visually inspected. Historical validation claims in `VALIDATION.md` are separate from our current test logs (`VALIDATION.md:27`, `VALIDATION.md:35`).

## Complete single-image inference flow

Use `B=1` for the CLI, input size `H x W`, baseline feature channels `C`, coarse grid `h x w`, `N=h*w`, and example width `d=32`. Official feature size is not yet specified; the concrete coarse-grid formulas below apply to the demo (`cli/predict.py:121`, `cli/baseline.py:19`, `cli/README.md:39`).

| Stage | Operation and tensor shape | Source |
| --- | --- | --- |
| Image loading | PNG/JPEG -> RGB NumPy float32 `[H,W,3]`, divide by 255 -> PyTorch `[1,3,H,W]` on chosen device. No spatial resize. | `cli/predict.py:100`, `cli/predict.py:119`, `cli/predict.py:121` |
| Frozen baseline | `base(image)` -> `(features, y0)`. Contract: RGB `[B,3,H,W]`, features with declared C, logits `[B,1,H,W]`. Loaded baseline is eval and requires no gradients. | `cli/baseline.py:4`, `cli/baseline.py:43`, `cli/predict.py:53` |
| Demo features | Adaptive average pooling -> `[B,3,h,w]`, `h=max(1,H//16)`, `w=max(1,W//16)`; coordinates `[B,2,h,w]`; concatenate -> `[B,5,h,w]`. | `cli/baseline.py:18`, `cli/baseline.py:24`, `cli/baseline.py:25` |
| Demo initial answer | `8*(mean_RGB(image)-0.5)` -> `y0: [B,1,H,W]`. This is a brightness heuristic, not learned segmentation. | `cli/baseline.py:14`, `cli/baseline.py:25` |
| Initial trace | Clone y0, sigmoid, detach and CPU NumPy -> `[B,1,H,W]`; trace starts with p0. Check initial logit shape and finiteness. | `cli/predict.py:54`, `cli/predict.py:59` |
| Initialize for K>0 | Cloned features and y0 -> example zero `z0: [B,N,32]`, with features' device/dtype. No core call; K=0 skips initialization. | `cli/predict.py:61`, `cli/submission/model.py:31` |
| Feedback at step t | Normal run supplies cloned current `yt: [B,1,H,W]`; cloned fixed features `[B,C,h,w]` and carried state `[B,N,32]`. | `cli/predict.py:64`, `cli/predict.py:67` |
| Mask embedding | sigmoid(yt) `[B,1,H,W]` -> bilinear downsample `[B,1,h,w]`; concatenate with features -> `[B,C+1,h,w]`. | `cli/submission/model.py:36`, `cli/submission/model.py:39` |
| Context embedding | Shared 1x1 convolution -> `[B,32,h,w]`; flatten -> `[B,32,N]`; transpose -> `[B,N,32]`. | `cli/submission/model.py:22`, `cli/submission/model.py:39`, `cli/submission/model.py:40` |
| Recurrent core | `core(zt + context)` -> `z(t+1): [B,N,32]`. One shared TransformerEncoderLayer invocation. | `cli/submission/model.py:24`, `cli/submission/model.py:41` |
| Decode correction | Linear 32->1 -> `[B,N,1]`; transpose -> `[B,1,N]`; reshape -> `[B,1,h,w]`; bilinear upsample -> `[B,1,H,W]`. | `cli/submission/model.py:28`, `cli/submission/model.py:42`, `cli/submission/model.py:43` |
| Update answer | `y(t+1)=yt+0.1*delta`, `[B,1,H,W]`; return with new state. Runner checks shape/finiteness and appends sigmoid output to CPU trace. | `cli/submission/model.py:45`, `cli/predict.py:73`, `cli/predict.py:75` |
| Output | After exactly K steps, select trace[-1][0,0] -> float32 `[H,W]` probability `.npy`; threshold >=0.5 -> uint8 `[H,W]` PNG with values 0/255. Optional per-step arrays k0..kK; `run.json` records hashes/calls/counts/timing. | `cli/predict.py:129`, `cli/predict.py:135`, `cli/predict.py:136`, `cli/predict.py:144` |

For each 96x96 smoke image, the concrete shapes are image `[1,3,96,96]`, features `[1,5,6,6]`, state `[1,36,32]`, and logits `[1,1,96,96]` (derived from `cli/make_demo.py:23`, `cli/baseline.py:19`, `cli/submission/model.py:33`). K=0 returns p0 unchanged, but `main` still constructs and loads the student model and requires a checkpoint (`cli/predict.py:83`, `cli/predict.py:109`, `cli/predict.py:61`).

## Student interface and example, block by block

`build_refiner(feature_channels)` constructs the entire refiner before strict safetensors loading. The example returns `Refiner(feature_channels=...)`; there is no K argument, no checkpoint-specific lazy construction, and no training entry point (`cli/submission/model.py:48`, `cli/predict.py:109`, `cli/predict.py:110`, `cli/submission/model.py:5`).

The constructor registers a shared Conv2d(C+1,32,1), one shared TransformerEncoderLayer with four heads, feed-forward width 64, zero dropout, batch-first tokens and pre-normalization, plus a Linear(32,1) head. These same adapters and core are used at every update. The class has no separate per-step model list (`cli/submission/model.py:20`, `cli/submission/model.py:22`, `cli/submission/model.py:24`, `cli/submission/model.py:28`). Each head's width is 32/4=8, derived from those constructor arguments.

`initialize(features, initial_logits)` is allowed to derive initial memory from both inputs, but this example uses only feature shape/device/dtype and returns zeros. It does not encode the initial answer or invoke attention. That is a minimal initializer, not a learned initialization mechanism (`cli/submission/model.py:31`, `handout/project_a.tex:380`).

`step` converts answer logits to a coarse probability channel, embeds it jointly with fixed image evidence, adds the embedding to carried memory, invokes the transformer once, and predicts a residual correction. The next state carries previous computation, while the full-resolution logit map carries the current answer. Neither state nor logits are model weights; labels never enter the update signature (`cli/submission/model.py:35`, `cli/submission/model.py:41`, `cli/submission/model.py:45`, `handout/project_a.tex:126`, `handout/project_a.tex:127`, `handout/project_a.tex:381`).

The handout's separate image/answer/memory token banks are a conceptual design, not three concatenated token sequences in the example. The example fuses image and answer channels with one convolution, then adds that context to memory. It has no explicit positional embedding inside Refiner; demo coordinates come from the baseline (`handout/project_a.tex:135`, `handout/project_a.tex:164`, `cli/submission/model.py:39`, `cli/submission/model.py:41`, `cli/baseline.py:21`).

## How recurrence is checked, and what remains unproved

1. **Core call count:** `predict_tensor` registers a forward hook on `model.core`, rejects any core call during the initial initializer, and requires cumulative calls to equal t+1 after each step. The hook is removed in `finally`. `test_no_op_core_is_caught` demonstrates rejection of a step that never calls core (`cli/predict.py:46`, `cli/predict.py:62`, `cli/predict.py:69`, `cli/predict.py:77`, `cli/tests/test_interface.py:60`).
2. **Parameter identity/storage:** `parameter_signature` records each named parameter's name, Python object ID, data pointer and shape before baseline/initialization; it compares the whole signature after every step. Adding/replacing parameters or changing their storage is caught (`cli/predict.py:26`, `cli/predict.py:50`, `cli/predict.py:71`). It does not compare parameter values, buffers, or the Python object ID of the core module itself; in-place weight mutation can retain the signature. This is a limitation inferred directly from the tuple fields, not a reproduced exploit (`cli/predict.py:27`).
3. **Outer loop and exact K:** the staff helper owns `range(steps)` and passes no total K to initialize/step. The CLI accepts every integer 0 through 8. `test_extrapolation_counts` tests 6 and 8; `test_zero_is_untouched_baseline` tests K=0 (`cli/predict.py:61`, `cli/predict.py:64`, `cli/predict.py:87`, `cli/tests/test_interface.py:37`, `cli/tests/test_interface.py:47`).
4. **Prefix consistency:** `test_exact_k_and_prefix` runs the same demo model/image separately at K=2 and K=4, then compares the common trace arrays exactly. The production runner does not itself perform a second run or prefix comparison. The handout requires staff to check it in broader review (`cli/tests/test_interface.py:30`, `handout/project_a.tex:411`).
5. **reset-state:** before each step, reinitialize memory from cloned original features/y0, while retaining current answer feedback. For this initializer the state becomes zeros each time (`cli/predict.py:65`, `cli/submission/model.py:33`).
6. **no-mask-feedback:** supply original y0 at every update while retaining the carried memory. In the residual example this changes both the embedding and the residual base: output becomes y0+0.1*delta, rather than yt+0.1*delta. It is therefore a confounded feedback intervention (`cli/predict.py:67`, `cli/submission/model.py:37`, `cli/submission/model.py:45`, `cli/README.md:98`).
7. **Effect and eligibility:** `test_updates_change_prediction` and `test_feedback_interventions_change_demo` show effects for the bundled random example; they do not establish these dependencies for arbitrary future student code. The score combiner rejects ablations. A dummy transformer call whose output is discarded can satisfy a call count yet violate the handout's dataflow rule (`cli/tests/test_interface.py:43`, `cli/tests/test_interface.py:54`, `cli/score.py:16`, `handout/project_a.tex:414`).

The runner clones features before student calls, executes in inference mode, and loads the baseline frozen/eval. It checks output shape and finiteness but not state shape, feature finiteness, semantic transformer architecture, or causality of the output. Audit hooks are explicitly not a security boundary (`cli/predict.py:52`, `cli/predict.py:61`, `cli/predict.py:68`, `cli/predict.py:73`, `cli/baseline.py:43`, `cli/predict.py:4`).

## Metrics and aggregation

`image_counts` requires integer `[H,W]` labels in 0..7 and float32 finite `[H,W]` probabilities in [0,1]. Define V=(labels!=0), G=(labels==3), and P=(probability>=0.5)&V. Other labels are non-road; ignore pixels are excluded before skeletonization (`cli/metrics.py:32`, `cli/metrics.py:39`, `cli/common.py:66`).

- **Road IoU:** store |P intersect G| and |P union G|; after pooling, divide intersection by union. True-negative background is absent from this ratio (`cli/metrics.py:51`, `cli/metrics.py:62`).
- **clDice:** Zhang skeletonize P and G independently on each image. Store |S(P)|, |S(G)|, |S(P) intersect G|, |S(G) intersect P|. Pooled skeleton precision Tp is the third count divided by the first; pooled recall Ts is the fourth divided by the second; clDice=2*Tp*Ts/(Tp+Ts), or 0 if Tp+Ts=0 (`cli/metrics.py:42`, `cli/metrics.py:51`, `cli/metrics.py:63`). It is hard evaluation, not a differentiable training loss or direct graph routability score (`handout/project_a.tex:203`, `handout/project_a.tex:290`).
- **Boundary IoU:** E3 uses three erosions by a 3x3 all-ones square with outside-image pixels zero. Safe region C=E3(V). The inner band B(M)=M & ~E3(M) & C. Pool |B(P) intersect B(G)| and |B(P) union B(G)|, then divide. This excludes the three-pixel image frame and ignored-pixel neighborhoods; it is the course's fixed-width variant (`cli/metrics.py:13`, `cli/metrics.py:45`, `cli/metrics.py:46`, `cli/metrics.py:66`, `handout/project_a.tex:258`).
- **Quality:** A=0.50*IoU+0.30*clDice+0.20*BoundaryIoU, in [0,1] (`cli/metrics.py:69`).
- **Final score:** 100*(0.20*Aordinary1+0.20*Aordinary2+0.40*Aordinary4+0.20*Astress4), in [0,100]. `combine` checks condition/K, protocol, no ablation, compatible checkpoint/baseline/source/source-set hashes, and demo eligibility (`cli/metrics.py:73`, `cli/score.py:12`, `cli/score.py:20`).

`Counts.__add__` adds all integer fields. `evaluate` sums per-image counts into overall/domain/corruption accumulators, then calls summarize on each. It does not average image scores, average domain scores, or skeletonize a joined mosaic. Urban/rural contributions to overall depend on their counts, not equal domain reweighting (`cli/metrics.py:28`, `cli/evaluate.py:31`, `cli/evaluate.py:46`, `cli/evaluate.py:59`, `cli/metrics.py:42`).

Every zero-denominator ratio is defined as 1, without epsilon. Thus both road masks empty give IoU=1 and clDice=1. Only one road mask empty gives clDice=0 and IoU=0. BoundaryIoU independently gives 1 whenever its band union is empty; for example, very small images may have no safe pixels even when one road set is nonempty. Do not generalize the tests' one-empty quality=0 examples to every geometry. All-ignore images add zero counts; a whole evaluation subset with valid=0 is rejected (`cli/metrics.py:55`, `cli/metrics.py:60`, `cli/metrics.py:65`, `cli/metrics.py:45`, `cli/tests/test_metrics.py:26`, `cli/tests/test_metrics.py:98`).

The evaluator also requires a single manifest condition, exact ID sets across manifest/run/probability files, correct recorded core_calls, matching input hashes, and matching image/label dimensions. It reads `.npy` probabilities with pickle disabled, not visualization masks (`cli/evaluate.py:17`, `cli/evaluate.py:24`, `cli/evaluate.py:36`, `cli/evaluate.py:40`, `cli/evaluate.py:43`). The source-set fingerprint uses sorted IDs, domains and label hashes, allowing ordinary/stress versions of the same source labels to match (`cli/evaluate.py:34`, `cli/evaluate.py:52`, `cli/evaluate.py:58`).

## Stress generation and seeds

`generate` accepts only ordinary manifests, creates a fresh output folder, and makes one corrupted RGB image per source row. Within each urban/rural domain, it sorts IDs by SHA-256 of `seed:assignment:id` and assigns shadow/occlusion/blur cyclically. Counts differ by at most one within each domain. Assignment is independent of manifest row order, but adding/removing IDs can shift cyclic category assignments (`cli/make_stress.py:51`, `cli/make_stress.py:56`, `cli/make_stress.py:59`).

`rng_for(seed,id)` hashes `seed:id`, converts the first eight digest bytes to a big-endian integer, and seeds NumPy default_rng. Each tile therefore has an independent deterministic stream for its parameters; identical seed/ID/input and pinned software reproduce the corruption. Changing the global seed can change both category assignment and parameters (`cli/make_stress.py:17`, `cli/make_stress.py:60`, `cli/make_stress.py:65`).

| Family | Exact transformation | Source |
| --- | --- | --- |
| Shadow | Uniform requested area fraction a in [0.04,0.12), aspect r in [0.5,2); rh=round(sqrt(aHW/r)), rw=round(sqrt(aHWr)), each clamped to [1,image extent]. Uniform fitting integer top-left; multiply rectangle RGB by uniform brightness alpha in [0.3,0.6). | `cli/make_stress.py:31`, `cli/make_stress.py:33`, `cli/make_stress.py:35`, `cli/make_stress.py:38` |
| Occlusion | Same rectangle logic, a in [0.005,0.02); fill rectangle with whole source image's per-channel RGB median. | `cli/make_stress.py:31`, `cli/make_stress.py:42` |
| Blur | Sigma uniform in [1,2) pixels; gaussian_filter with sigma=(sigma,sigma,0), reflected padding, truncate=4.0; channels are not mixed. | `cli/make_stress.py:26` |

The table uses half-open notation for NumPy's uniform draws; the handout describes nominal ranges with closed intervals. All operations use normalized float32 RGB, followed by clip [0,1], multiply 255, np.rint and uint8 conversion. Rectangle rounding/clamping means actual area can differ from requested area (`cli/make_stress.py:23`, `cli/make_stress.py:33`, `cli/make_stress.py:37`, `cli/make_stress.py:48`, `handout/project_a.tex:345`).

The generator never opens label images; `load_manifest` resolves their paths and the stress manifest retains those absolute paths. It preserves source IDs/domains, changes condition/image, and records corruption parameters. Moving data requires regenerating manifests. The corrupted image goes through the baseline again; clean features are not cached for stress inference (`cli/common.py:61`, `cli/make_stress.py:64`, `cli/make_stress.py:69`, `cli/README.md:153`, `handout/project_a.tex:352`).

## Demo workflow, handout guidance, and missing work

`make_demo.generate` seeds Torch with 523, constructs the five-channel random model, saves safetensors marked demo/untrained, and draws two 96x96 cross-like road masks on dark RGB backgrounds. Labels are 3 for road and 1 otherwise. No optimizer or training is involved (`cli/make_demo.py:15`, `cli/make_demo.py:17`, `cli/make_demo.py:21`, `cli/make_demo.py:28`, `cli/make_demo.py:34`).

`smoke_test.main` launches generator, stress generator with seed 523, ordinary K=1/2/4 and stress K=4 inference with traces, four evaluators, then score.py with allow-demo. It uses the active Python executable and limits OMP/MKL threads to one; it asserts demo status and score range (`cli/smoke_test.py:20`, `cli/smoke_test.py:24`, `cli/smoke_test.py:27`, `cli/smoke_test.py:34`, `cli/smoke_test.py:38`). The two-image smoke dataset has one image per domain, so cyclic assignment picks shadow for each domain and the smoke workflow itself does not exercise blur/occlusion. The separate family test does (`cli/make_demo.py:21`, `cli/make_stress.py:61`, `cli/tests/test_interface.py:108`).

Staff must release the real frozen baseline adapter/checkpoint, feature contract, course data and manifests, and finalized resources. `load_official_baseline` intentionally raises RuntimeError; specifying a checkpoint cannot make it work without the staff release (`cli/baseline.py:28`, `README.md:9`, `handout/project_a.tex:76`). No training scripts, dataset loader for training, optimizer, learned checkpoint, real data, GPU pilot, deployment service or leaderboard site are bundled (`cli/README.md:6`, `VALIDATION.md:48`).

Our team must implement the refiner in submission/, training data handling with road=3/ignore=0, loss/optimizer/unroll schedule, configuration/checkpoint export and experiment tracking. The handout suggests tiny-subset overfitting, crop training, stable logits BCE with valid-pixel masking and intermediate supervision; training step weights differ from leaderboard weights (`cli/README.md:29`, `handout/project_a.tex:190`, `handout/project_a.tex:198`, `handout/project_a.tex:203`). No implementation changes were made in this review.

The research plan requires K curves including baseline/6/8, state and feedback interventions, retrained controls for stronger feedback claims, a nonrecurrent control, a hypothesis-driven ablation, failure analysis and uncertainty. Targets of 5M core/10M total parameters and 32 GiB training memory are proposals, not enforced runner limits or confirmed feasibility (`handout/project_a.tex:208`, `handout/project_a.tex:400`). Milestones and rubric remain draft; source specifies November 2 and December 14–18 with final time to be confirmed. It also asks for contribution and AI-use disclosures; final reports are student-written (`handout/project_a.tex:423`, `handout/project_a.tex:430`, `handout/project_a.tex:431`, `handout/project_a.tex:436`, `handout/project_a.tex:452`).

## Confusing points and possible gaps

- **Weight sharing checks have limits:** object/storage signatures do not hash weight values or track buffers, nor prove that core outputs influence logits. Baseline freezing is configured, not audited against a later mutation. These are review gaps, not evidence of an actual broken bundled model (`cli/predict.py:26`, `cli/predict.py:71`, `cli/baseline.py:43`, `handout/project_a.tex:414`).
- **Condition metadata gap:** evaluate derives condition from the manifest without verifying that run.json includes a matching condition (the runner does not record one). Source-set identity includes labels and domains but not ordinary-image hashes or corruption recipe/seed; each individual input is still hash-checked. A combined score does not by itself certify the approved stress recipe or identical inference seeds. Staff-controlled generation/rerunning supplies those guarantees (`cli/predict.py:144`, `cli/evaluate.py:38`, `cli/evaluate.py:52`, `cli/evaluate.py:53`, `cli/score.py:20`, `cli/README.md:173`).
- **Initializer guards:** parameter changes made during initialize are noticed only after the first step; at K=0 there is no initialize. Under reset-state, the reinitializer has no immediately separate core-call check, though the cumulative count after step detects extra calls. State shape/value semantics are not explicitly validated (`cli/predict.py:61`, `cli/predict.py:65`, `cli/predict.py:69`, `cli/predict.py:71`).
- **K=0 still loads student weights:** baseline-only evaluation needs a valid student checkpoint through main, despite skipping student initialization and steps. This is an interface inconvenience rather than changed p0 semantics (`cli/predict.py:83`, `cli/predict.py:110`, `cli/predict.py:61`).
- **Memory and timing:** every p0..pK is transferred to CPU even without save-trace. run.json timing excludes model loading and RGB preparation, includes trace transfers and has no warmup; it is not the official end-to-end latency. Example global attention is coarse but still quadratic in N, so demo success does not validate full-image training on L40S (`cli/predict.py:60`, `cli/predict.py:75`, `cli/predict.py:119`, `cli/predict.py:124`, `cli/predict.py:153`, `handout/project_a.tex:206`).
- **Boundary empty bands:** an empty band union scores 1 regardless of whether road masks themselves are empty. The geometry follows the documented metric but can surprise users on tiny crops or masks touching only excluded margins (`cli/metrics.py:45`, `cli/metrics.py:55`, `cli/metrics.py:66`, `cli/tests/test_metrics.py:98`).
- **Fixed direct pins are not a full environment lock:** requirements fixes six packages but not transitive dependencies; CUDA compatibility and exact numerical reproducibility beyond tested CPU behavior remain release responsibilities (`cli/requirements.txt:1`, `cli/README.md:21`, `cli/README.md:198`).
- **Prefix evidence scope:** the test compares repeated helper calls for one synthetic model/input; it is not a universal CLI/GPU prefix audit. Confirm the final trained model separately at identical image/checkpoint/seed (`cli/tests/test_interface.py:30`, `handout/project_a.tex:411`).

## Least certain conclusions and verification needed

1. **Official baseline feature interface and normalization:** demo dimensions are exact, but official C, stride, preprocessing and checkpoint behavior cannot be inferred until staff release the adapter. Revisit the shape table then (`cli/baseline.py:28`, `cli/README.md:39`).
2. **Final GPU reproducibility and training feasibility:** CPU test success does not establish L40S memory use, supported deterministic CUDA operators, or final limits. Benchmark the released environment and trained architecture (`VALIDATION.md:50`, `cli/README.md:198`, `handout/project_a.tex:208`).
3. **Severity of audit/report gaps:** the omissions described above follow source inspection, but their practical impact depends on the staff evaluation service and review process, which are absent. They have not been tested as adversarial exploits (`cli/predict.py:4`, `cli/README.md:193`, `VALIDATION.md:52`).
