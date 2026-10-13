# Work log

Entries are sorted by date. Historical course-file references refer to the externally obtained starter package. Local artifacts and course materials are not included in this public repository.

## 2026-10-05 — Initial setup attempt (recorded retrospectively)

- Checked the local environment, Python modules, and storage availability. Found Python 3.12.4 and sufficient space; private operational details are omitted from this public copy.
- Extracted the original archive into project_a without changing the archive. Created env/venv and env.sh, with project-local pip, temporary, XDG, Torch, Hugging Face and Python bytecode locations.
- Installation failed on sandbox DNS/network access; the explicit PyPI retry also failed. Unittest and smoke commands failed during import because NumPy was absent. Historical complete logs are retained under runs/logs.
- Read all distributed text/code, extracted and checked handout PDF text, and inspected the course illustration. The session was interrupted before setup/walkthrough documents were written.
- Remaining at interruption: user-managed dependency installation, successful tests, final setup documentation and code walkthrough.

## 2026-10-05 — Resumed offline validation and documentation

- The user reported manual dependency installation outside the sandbox. No network operation or dependency installation was attempted in this session.
- Checked env.sh and appended nodejs/20.12.2 plus the project tools/node_modules/.bin PATH entry. Confirmed Python 3.12.4, Node v20.12.2, venv activation, and shell syntax.
- Verified exact required/installed versions: numpy 2.3.5, Pillow 12.3.0, torch 2.6.0, scipy 1.15.2, scikit-image 0.25.2, safetensors 0.5.3. All six imported successfully. Recorded versions and pip freeze in runs/logs.
- Ran the prescribed unittest command in project_a/cli: all 27 tests passed in 1.513 seconds, exit 0. Ran the prescribed smoke output command: the complete ordinary K=1/2/4 and stress K=4 workflow passed, exit 0; combined score about 96.499636, marked DEMO ONLY.
- Preserved pre-installation test failure logs; saved fresh successful unittest/smoke logs to runs/logs. Smoke artifacts are in runs/smoke_outputs.
- Compared all 20 distributed files to zip entries byte-for-byte; all matched. No staff-owned file, test, or student example was modified.
- Created SETUP.md with activation, manual installation outside the sandbox, locations, actual test results, historical errors and rerun instructions. Completed notes/code_walkthrough.md with source-line citations, all stage shapes, interface explanation, recurrence checks, metrics, stress recipe, missing pieces and review uncertainties.
- Remaining issues: official baseline/data/feature contract and finalized resource limits await staff release; training scripts and learned model remain future project work. Torch is a CUDA 12.4 build but no GPU was available on the login-node check; CUDA determinism and L40S feasibility are unverified. Audit/report omissions are source-review observations, not tested exploits. No blocker remains for the requested CPU setup and smoke workflow.

## 2026-10-05 — Structured synthetic road benchmark

- Read notes/code_walkthrough.md, notes/worklog.md, and official metrics.py before implementation. All new files remain under analysis/; project_a and the original notes/worklog.md remain unchanged. This copy appends today's record while respecting the all-files-under-analysis constraint.
- Implemented configurable CPU-only analysis/synth_benchmark.py with 20 seeded 1024x1024 integer-label images, mixed road widths, crossing roads, three isolated parallel pairs, and occasional ignore rectangles. Used only installed NumPy, SciPy, scikit-image, and Pillow plus the Python standard library.
- Extracted Zhang centerlines, endpoints, junction pixels, and segment width statistics from Euclidean distance transforms. Recorded geometry, all error parameters and locations, gap anchors, individual repair outcomes, versions, and pooled official Counts in results/metadata.json.
- Ran five gap lengths, one/three bridges, three uniform dilation radii, two proportional dilation fractions, four closing radii for every gap condition, and matching clean-closing controls. Official Counts are added across images before summarize; every clean image scored exactly 1.0 on all four requested metrics before the scan.
- Verified identical labels for repeated seeds, invariance of official Counts to predictions inside ignore pixels, disconnected pre-repair anchors for every gap, and separate original components for designated bridge pairs.
- The complete scan took about 134 seconds. Review caught proportional dilation using nearest foreground boundary widths; corrected the runnable script to use nearest centerline widths and recomputed only affected metrics, metadata, report rows, and example images. Extra recompute timing is recorded in metadata and results.md.
- Saved metrics.csv, gap_repair.csv, clean_false_bridges.csv, English results.md with a table for each experiment, raw integer labels for all images, and clean/error/closed Pillow PNG examples. Observed high metric scores despite disrupted connectivity; larger closing footprints repaired some gaps but also merged clean parallel components.
- Key assumptions to review: axis-aligned terminal-branch gap placement without alternate routes; 8-neighbor component connectivity; proportional outward dilation defined as a fraction of full centerline-estimated width. Synthetic geometry omits curves, imagery, grade separation, occlusion, and realistic prediction noise, so findings may differ on real data.

## 2026-10-09 — Public repository structure and import

- Created the src/looproad package scaffold, submission, analysis, scripts, configs, jobs, tests, results, and notes directories. Added source package initializers and placeholders for empty directories.
- Extended the existing Python Git ignore rules for private datasets, features, weights, environments, course materials, generated artifacts, logs, and credentials. Preserved the source data-loader package through explicit exceptions.
- Imported the original synthetic benchmark script, three result CSVs, the English report, four selected visualization PNGs, and the code walkthrough. Excluded raw labels, large metadata, logs, PDFs, and all course source code. Merged historical work logs by dated section and deduplicated identical entries.
- Replaced the benchmark's fixed starter import path with EC523_STARTER_DIR and clear missing/invalid-directory errors. Default and allowed raw-output locations are under ignored tmp/. Generated metadata stores repository-relative output paths.
- Added a concise README, an SCC environment template with placeholders, and exact dependency pins matching the external starter requirements. Removed private storage figures from the public work-log copy.
- Verified shell syntax, missing-variable behavior, and a real CPU run with n=2: all clean self-scores equaled 1.0, all 37 pooled metric conditions completed in 14.1 seconds, and outputs stayed under ignored tmp/synth_smoke/.
- Scanned public text for personal account identifiers, private filesystem prefixes, absolute paths, and credential patterns. Confirmed selected result Markdown, CSV, and PNG files remain trackable. Used the configured global Git identity for the local commit; no network request or push was made.

## 2026-10-13 — Shared dataset audit on charles/data-audit

- Created local branch charles/data-audit from main with a clean working tree. Read the shared RoadSeg dataset.py for directory/pairing conventions; used Pillow and preserved raw 0–7 masks rather than its clipping to 0–6.
- Added scripts/data_audit.py with EC523_DATA_DIR input, a default 540-second deadline, sequential CPU processing and reproducible 50-image samples per Train/Val × Urban/Rural group (seed 523). Added the shared dataset environment variable to env.sh.example. No dependency installation, network operation, push, or modification/copy of shared source data occurred.
- Completed full integrity/label checks over 4191 image-mask pairs: Train/Urban 1156, Train/Rural 1366, Val/Urban 677, Val/Rural 992. Observed no missing/orphan masks, decode errors, image format/size errors, mask format/size errors or values outside 0–7. All images were RGB uint8 1024×1024; masks were single-channel integers.
- Observed road/valid fractions of 9.28%, 1.94%, 7.76%, 2.61% in the group order above. No-road counts were 141, 486, 96, 314. Label 7 occupied 1.77% of Train/Urban versus 24.67% of Val/Urban; this is a descriptive split difference, with no causal interpretation.
- Computed 8-neighbor road components/sizes, Zhang centerline degree-based endpoints and clustered junctions, and 2×Euclidean-distance width estimates on 200 sampled masks. Observed median widths 26.68, 12.65, 29.12, 16.00 pixels; p90 widths 111.89, 24.74, 110.89, 84.00 pixels. Width distributions pool centerline pixels; junctions, broad regions, rasterization and truncated image boundaries affect interpretation. Actual gap lengths were not measured.
- Full audit runtime was 394.14 seconds, within the 10-minute budget. A separate synthetic fixture check verified exact label-7 preservation/fractions and detection of missing masks, orphan masks, invalid images and invalid label values in under one second. Verified 200 sample rows, fractions summing to one and four derived RGB overlays. Full runtime log is in the parent project's runs/logs/data_audit_20261013.log (ignored, not committed).
- Saved six CSVs, summary.md with one table per audit item, and four derived road-overlaid examples under results/data_audit. The original first Urban sample had no road, so example selection was adjusted to the first road-containing member of the same sampled subset; regenerated only four overlays and updated the script/report, without rerunning or changing statistics.
- Remaining issues: structural measurements are annotation proxies rather than validated graph topology or physical road width. Wide upper tails and large label-distribution shifts merit further visual inspection before synthetic calibration. No GPU or training work was performed. Commit contains only the requested script, environment-template change, derived results and this work-log entry; no push.
