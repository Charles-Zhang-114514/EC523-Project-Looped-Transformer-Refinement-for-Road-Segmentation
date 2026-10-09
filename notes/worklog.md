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
