# Measured resources

Observed GPU hours mean allocated one-GPU job wallclock, not GPU utilization. Training GPU hours use summed epoch wallclock.
Observed No time averages combine GPU types. Submission CLI phase times include feature I/O/output; stress refiner times are synchronized init+step time.
Observed CUDA peaks are allocator peaks, not nvidia-smi process memory; existing training peaks include encoder/probe/refiner/optimizer allocations.
Observed Missing measurements remain null/NA. No interpolation supplies K6.
Observed Inference measurements describe the submitted e0_v2 seed-0 model. Training totals cover all controlled experiment seeds.

## Trainable parameters

| Experiment | Core | Feature projection | Answer projection | Initializer | Output head | Total | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e0_v2 | 789760 | 262400 | 65792 | 65792 | 9281 | 1193025 | runs/logs/ec523-e0-s0.o8339750:4 |
| e1_reset | 789760 | 262400 | 65792 | 65792 | 9281 | 1193025 | runs/logs/ec523-e1-s0.o8339753:4 |
| e2_nofb | 789760 | 262400 | 65792 | 65792 | 9281 | 1193025 | runs/logs/ec523-e2-s0.o8339754:4 |
| e3_single | 789760 | 262400 | 65792 | 65792 | 9281 | 1193025 | runs/logs/ec523-e3-s0.o8339755:4 |
| e4_untied | 3159040 | 262400 | 65792 | 65792 | 9281 | 3562305 | runs/logs/ec523-e4-s0.o8432874:4 |
| e5_noadd | 789760 | 262400 | 65792 | 65792 | 9281 | 1193025 | runs/logs/ec523-e5-s0.o8451634:4 |

Observed frozen parameter counts: {'encoder': 303129600, 'probe': 3073}; source results/exp/e0_v2_s0/metrics.json:63 frozen_parameters.

## Training jobs

Observed training duration is sum of epoch seconds, excluding selection/evaluation. Each job reserves one GPU. E0 rerun is listed separately; totals explicitly distinguish the three-seed budget from the budget including rerun.

| Run | GPU | Training seconds | Peak GiB | Whole job seconds | Allocated GPU hours | Training GPU hours | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e0_v2_s0 | NVIDIA L40S | 2152.121489 | 6.170773 | 2616.000000 | 0.726667 | 0.597812 | runs/logs/ec523-e0-s0.o8339750:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19348535 job=8339750 fields wallclock/exit_status/failed |
| e0_v2_s1 | NVIDIA L40S | 2109.297857 | 6.170773 | 2602.000000 | 0.722778 | 0.585916 | runs/logs/ec523-e0-s1.o8339751:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19348518 job=8339751 fields wallclock/exit_status/failed |
| e0_v2_s2 | NVIDIA A40 | 3754.615985 | 6.390328 | 4401.000000 | 1.222500 | 1.042949 | runs/logs/ec523-e0-s2.o8339752:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19350058 job=8339752 fields wallclock/exit_status/failed |
| e0_v2_s0_rerun | NVIDIA L40S | 2081.722896 | 6.170773 | 2487.000000 | 0.690833 | 0.578256 | runs/logs/ec523-e0-s0-rerun.o8451637:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19874023 job=8451637 fields wallclock/exit_status/failed |
| e1_reset_s0 | NVIDIA L40S | 2139.979898 | 6.170773 | 2629.000000 | 0.730278 | 0.594439 | runs/logs/ec523-e1-s0.o8339753:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19348548 job=8339753 fields wallclock/exit_status/failed |
| e1_reset_s1 | NVIDIA A40 | 3717.744783 | 6.390328 | 4383.000000 | 1.217500 | 1.032707 | runs/logs/ec523-e1-s1.o8346482:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19386026 job=8346482 fields wallclock/exit_status/failed |
| e1_reset_s2 | NVIDIA L40S | 2111.863410 | 6.170773 | 2644.000000 | 0.734444 | 0.586629 | runs/logs/ec523-e1-s2.o8346485:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19383175 job=8346485 fields wallclock/exit_status/failed |
| e2_nofb_s0 | NVIDIA L40S | 2070.233427 | 6.061398 | 2544.000000 | 0.706667 | 0.575065 | runs/logs/ec523-e2-s0.o8339754:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19349210 job=8339754 fields wallclock/exit_status/failed |
| e2_nofb_s1 | NVIDIA A40 | 3702.883633 | 6.280953 | 4373.000000 | 1.214722 | 1.028579 | runs/logs/ec523-e2-s1.o8346483:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19386001 job=8346483 fields wallclock/exit_status/failed |
| e2_nofb_s2 | NVIDIA A100 80GB PCIe | 2030.892744 | 6.282603 | 2605.000000 | 0.723611 | 0.564137 | runs/logs/ec523-e2-s2.o8346486:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19385824 job=8346486 fields wallclock/exit_status/failed |
| e3_single_s0 | NVIDIA A100 80GB PCIe | 1341.027258 | 3.014413 | 1863.000000 | 0.517500 | 0.372508 | runs/logs/ec523-e3-s0.o8339755:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19349810 job=8339755 fields wallclock/exit_status/failed |
| e3_single_s1 | NVIDIA A40 | 2404.579315 | 3.012764 | 2951.000000 | 0.819722 | 0.667939 | runs/logs/ec523-e3-s1.o8346484:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19383889 job=8346484 fields wallclock/exit_status/failed |
| e3_single_s2 | NVIDIA L40S | 1408.068239 | 2.793208 | 1787.000000 | 0.496389 | 0.391130 | runs/logs/ec523-e3-s2.o8346487:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19386850 job=8346487 fields wallclock/exit_status/failed |
| e4_untied_s0 | NVIDIA A100 80GB PCIe | 2104.946127 | 6.418457 | 2899.000000 | 0.805278 | 0.584707 | runs/logs/ec523-e4-s0.o8432874:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19829864 job=8432874 fields wallclock/exit_status/failed |
| e4_untied_s1 | NVIDIA L40S | 2106.142730 | 6.197252 | 2560.000000 | 0.711111 | 0.585040 | runs/logs/ec523-e4-s1.o8432875:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19829651 job=8432875 fields wallclock/exit_status/failed |
| e4_untied_s2 | NVIDIA A40 | 3755.752133 | 6.416807 | 4264.000000 | 1.184444 | 1.043264 | runs/logs/ec523-e4-s2.o8432876:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19831523 job=8432876 fields wallclock/exit_status/failed |
| e5_noadd_s0 | NVIDIA A40 | 3745.613812 | 6.390331 | 4379.000000 | 1.216389 | 1.040448 | runs/logs/ec523-e5-s0.o8451634:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19873769 job=8451634 fields wallclock/exit_status/failed |
| e5_noadd_s1 | NVIDIA L40S | 2090.541564 | 6.170776 | 2553.000000 | 0.709167 | 0.580706 | runs/logs/ec523-e5-s1.o8451635:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19872648 job=8451635 fields wallclock/exit_status/failed |
| e5_noadd_s2 | NVIDIA A100 80GB PCIe | 2243.002754 | 6.391981 | 2832.000000 | 0.786667 | 0.623056 | runs/logs/ec523-e5-s2.o8451636:1; epochs 636,1269,1902,2535,3168,3801,4434,5067,5700,6333; SGE accounting:19873100 job=8451636 fields wallclock/exit_status/failed |

| Experiment | Three-seed training GPU hours | Three-seed allocated GPU hours | Allocated GPU hours including rerun |
| --- | --- | --- | --- |
| e0_v2 | 2.226676 | 2.671944 | 3.362778 |
| e1_reset | 2.213774 | 2.682222 | 2.682222 |
| e2_nofb | 2.167781 | 2.645000 | 2.645000 |
| e3_single | 1.431576 | 1.833611 | 1.833611 |
| e4_untied | 2.213011 | 2.700833 | 2.700833 |
| e5_noadd | 2.244211 | 2.712222 | 2.712222 |

## Inference

Observed submission-check phases and stress synchronized refinement are distinct timing scopes; no mixed-scope/GPU mean is reported. K6 was not measured in these preferred sources and remains NA.

| GPU | Phase | K | Seconds/image | Peak GiB | Measurement scope | Source |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | encoding | 1 | 0.148730 | 1.304207 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5143; GPU job.log:1; image count none_k1_run.json:images |
| NVIDIA A40 | refinement | 1 | 0.044710 | 0.539695 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5143; GPU job.log:1; image count none_k1_run.json:images |
| NVIDIA A40 | encoding | 2 | 0.143566 | 1.304207 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5156; GPU job.log:1; image count none_k2_run.json:images |
| NVIDIA A40 | refinement | 2 | 0.054117 | 0.543602 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5156; GPU job.log:1; image count none_k2_run.json:images |
| NVIDIA A40 | encoding | 4 | 0.147885 | 1.304207 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5169; GPU job.log:1; image count none_k4_run.json:images |
| NVIDIA A40 | refinement | 4 | 0.079126 | 0.543602 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5169; GPU job.log:1; image count none_k4_run.json:images |
| NVIDIA A40 | encoding | 8 | 0.144541 | 1.304207 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5182; GPU job.log:1; image count none_k8_run.json:images |
| NVIDIA A40 | refinement | 8 | 0.127345 | 0.543602 | two-phase CLI phase wall time including I/O, not pure model latency | runs/submission_check/8386872/job.log:5182; GPU job.log:1; image count none_k8_run.json:images |
| NVIDIA A40 | encoding | NA | 0.126227 | 1.304207 | encoding wall time includes loading and float32 cache I/O | results/stress/v2/metrics.json:9691 datasets.ordinary.encoding; runs/logs/ec523-stress-eval.o8411770:1 |
| NVIDIA A40 | refiner | 1 | 0.012156 | 0.524070 | cumulative synchronized init+steps, excludes encoder/metrics/output writing | results/stress/v2/metrics.json:3285 / 3291 datasets.ordinary.models.0.refiner_seconds / peak_cuda_bytes; runs/logs/ec523-stress-eval.o8411770:1 |
| NVIDIA A40 | refiner | 2 | 0.023510 | 0.527977 | cumulative synchronized init+steps, excludes encoder/metrics/output writing | results/stress/v2/metrics.json:3285 / 3291 datasets.ordinary.models.0.refiner_seconds / peak_cuda_bytes; runs/logs/ec523-stress-eval.o8411770:1 |
| NVIDIA A40 | refiner | 4 | 0.046089 | 0.531883 | cumulative synchronized init+steps, excludes encoder/metrics/output writing | results/stress/v2/metrics.json:3285 / 3291 datasets.ordinary.models.0.refiner_seconds / peak_cuda_bytes; runs/logs/ec523-stress-eval.o8411770:1 |
| NVIDIA A40 | refiner | 6 | NA | NA | cumulative synchronized init+steps, excludes encoder/metrics/output writing | results/stress/v2/metrics.json:3285 / 3291 datasets.ordinary.models.0.refiner_seconds / peak_cuda_bytes; runs/logs/ec523-stress-eval.o8411770:1 |
| NVIDIA A40 | refiner | 8 | 0.091086 | 0.531883 | cumulative synchronized init+steps, excludes encoder/metrics/output writing | results/stress/v2/metrics.json:3285 / 3291 datasets.ordinary.models.0.refiner_seconds / peak_cuda_bytes; runs/logs/ec523-stress-eval.o8411770:1 |
