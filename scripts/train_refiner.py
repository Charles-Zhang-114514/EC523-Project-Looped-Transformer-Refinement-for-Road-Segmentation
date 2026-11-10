#!/usr/bin/env python3
"""Train only a shared float32 refiner on frozen RoadSeg probe drafts."""
import argparse
from collections import deque
from contextlib import contextmanager
import json
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import y0_error_analysis as y0
from scripts import train_linear_probe as probe
from looproad.models.refiner import LoopRefiner
from looproad.models.refiner_losses import RefinerObjective
from looproad.data.augmentation import D4TrainingDataset

DOMAINS = ("Urban", "Rural")
INFER_ABLATIONS = ("none", "reset-state", "no-mask-feedback")
SUPPORTED_INFER_ABLATIONS = (*INFER_ABLATIONS, "y0_answer_keep_residual")
STRUCTURE_KEYS = ("gap_skeleton_pixels", "terminal_missing_skeleton_pixels",
                  "whole_missed_skeleton_pixels", "fp_fragments", "predicted_components")


def stamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def frozen_draft(encoder, head, images):
    """Use the exact existing encoder.forward -> LinearProbe.forward path.

    Normalization/bf16 live in the encoder; projection/interpolation live in
    LinearProbe.forward. Neither is duplicated here or changed for this run.
    """
    encoder.eval()
    head.eval()
    with torch.no_grad():
        features = encoder(images)
        draft = head(features).float()
    return features, draft


def refine_steps(model, features, draft, steps):
    """The caller owns the loop bound; preserve the entire recurrent graph."""
    state, feature_tokens = model.init(features, draft)
    answer, outputs = draft, []
    for _ in range(steps):
        state, answer = model.step(state, answer, feature_tokens)
        outputs.append(answer)
    return outputs


@contextmanager
def draft_answer_tokens(model, draft, enabled):
    """Override only answer-token output; step still adds delta to its answer."""
    if not enabled:
        yield
        return
    tokens = model.answer_tokens(draft)
    handle = model.answer_projection.register_forward_hook(lambda _module, _args, _output: tokens)
    try:
        yield
    finally:
        handle.remove()


def inference_steps(model, features, draft, ks, device, ablation="none"):
    """Adapt predict.py:61-68 without changing LoopRefiner's init/step API.

    Reset means rerunning init(features, draft) before EVERY step, not zeroing
    state. Replacing the step answer also replaces its residual base with y0.
    Yield requested logits only; the caller owns K and converts masks promptly.
    """
    if ablation not in SUPPORTED_INFER_ABLATIONS:
        raise ValueError(f"Unknown inference ablation: {ablation}")
    check_untied_limit(model.untied_steps, ks, "inference K")
    if model.reset_state or model.no_mask_feedback:
        raise ValueError("Inference interventions require an unmodified checkpoint/model")
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        phase = stamp(device)
        state, feature_tokens = model.init(features.float().clone(), draft.float().clone())
        elapsed = stamp(device) - phase
        current = draft.clone()
        phase = stamp(device) if ablation == "y0_answer_keep_residual" else None
        with draft_answer_tokens(model, draft, ablation == "y0_answer_keep_residual"):
            if ablation == "y0_answer_keep_residual":
                elapsed += stamp(device) - phase
            for step in range(1, max(ks) + 1):
                phase = stamp(device)
                if ablation == "reset-state":
                    state, feature_tokens = model.init(features.float().clone(), draft.float().clone())
                feedback = draft.clone() if ablation == "no-mask-feedback" else current.clone()
                state, current = model.step(state, feedback, feature_tokens)
                elapsed += stamp(device) - phase
                if step in ks:
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    yield str(step), current, elapsed, peak


def deep_supervision(outputs, targets, valid, pos_weight=4.):
    losses = [probe.masked_bce_with_logits(answer, targets, valid, pos_weight=pos_weight) for answer in outputs]
    return torch.stack(losses).mean(), losses


def noadd_penalties(outputs, draft, targets, valid):
    """Penalize probability increases on valid GT background with y0 < 0.5."""
    initial_probability = draft.detach().sigmoid().squeeze(1)
    mask = valid.bool() & (targets == 0) & (initial_probability < .5)
    # Match masked_bce_with_logits: sum over the full batch's valid pixels.
    denominator = valid.to(dtype=outputs[0].dtype).sum().clamp_min(1)
    penalties = [((answer.sigmoid().squeeze(1) - initial_probability).relu().square()
                  * mask).sum() / denominator for answer in outputs]
    return torch.stack(penalties).mean(), penalties


def refiner_objective(outputs, targets, valid, draft, noadd_lambda=0., pos_weight=4.):
    """Return total, per-step BCE, mean BCE, mean penalty and per-step penalty."""
    bce, losses = deep_supervision(outputs, targets, valid, pos_weight=pos_weight)
    if noadd_lambda == 0:
        # No new tensor operation, sigmoid, graph branch or RNG use at lambda=0.
        return bce, losses, bce, 0., [0.] * len(outputs)
    penalty, penalties = noadd_penalties(outputs, draft, targets, valid)
    return bce + noadd_lambda * penalty, losses, bce, penalty, penalties


def initialize_eval_worker(starter):
    probe.initialize_metric_worker(starter, (.5,))


def score_image(domain, image_id, predictions, labels, geometry, eval_extra=False):
    started = time.perf_counter()
    valid = labels != 0
    draft = predictions["y0_draft"].astype(bool)
    if eval_extra:
        from analysis import refiner_diagnostics as diagnostics
        truth = labels == 3
        gap, terminal, covered = diagnostics.baseline_skeleton_sets(truth, draft & valid)
    outputs, cache = {}, []
    for key, predicted in predictions.items():
        cached = next((item for mask, item in cache if np.array_equal(mask, predicted)), None)
        if cached is None:
            _, counts, _, _, _ = probe.metric_counts_for_image(domain, predicted, labels)
            structure = None
            if geometry:
                analysis = y0.analyze_masks(predicted, labels)
                structure = {kind + "_skeleton_pixels": sum(
                    item["skeleton_pixels"] for item in analysis["segments"][kind].values())
                    for kind in ("gap", "terminal_missing", "whole_missed")}
                structure.update(fp_fragments=len(analysis["fragment_areas"]),
                                 predicted_components=analysis["predicted_components"])
            cached = {"counts": counts["0.5"], "structure": structure}
            cache.append((predicted, cached))
        mask = predicted.astype(bool)
        outputs[key] = {**cached, "added": int((mask & ~draft & valid).sum()),
                        "deleted": int((~mask & draft & valid).sum())}
        if eval_extra:
            fragments = diagnostics.fragment_shapes(mask, draft, truth, valid)
            outputs[key]["extra"] = {
                "changes": diagnostics.change_counts(draft, mask, truth, valid),
                "tracking": {"gap_total": int(gap.sum()), "gap_covered": int((gap & mask).sum()),
                             "terminal_total": int(terminal.sum()), "terminal_covered": int((terminal & mask).sum()),
                             "covered_total": int(covered.sum()), "covered_lost": int((covered & ~mask).sum())},
                "new_fragments": {"count": len(fragments), "aligned_edges": sum(f["aligned_edges"] for f in fragments)}}
    return domain, image_id, outputs, time.perf_counter() - started


def make_loader(dataset, batch_size, workers, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True,
                      persistent_workers=False, collate_fn=probe.collate_numpy_batch,
                      worker_init_fn=probe.initialize_data_worker if workers else None)


def evaluate(encoder, head, model, datasets, ks, device, loader_workers=1, metric_workers=6, geometry=True, eval_extra=False,
             infer_ablations=None):
    """One streamed trajectory to max(ks); workers receive only uint8 masks."""
    check_untied_limit(model.untied_steps, ks, "evaluation K")
    started = time.perf_counter()
    model.eval()
    metrics = probe.official_metrics()
    if infer_ablations is not None:
        if not infer_ablations or any(mode not in SUPPORTED_INFER_ABLATIONS for mode in infer_ablations):
            raise ValueError("Expected at least one valid inference ablation")
        if len(set(infer_ablations)) != len(infer_ablations):
            raise ValueError("Inference modes must be unique")
    keys = ["y0_draft"] + ([f"{mode}:{k}" for mode in infer_ablations for k in ks]
                           if infer_ablations is not None else [str(k) for k in ks])
    counts = {domain: {key: metrics.Counts() for key in keys} for domain in DOMAINS}
    sums = {domain: {key: {name: 0 for name in (*STRUCTURE_KEYS, "added", "deleted")} for key in keys}
            for domain in DOMAINS}
    times = {domain: {key: [] for key in keys} for domain in DOMAINS}
    peaks = {domain: {key: None for key in keys} for domain in DOMAINS}
    extra_sums = {domain: {key: {} for key in keys} for domain in DOMAINS}
    pending, images = deque(), []
    worker_seconds, encoder_seconds = 0., 0.

    def consume():
        nonlocal worker_seconds
        domain, image_id, values, seconds = pending.popleft().get()
        worker_seconds += seconds
        images.append({"domain": domain, "image_id": image_id})
        for key, item in values.items():
            counts[domain][key] = counts[domain][key] + metrics.Counts(**item["counts"])
            for name in ("added", "deleted"):
                sums[domain][key][name] += item[name]
            if geometry:
                for name in STRUCTURE_KEYS:
                    sums[domain][key][name] += item["structure"][name]
            if eval_extra:
                for section, values in item["extra"].items():
                    target = extra_sums[domain][key].setdefault(section, {})
                    for name, value in values.items():
                        target[name] = target.get(name, 0) + value

    pool = mp.get_context("spawn").Pool(metric_workers, initializer=initialize_eval_worker,
                                        initargs=(os.environ.get("EC523_STARTER_DIR"),))
    try:
        for domain in DOMAINS:
            loader = make_loader(datasets[domain], 1, loader_workers)
            for image, _, _, image_id, raw in loader:
                image = probe.as_device_tensor(image, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                phase = stamp(device)
                features, draft = frozen_draft(encoder, head, image)
                encoder_seconds += stamp(device) - phase
                predictions = {"y0_draft": (draft.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)}
                times[domain]["y0_draft"].append(0.)
                if device.type == "cuda":
                    value = torch.cuda.max_memory_allocated(device)
                    peaks[domain]["y0_draft"] = max(peaks[domain]["y0_draft"] or 0, value)
                    torch.cuda.reset_peak_memory_stats(device)
                if infer_ablations is not None:
                    for mode in infer_ablations:
                        for key, answer, elapsed, peak in inference_steps(model, features, draft, ks, device, mode):
                            name = f"{mode}:{key}"
                            times[domain][name].append(elapsed)
                            if peak is not None:
                                peaks[domain][name] = max(peaks[domain][name] or 0, peak)
                            predictions[name] = (answer.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)
                else:
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        phase = stamp(device)
                        state, feature_tokens = model.init(features.float(), draft.float())
                        elapsed = stamp(device) - phase
                        answer = draft
                        for step in range(1, max(ks) + 1):
                            phase = stamp(device)
                            state, answer = model.step(state, answer, feature_tokens)
                            elapsed += stamp(device) - phase
                            if step in ks:
                                key = str(step)
                                times[domain][key].append(elapsed)
                                if device.type == "cuda":
                                    peaks[domain][key] = max(peaks[domain][key] or 0,
                                                             torch.cuda.max_memory_allocated(device))
                                predictions[key] = (answer.sigmoid().squeeze() >= .5).cpu().numpy().astype(np.uint8)
                labels = np.asarray(raw).squeeze().astype(np.uint8, copy=False)
                pending.append(pool.apply_async(score_image, (domain, image_id[0], predictions, labels, geometry, eval_extra)))
                if eval_extra:
                    print(f"EVALUATED {domain} {image_id[0]} K={max(ks)}", flush=True)
                if len(pending) >= 2 * metric_workers:
                    consume()
            del loader
        while pending:
            consume()
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    report = {domain: {} for domain in (*DOMAINS, "Overall")}
    for domain in report:
        for key in keys:
            domains = DOMAINS if domain == "Overall" else (domain,)
            pooled = metrics.Counts()
            for item in domains:
                pooled = pooled + counts[item][key]
            totals = {name: sum(sums[item][key][name] for item in domains) for name in sums[DOMAINS[0]][key]}
            durations = [value for item in domains for value in times[item][key]]
            memory = [peaks[item][key] for item in domains if peaks[item][key] is not None]
            score = metrics.summarize(pooled)
            score["A"] = score["quality"]
            report[domain][key] = {"scores": score,
                "structure": {name: totals[name] for name in STRUCTURE_KEYS} if geometry else None,
                "added_road_pixels": totals["added"], "deleted_road_pixels": totals["deleted"],
                "added_road_fraction": totals["added"] / pooled.valid,
                "deleted_road_fraction": totals["deleted"] / pooled.valid,
                "refiner_seconds": sum(durations),
                "refiner_seconds_per_image": sum(durations) / len(durations),
                "peak_cuda_bytes": max(memory) if memory else None}
            if eval_extra:
                from analysis.refiner_diagnostics import fraction
                sections = {section: {name: sum(extra_sums[d][key][section][name] for d in domains)
                                      for name in values} for section, values in extra_sums[domains[0]][key].items()}
                changes, tracking, fragments = (sections[s] for s in ('changes', 'tracking', 'new_fragments'))
                changed = sum(changes.values())
                changes.update(fractions={n: fraction(v, pooled.valid) for n, v in changes.items()},
                               correct_fraction=fraction(changes['added_correct']+changes['deleted_correct'], changed),
                               added_correct_fraction=fraction(changes['added_correct'], changes['added_correct']+changes['added_wrong']),
                               deleted_correct_fraction=fraction(changes['deleted_correct'], changes['deleted_correct']+changes['deleted_wrong']))
                tracking.update(gap_repair_fraction=fraction(tracking['gap_covered'], tracking['gap_total']),
                                terminal_coverage_fraction=fraction(tracking['terminal_covered'], tracking['terminal_total']),
                                new_break_fraction=fraction(tracking['covered_lost'], tracking['covered_total']))
                fragments['aligned_edge_fraction'] = fraction(fragments['aligned_edges'], 4*fragments['count'])
                report[domain][key]['extra'] = sections
    result = {"groups": report, "images": images, "encoder_seconds": encoder_seconds,
              "metric_geometry_worker_seconds": worker_seconds, "wall_seconds": time.perf_counter() - started}
    if infer_ablations is not None:
        result['ablations'] = {mode: {'groups': {domain: {
            'y0_draft': variants['y0_draft'], **{str(k): variants[f'{mode}:{k}'] for k in ks}}
            for domain, variants in report.items()}} for mode in infer_ablations}
        reference = 'none' if 'none' in infer_ablations else infer_ablations[0]
        result['groups'] = result['ablations'][reference]['groups']
        result['groups_reference_mode'] = reference
        result['timing_note'] = 'One shared encoder/draft pass. Wall/worker times cover all modes; per-K refiner times are cumulative within each mode.'
    return result


def parse_infer_ablations(text):
    modes = [value.strip() for value in text.split(',')]
    if not modes or any(mode not in SUPPORTED_INFER_ABLATIONS for mode in modes) or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError('Expected unique comma-separated modes: '+','.join(SUPPORTED_INFER_ABLATIONS))
    return modes


def parse_ks(text):
    try:
        values = sorted(set(int(value) for value in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--eval-k must be comma-separated positive integers") from error
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("y0_draft is a reference, not K=0")
    return values


def lr_multiplier(completed, total_steps, warmup=500):
    if completed < warmup:
        return (completed + 1) / warmup
    progress = min(1., (completed - warmup) / max(1, total_steps - warmup))
    return .5 * (1. + math.cos(math.pi * progress))


def capture_rng(loader):
    """Use tensors and Python primitives so weights_only=True can load states."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "loader": loader.generator.get_state(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]}}


def restore_rng(state, loader, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    loader.generator.set_state(state["loader"].cpu())
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def parse_selection_ks(text):
    try:
        pairs = [value.split(':') for value in text.split(',')]
        result = {str(int(k)): float(weight) for k, weight in pairs}
        if len(result) != len(pairs) or any(int(k) < 1 or not math.isfinite(w) or w <= 0 for k, w in result.items()):
            raise ValueError
        if not math.isclose(sum(result.values()), 1., abs_tol=1e-9):
            raise ValueError
        return dict(sorted(result.items(), key=lambda item: int(item[0])))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('Expected unique positive K:weight pairs with finite positive weights summing to one')


def protocol_datasets(split, part, limit=None):
    """Selection and limited report sampling both use seed 0, never training seed."""
    from scripts.make_val_split import validate_split
    datasets = {domain: probe.RoadDataset('Val', domain) for domain in DOMAINS}
    names = {domain: [a.name for a, _ in dataset.pairs] for domain, dataset in datasets.items()}
    validate_split(split, names)
    for domain, dataset in datasets.items():
        allowed = set(split[part][domain])
        dataset.pairs = [pair for pair in dataset.pairs if pair[0].name in allowed]
        if limit is not None and len(dataset) > limit:
            indices = np.sort(np.random.default_rng(0).choice(len(dataset), size=limit, replace=False))
            dataset.pairs = [dataset.pairs[int(i)] for i in indices]
    return datasets


def selection_score(scores, weights=None):
    """Strict improvements select a checkpoint; ties keep the earlier epoch."""
    value = (.25 * scores["1"] + .25 * scores["2"] + .5 * scores["4"] if weights is None
             else sum(weight * scores[key] for key, weight in weights.items()))
    if not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint selection score")
    return value


def is_new_best(score, best):
    return best is None or score > best


def summary_markdown(report):
    if report.get('infer_ablations'):
        observed = report['evaluation']
        lines = ['# Inference interventions', '',
                 'Observed the same frozen checkpoint under inference interventions; no retraining.',
                 'reset-state reruns init(features, y0) before every update.',
                 'no-mask-feedback passes y0 as the step answer/residual base; each output is y0 + delta_t.',
                 f"Shared encoder+probe seconds: {observed['encoder_seconds']:.6f}; all-mode wall seconds: {observed['wall_seconds']:.6f}.",
                 observed['timing_note'], '']
        for mode, evaluation in observed['ablations'].items():
            local = dict(report, evaluation=evaluation)
            local.pop('infer_ablations')
            lines += [f'## Mode: {mode}', '', summary_markdown(local)]
        if 'y0_answer_keep_residual' in report['infer_ablations']:
            lines += ['', 'Observed y0_answer_keep_residual uses fixed y0 answer tokens, recurrent memory and the current-logit residual base.']
        return '\n'.join(lines)
    if report.get('protocol') == 'val_sel_rep':
        primary = dict(report)
        primary.pop('protocol')
        text = summary_markdown(primary).replace('# Looped refiner v1', '# Controlled refiner experiment', 1)
        text = text.replace('Observed results for the final epoch; no Val model selection was performed.',
                            f"Observed checkpoint at epoch {report['evaluated_epoch']}; selection uses Val-sel only. Final evaluation uses disjoint Val-rep only.", 1)
        weights = report['config']['selection_ks']
        keys = list(weights)
        text += '\n## Val-sel checkpoint selection\n\n'
        text += '| Epoch | y0 A | ' + ' | '.join('K'+k+' A' for k in keys) + ' | Weighted A | New best |\n'
        text += '|' + '---|' * (4+len(keys)) + '\n'
        for row in report['selection_curve']:
            text += '| ' + ' | '.join([str(row['epoch']), f"{row['y0_draft']:.6f}"] + [f"{row[k]:.6f}" for k in keys] + [f"{row['score']:.6f}", str(row['new_best'])]) + ' |\n'
        text += f"\nVal split SHA-256: {report['config']['val_split_sha256']}.\n"
        if report['config']['eval_extra']:
            text += '\n## Val-rep extra diagnostics\n\n'
            rows = []
            for domain, variants in report['evaluation']['groups'].items():
                for key, item in variants.items():
                    extra = item['extra']
                    c, t, f = extra['changes'], extra['tracking'], extra['new_fragments']
                    rows.append([domain, key, c['added_correct'], c['added_wrong'], c['deleted_correct'], c['deleted_wrong'],
                                 c['added_correct_fraction'], c['deleted_correct_fraction'], t['gap_repair_fraction'],
                                 t['terminal_coverage_fraction'], t['new_break_fraction'], f['count'], f['aligned_edge_fraction']])
            text += '\n'.join(y0.markdown_table(['Group','K/reference','Added correct','Added wrong','Deleted correct','Deleted wrong','Correct/added','Correct/deleted','Gap repair','Terminal completion','New break','New fragments','Aligned edges/all edges'], rows))+'\n'
        return text
    if "last_evaluation" in report:
        primary = dict(report)
        primary.pop("last_evaluation")
        last = dict(primary, evaluation=report["last_evaluation"])
        text = summary_markdown(primary).replace("# Looped refiner v1", "# Looped refiner v2", 1)
        text = text.replace("Observed results for the final epoch; no Val model selection was performed.",
                            f"Observed best checkpoint at epoch {report['best_epoch']}; selected on a Val subset included in full Val, so results are optimistic.", 1)
        keys = list(report['config'].get('selection_ks', {'1': .25, '2': .25, '4': .5}))
        text += "\n## Checkpoint selection\n\n| Epoch | y0 A | " + ' | '.join('K'+k+' A' for k in keys) + ' | Weighted A | New best |\n|' + '---|' + '---:|' * (len(keys)+2) + '---|\n'
        for row in report["selection_curve"]:
            text += '| ' + ' | '.join([str(row['epoch']), f"{row['y0_draft']:.6f}"] + [f"{row[k]:.6f}" for k in keys] + [f"{row['score']:.6f}", str(row['new_best'])]) + ' |\n'
        last_text = summary_markdown(last)
        text += "\n## Last checkpoint evaluation\n\n" + last_text[last_text.index("## K curves"):last_text.index("## Training epochs")]
        return text
    lines = [*("WARNING: " + warning for warning in report["warnings"]),
             "# Looped refiner v1", "", "Observed results for the final epoch; no Val model selection was performed.",
             "The y0_draft row is a reference, not a legal loop count. Threshold is fixed at 0.5.", "",
             "## Parameter counts", "", "| Part | Parameters |", "|---|---:|"]
    for name, count in report["parameters"].items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "## K curves", "", "| Group | K/reference | IoU | clDice | BIoU | A | Added/valid | Deleted/valid | Refiner s/image | Peak CUDA bytes |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for domain, variants in report["evaluation"]["groups"].items():
        for key, item in variants.items():
            score = item["scores"]
            lines.append(f"| {domain} | {key} | {score['road_iou']:.6f} | {score['cldice']:.6f} | {score['boundary_iou']:.6f} | {score['A']:.6f} | {item['added_road_fraction']:.6f} | {item['deleted_road_fraction']:.6f} | {item['refiner_seconds_per_image']:.6f} | {item['peak_cuda_bytes']} |")
    lines += ["", "## Structural counts", "", "| Group | K/reference | Gap skeleton pixels | Terminal missing pixels | Whole missed pixels | FP fragments | Predicted components |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for domain, variants in report["evaluation"]["groups"].items():
        for key, item in variants.items():
            values = " | ".join(str(item["structure"][name]) for name in STRUCTURE_KEYS)
            lines.append(f"| {domain} | {key} | {values} |")
    step_headers = " | ".join(f"Loss t={step}" for step in range(1, report["config"]["k_train"] + 1))
    lines += ["", "## Training epochs", "", f"| Epoch | Loss | {step_headers} | Wall s | Encoder+probe s | Refiner forward+backward s | Peak CUDA bytes |",
              "|" + "---|" * (6 + report["config"]["k_train"])]
    for epoch in report["epochs"]:
        losses = " | ".join(f"{value:.6f}" for value in epoch["step_losses"])
        lines.append(f"| {epoch['epoch']} | {epoch['loss']:.6f} | {losses} | {epoch['seconds']:.3f} | {epoch['encoder_seconds']:.3f} | {epoch['refiner_forward_backward_seconds']:.3f} | {epoch['peak_cuda_bytes']} |")
    lines += ["", "## Fixed-subset monitoring (not used for selection)", "",
              "| Epoch | y0_draft A | K=1 A | K=4 A | Seconds |", "|---|---:|---:|---:|---:|"]
    for row in report["train_curve"]:
        lines.append(f"| {row['epoch']} | {row['y0_draft']:.6f} | {row['1']:.6f} | {row['4']:.6f} | {row['seconds']:.3f} |")
    if report['config'].get('noadd_lambda', 0):
        lines += ['', '## No-add training objective', '',
                  f"Observed noadd_lambda={report['config']['noadd_lambda']}. Total loss = mean BCE + lambda * mean penalty.",
                  'Legacy step_losses and explicit step_bce record BCE; step_penalties record the unweighted penalty.', '',
                  '| Epoch | Mean BCE | Step BCE | Mean penalty | Step penalties | Total loss |',
                  '|---|---:|---|---:|---|---:|']
        for row in report['epochs']:
            lines.append(f"| {row['epoch']} | {row['bce']:.9f} | {row['step_bce']} | "
                         f"{row['penalty']:.9f} | {row['step_penalties']} | {row['loss']:.9f} |")
    lines += ["", "Observed timings synchronize CUDA at both boundaries. Refiner timings include initialization but exclude encoder/probe, snapshot copies and metric workers.",
              "Epoch losses are weighted by batch sample count; every batch loss averages valid pixels and supervised steps.",
              "Geometry uses the existing y0 analysis definitions. Counts are summed before official summarize; Overall = Urban + Rural.",
              "CPU peak CUDA memory is reported as None. Fixed positional buffers have zero trainable parameters.", ""]
    return "\n".join(lines)


def check_untied_limit(untied_steps, requested, context):
    """Reject unsupported depths explicitly; never silently shorten a loop."""
    if untied_steps < 0:
        raise ValueError("--untied-steps must be nonnegative")
    if untied_steps and max(requested) > untied_steps:
        raise ValueError(f"{context}={max(requested)} exceeds --untied-steps={untied_steps}")


def validate_untied_settings(args):
    check_untied_limit(args.untied_steps, [args.k_train], "--k-train")
    check_untied_limit(args.untied_steps, args.eval_k, "--eval-k")
    check_untied_limit(args.untied_steps, [int(k) for k in args.selection_ks], "--selection-ks K")
    if not args.eval_only and args.epochs and args.val_split is None:
        check_untied_limit(args.untied_steps, [1, 4], "legacy monitoring K (use --val-split for another selection grid)")


def validate_noadd_lambda(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('--noadd-lambda must be finite and nonnegative')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Zero evaluates an untrained zero-output refiner")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pos-weight", type=float, default=4.,
                        help="Positive-class BCE weight for the refiner only; does not change the frozen probe")
    parser.add_argument('--loss', choices=('bce', 'false_add', 'lovasz', 'soft_cldice'), default='bce')
    parser.add_argument('--aux-weight', type=float, default=.1,
                        help='Coefficient for Lovasz or soft-clDice; false_add is calibrated on Train')
    parser.add_argument('--skeleton-iterations', type=int, default=20)
    parser.add_argument('--false-add-gradient-ratio', type=float, default=.1)
    parser.add_argument('--false-add-warmup-steps', type=int, default=500)
    parser.add_argument('--false-add-calibration-batches', type=int, default=8)
    parser.add_argument('--augmentation', choices=('none', 'd4'), default='none')
    parser.add_argument("--k-train", type=int, default=4)
    parser.add_argument("--eval-k", type=parse_ks, default=parse_ks("1,2,4,6,8"))
    parser.add_argument("--limit-per-domain", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/refiner/v1")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "runs/refiner/v1/checkpoints")
    parser.add_argument("--probe-ckpt", type=Path, default=ROOT / "runs/linear_probe/checkpoints/dinov3_vitl16_web_bn_posw4.pt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true", help="Load --ckpt and evaluate without Train access")
    mode.add_argument("--resume", action="store_true", help="Continue ckpt-dir/last.pt to the total --epochs target")
    parser.add_argument("--ckpt", type=Path, help="Refiner checkpoint for --eval-only")
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--monitor-per-domain", type=int, default=50)
    parser.add_argument("--loader-workers", type=int, default=7)
    parser.add_argument("--eval-loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--no-mask-feedback", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--untied-steps", type=int, default=0,
                        help="Independent cores for a finite stack; zero preserves shared recurrence")
    parser.add_argument('--noadd-lambda', type=float, default=0.,
                        help='Weight of the valid-background probability-increase penalty')
    parser.add_argument("--head-type", choices=("patch_linear", "conv_upsample"), default="patch_linear")
    parser.add_argument("--select-every", type=int, default=0, help="Zero preserves final-epoch-only v1 behavior")
    parser.add_argument("--select-per-domain", type=int, default=100)
    parser.add_argument('--val-split', type=Path)
    parser.add_argument('--selection-ks', type=parse_selection_ks, default='1:0.25,2:0.25,4:0.5')
    parser.add_argument('--eval-extra', action='store_true')
    parser.add_argument('--infer-ablations', type=parse_infer_ablations,
                        help='Eval-only interventions matching the staff predict.py loop')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.infer_ablations is not None and not args.eval_only:
        parser.error('--infer-ablations requires --eval-only')
    if args.eval_only and args.ckpt is None:
        parser.error("--eval-only requires --ckpt")
    if args.ckpt is not None and not args.eval_only:
        parser.error("--ckpt is used with --eval-only; --resume uses ckpt-dir/last.pt")
    restored = None
    if args.resume or args.eval_only:
        source = args.ckpt if args.eval_only else args.ckpt_dir / "last.pt"
        if not source.is_file():
            parser.error(f"Checkpoint not found: {source}")
        restored = torch.load(source, map_location="cpu", weights_only=True)
        if not {"state_dict", "config", "epoch"}.issubset(restored):
            parser.error("Checkpoint lacks refiner weights/config/epoch")
        explicit = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
        if '--eval-extra' not in explicit:
            args.eval_extra = restored['config'].get('eval_extra', False)
        settings = ["dimension", "layers", "heads", "mlp_ratio", "patch", "no_mask_feedback", "reset_state", "k_train", "head_type", "selection_ks", "untied_steps", "noadd_lambda", "pos_weight",
                    'loss', 'aux_weight', 'skeleton_iterations', 'false_add_gradient_ratio',
                    'false_add_warmup_steps', 'false_add_calibration_batches', 'augmentation']
        if args.resume:
            settings += ["batch_size", "lr", "seed", "limit_per_domain", "monitor_every", "monitor_per_domain", "select_every", "select_per_domain"]
        legacy_defaults = {'untied_steps': 0, 'noadd_lambda': 0., 'head_type': 'patch_linear', 'select_every': 0, 'select_per_domain': 100,
                           'selection_ks': parse_selection_ks('1:0.25,2:0.25,4:0.5'), 'pos_weight': 4.,
                           'loss': 'bce', 'aux_weight': .1, 'skeleton_iterations': 20,
                           'false_add_gradient_ratio': .1, 'false_add_warmup_steps': 500,
                           'false_add_calibration_batches': 8, 'augmentation': 'none'}
        for name in settings:
            saved = restored["config"].get(name, legacy_defaults.get(name))
            if "--" + name.replace("_", "-") in explicit and getattr(args, name) != saved:
                parser.error(f"Checkpoint setting {name}={saved!r} conflicts with the requested value")
            setattr(args, name, saved)
        if args.eval_only:
            args.epochs = restored["config"]["epochs"]
            if "--seed" not in explicit:
                args.seed = restored["config"]["seed"]
        if restored["config"]["probe_ckpt"] != str(args.probe_ckpt.resolve().relative_to(ROOT)):
            parser.error("The frozen probe checkpoint must match the training checkpoint config")
        saved_split = restored['config'].get('val_split')
        if saved_split:
            if args.val_split is not None and args.val_split.resolve() != (ROOT/saved_split).resolve():
                parser.error('Checkpoint Val split cannot change')
            args.val_split = ROOT/saved_split
        elif args.resume and args.val_split is not None:
            parser.error('Cannot change a legacy training run to the controlled protocol on resume')
    try:
        validate_untied_settings(args)
        validate_noadd_lambda(args.noadd_lambda)
        if args.noadd_lambda and args.loss != 'bce':
            raise ValueError('--noadd-lambda is a BCE experiment; select one auxiliary objective per run')
    except ValueError as error:
        parser.error(str(error))
    split = None
    split_hash = None
    if args.val_split is not None:
        if not args.val_split.resolve().is_relative_to(ROOT):
            parser.error('Val split must be inside the repository')
        contents = args.val_split.read_bytes()
        split, split_hash = json.loads(contents), hashlib.sha256(contents).hexdigest()
        if restored is not None and restored['config'].get('val_split_sha256', split_hash) != split_hash:
            parser.error('Val split SHA-256 changed since checkpoint creation')
        if args.select_every == 0:
            args.select_every = 1
        if args.select_every != 1 or (args.epochs < 1 and not args.eval_only):
            parser.error('Controlled training requires positive epochs and selection every epoch')
    if args.epochs < 0 or min(args.batch_size, args.k_train, args.monitor_every, args.monitor_per_domain, args.metric_workers) < 1:
        parser.error("Epochs must be nonnegative; batch/loop/monitor/metric settings must be positive")
    if args.infer_ablations is not None and (args.reset_state or args.no_mask_feedback):
        parser.error('Inference interventions require a checkpoint with both trained ablation flags disabled')
    if args.resume and args.epochs < int(restored["epoch"]):
        parser.error("--epochs is the total target and cannot precede the saved epoch")
    if args.lr <= 0 or min(args.loader_workers, args.eval_loader_workers) < 0:
        parser.error("Learning rate must be positive and loader workers nonnegative")
    if not math.isfinite(args.pos_weight) or args.pos_weight <= 0:
        parser.error("--pos-weight must be finite and positive")
    if not math.isfinite(args.aux_weight) or args.aux_weight <= 0:
        parser.error('--aux-weight must be finite and positive')
    if not math.isfinite(args.false_add_gradient_ratio) or args.false_add_gradient_ratio <= 0:
        parser.error('--false-add-gradient-ratio must be finite and positive')
    if min(args.skeleton_iterations, args.false_add_calibration_batches) < 1 or args.false_add_warmup_steps < 0:
        parser.error('Skeleton/calibration counts must be positive and warmup steps nonnegative')
    if args.select_every < 0 or args.select_per_domain < 1:
        parser.error("Selection interval must be nonnegative and subset size positive")
    if args.limit_per_domain is not None and args.limit_per_domain < 1:
        parser.error("--limit-per-domain must be positive")
    slots = int(os.environ.get("NSLOTS", "0"))
    if slots and max(args.loader_workers + 1, args.eval_loader_workers + args.metric_workers + 1) > slots:
        parser.error("Main plus active loader/metric workers exceed NSLOTS")
    if not args.out_dir.resolve().is_relative_to(ROOT) or not args.ckpt_dir.resolve().is_relative_to(ROOT / "runs"):
        parser.error("Reports must stay in the repository; checkpoints must stay under runs/")
    if (args.out_dir / "metrics.json").exists() or (args.out_dir / "summary.md").exists():
        parser.error("Refusing to overwrite existing reports; choose a fresh --out-dir")
    if args.epochs and not args.resume and not args.eval_only and (args.ckpt_dir / "last.pt").exists():
        parser.error("Refusing to overwrite a checkpoint from an earlier run")
    if args.select_every and not args.resume and not args.eval_only and (args.ckpt_dir / "best.pt").exists():
        parser.error("Refusing to overwrite an existing best checkpoint")
    probe.initialize_data_worker(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    encoder, head, checkpoint = y0.load_frozen_probe(args.probe_ckpt, device)
    head.requires_grad_(False).eval()
    encoder.requires_grad_(False).eval()
    model = LoopRefiner(encoder.out_channels, args.dimension, args.layers, args.heads,
                        args.mlp_ratio, args.patch, no_mask_feedback=args.no_mask_feedback,
                        reset_state=args.reset_state, head_type=args.head_type,
                        untied_steps=args.untied_steps).to(device=device, dtype=torch.float32)
    if restored is not None:
        model.load_state_dict(restored["state_dict"])
    parameters = model.parameter_counts()
    print("PARAMETERS " + json.dumps(parameters), flush=True)
    frozen_parameters = {"encoder": sum(p.numel() for p in encoder.parameters()),
                         "probe": sum(p.numel() for p in head.parameters())}
    if any(p.requires_grad for p in (*encoder.parameters(), *head.parameters())):
        raise AssertionError("Encoder and probe must be completely frozen")
    config = {key: (str(value.resolve().relative_to(ROOT)) if value.resolve().is_relative_to(ROOT) else value.name)
              if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(weight_decay=.01, warmup_steps=500, clip_grad_norm=1.,
                  refiner_dtype="float32", encoder=y0.ENCODER, probe_head=checkpoint["head"])
    if args.infer_ablations is None:
        config.pop('infer_ablations')
    objective = RefinerObjective(args.loss, args.pos_weight, args.aux_weight, args.skeleton_iterations,
                                 args.false_add_gradient_ratio, args.false_add_warmup_steps,
                                 args.false_add_calibration_batches,
                                 state=restored['config'].get('objective_state') if restored else None)
    config.update(loss_implementation_version=1, objective_state=objective.state)
    if split is not None:
        config['val_split_sha256'] = split_hash
    train_sets = ([probe.RoadDataset("Train", domain, limit=args.limit_per_domain, seed=args.seed + i)
                   for i, domain in enumerate(DOMAINS)] if args.epochs and not args.eval_only else [])
    if args.augmentation == 'd4':
        train_sets = [D4TrainingDataset(dataset, args.seed + i) for i, dataset in enumerate(train_sets)]
    loader = (make_loader(ConcatDataset(train_sets), args.batch_size, args.loader_workers, True, args.seed)
              if train_sets else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01) if loader else None
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, len(loader) * args.epochs))
                 if loader else None)
    start_epoch = int(restored["epoch"]) if restored is not None else 0
    epochs = list(restored.get("epochs", [])) if restored is not None else []
    curve = list(restored.get("train_curve", [])) if restored is not None else []
    selection_curve = list(restored.get("selection_curve", [])) if restored is not None else []
    best_score = restored.get("best_score") if restored is not None else None
    best_epoch = restored.get("best_epoch") if restored is not None else None
    selection_sets = ((protocol_datasets(split, 'Val-sel', args.select_per_domain) if split is not None else
                       {domain: probe.RoadDataset("Val", domain, limit=args.select_per_domain, seed=0) for domain in DOMAINS})
                      if args.select_every and not args.eval_only else {})
    selection_files = {domain: [pair[0].name for pair in dataset.pairs] for domain, dataset in selection_sets.items()}
    if split is not None and not selection_files and restored is not None:
        selection_files = restored.get('selection_files', {})
    config["scheduler_total_steps"] = (len(loader) * args.epochs if loader else
                                        restored["config"].get("scheduler_total_steps") if restored else None)
    if args.resume:
        if loader is None:
            parser.error("--resume requires a positive total --epochs target")
        if not {"optimizer", "scheduler", "rng"}.issubset(restored):
            parser.error("Resume requires optimizer, scheduler and RNG states")
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        restore_rng(restored["rng"], loader, device)
        print(f"RESUME loaded_epoch={start_epoch} next_epoch={start_epoch + 1} target_epochs={args.epochs} "
              f"scheduler_last_epoch={scheduler.last_epoch} lr={optimizer.param_groups[0]['lr']:.12g} "
              "restored=refiner,optimizer,scheduler,python_rng,numpy_rng,torch_rng,loader_rng "
              f"cuda_rng_states={len(restored['rng']['cuda'])}", flush=True)
        old_horizon = restored["config"].get("scheduler_total_steps")
        if old_horizon != config["scheduler_total_steps"]:
            print(f"RESUME scheduler_total_steps={old_horizon}->{config['scheduler_total_steps']} "
                  "(changed total epoch target; completed steps and current LR preserved)", flush=True)
    if args.eval_only:
        print(f"EVAL_ONLY loaded_epoch={start_epoch}; Train is not constructed", flush=True)
    for epoch in range(start_epoch, args.epochs) if not args.eval_only else ():
        for dataset in train_sets:
            if isinstance(dataset, D4TrainingDataset):
                dataset.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = stamp(device)
        batches, seen = [], 0
        for batch_index, (images, targets, valid, _, _) in enumerate(loader):
            images, targets, valid = (probe.as_device_tensor(value, device) for value in (images, targets, valid))
            optimizer.zero_grad(set_to_none=True)
            phase = stamp(device)
            features, draft = frozen_draft(encoder, head, images)
            encoder_time = stamp(device) - phase
            phase = stamp(device)
            with torch.autocast(device_type=device.type, enabled=False):
                logits = refine_steps(model, features.float(), draft.float(), args.k_train)
                if args.loss == 'bce':
                    loss, losses, bce, penalty, penalties = refiner_objective(
                        logits, targets, valid, draft, args.noadd_lambda, pos_weight=args.pos_weight)
                    loss_terms = {'bce': float(bce.detach()),
                                  'auxiliary': float(penalty.detach()) if args.noadd_lambda else 0.,
                                  'auxiliary_weight_used': args.noadd_lambda, 'phase': 'active'}
                else:
                    loss, losses, loss_terms = objective(logits, targets, valid, draft,
                                                         step=epoch * len(loader) + batch_index + 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
            refiner_time = stamp(device) - phase
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            batch = {"batch": batch_index + 1, "samples": images.shape[0], "loss": float(loss.detach()),
                     "step_losses": [float(value.detach()) for value in losses],
                     "encoder_seconds": encoder_time, "refiner_forward_backward_seconds": refiner_time,
                     "lr": learning_rate, "gradient_norm_before_clip": float(gradient_norm),
                     "loss_terms": loss_terms}
            if args.loss == 'bce':
                batch.update(bce=float(bce.detach()), step_bce=batch['step_losses'],
                             penalty=float(penalty.detach()) if isinstance(penalty, torch.Tensor) else penalty,
                             step_penalties=[float(value.detach()) if isinstance(value, torch.Tensor) else value
                                             for value in penalties])
                del bce, penalty, penalties
            batches.append(batch)
            seen += images.shape[0]
            print(f"BATCH epoch={epoch + 1} " + json.dumps(batch), flush=True)
            del logits, loss, losses, features, draft
        record = {"epoch": epoch + 1, "loss": sum(row["loss"] * row["samples"] for row in batches) / seen,
                  "step_losses": [sum(row["step_losses"][step] * row["samples"] for row in batches) / seen
                                  for step in range(args.k_train)],
                  "seconds": stamp(device) - epoch_started,
                  "encoder_seconds": sum(row["encoder_seconds"] for row in batches),
                  "refiner_forward_backward_seconds": sum(row["refiner_forward_backward_seconds"] for row in batches),
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "batches": batches}
        if args.loss == 'bce':
            record.update(bce=sum(row['bce'] * row['samples'] for row in batches) / seen,
                          step_bce=record['step_losses'],
                          penalty=sum(row['penalty'] * row['samples'] for row in batches) / seen,
                          step_penalties=[sum(row['step_penalties'][step] * row['samples'] for row in batches) / seen
                                          for step in range(args.k_train)])
        epochs.append(record)
        print("EPOCH " + json.dumps({key: value for key, value in record.items() if key != "batches"}), flush=True)
        if split is None and (epoch == 0 or (epoch + 1) % args.monitor_every == 0):
            monitor = {domain: probe.RoadDataset("Val", domain) for domain in DOMAINS}
            for dataset in monitor.values():
                dataset.pairs = dataset.pairs[:args.monitor_per_domain]
            observed = evaluate(encoder, head, model, monitor, [1, 4], device,
                                args.eval_loader_workers, args.metric_workers, geometry=False)
            point = {"epoch": epoch + 1, "seconds": observed["wall_seconds"],
                     "images_per_domain": {domain: len(dataset) for domain, dataset in monitor.items()},
                     "selection": "first sorted filenames; monitoring only"}
            point.update({key: observed["groups"]["Overall"][key]["scores"]["A"] for key in ("y0_draft", "1", "4")})
            curve.append(point)
            print("MONITOR " + json.dumps(point), flush=True)
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        improved = False
        if args.select_every and ((epoch + 1) % args.select_every == 0 or epoch + 1 == args.epochs):
            selected = evaluate(encoder, head, model, selection_sets, [int(k) for k in args.selection_ks], device,
                                args.eval_loader_workers, args.metric_workers, geometry=False)
            point = {"epoch": epoch + 1, "seconds": selected["wall_seconds"]}
            point.update({key: selected["groups"]["Overall"][key]["scores"]["A"] for key in ('y0_draft', *args.selection_ks)})
            point["score"] = selection_score(point, args.selection_ks)
            improved = is_new_best(point["score"], best_score)
            point["new_best"] = improved
            if improved:
                best_score, best_epoch = point["score"], epoch + 1
            selection_curve.append(point)
            print("SELECTION " + json.dumps(point), flush=True)
        payload = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "config": config,
                    "epochs": epochs, "train_curve": curve, "rng": capture_rng(loader)}
        if args.select_every:
            payload.update(selection_curve=selection_curve, best_score=best_score, best_epoch=best_epoch,
                           selection_files=selection_files)
        torch.save(payload, args.ckpt_dir / "last.pt")
        if improved:
            torch.save(payload, args.ckpt_dir / "best.pt")
    if args.loss == 'false_add' and not args.eval_only and objective.state.get('effective_aux_weight') is None:
        raise RuntimeError('Training ended before false-add calibration completed; no final report will be written')
    del loader, train_sets
    datasets = (protocol_datasets(split, 'Val-rep', args.limit_per_domain) if split is not None else
                {domain: probe.RoadDataset("Val", domain, limit=args.limit_per_domain, seed=args.seed + i + 10) for i, domain in enumerate(DOMAINS)})
    if split is None and args.limit_per_domain is None and sum(map(len, datasets.values())) != 1669:
        raise ValueError("Full Val must contain 1669 images")
    if split is not None and not args.eval_only:
        best = torch.load(args.ckpt_dir/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(best['state_dict'])
    evaluation_options = {} if args.infer_ablations is None else {'infer_ablations': args.infer_ablations}
    observed = evaluate(encoder, head, model, datasets, args.eval_k, device,
                        args.eval_loader_workers, args.metric_workers, eval_extra=args.eval_extra, **evaluation_options)
    last_observed = None
    if split is None and args.select_every and not args.eval_only:
        last_observed = observed
        best = torch.load(args.ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"])
        observed = evaluate(encoder, head, model, datasets, args.eval_k, device,
                            args.eval_loader_workers, args.metric_workers, eval_extra=args.eval_extra)
    warnings = []
    draft_a = observed["groups"]["Overall"]["y0_draft"]["scores"]["A"]
    regression = "not checked: CPU or limited subset"
    if split is not None:
        regression = 'not checked: Val-rep differs from full Val'
    if split is None and device.type == "cuda" and args.limit_per_domain is None:
        regression = "passed" if f"{draft_a:.6f}" == "0.526507" else "warning"
        if regression == "warning":
            warnings.append(f"GPU y0_draft Overall A@0.5={draft_a:.6f} (actual {draft_a:.12f}), expected 0.526507.")
    for warning in warnings:
        print("WARNING: " + warning, flush=True)
    report = {"config": config, "parameters": parameters, "frozen_parameters": frozen_parameters,
              "epochs": epochs, "train_curve": curve, "evaluation": observed,
              "warnings": warnings, "y0_gpu_regression": regression,
              "loaded_epoch": start_epoch if restored is not None else None,
              "evaluated_epoch": epochs[-1]["epoch"] if epochs else start_epoch,
              "model_selection": "loaded checkpoint" if args.eval_only else "final epoch only; monitor scores never select weights"}
    if last_observed is not None:
        report.update(last_evaluation=last_observed, best_epoch=best_epoch, best_score=best_score,
                      selection_curve=selection_curve, selection_files=selection_files,
                      evaluated_epoch=best_epoch,
                      model_selection="best weighted K1/K2/K4 score on fixed Val subset; optimistic full-Val estimate")
    if split is not None:
        report.update(protocol='val_sel_rep', best_epoch=best_epoch, best_score=best_score,
                      selection_curve=selection_curve, selection_files=selection_files,
                      evaluated_epoch=start_epoch if args.eval_only else best_epoch,
                      evaluation_files={d: [a.name for a, _ in dataset.pairs] for d, dataset in datasets.items()},
                      model_selection='loaded checkpoint on Val-rep' if args.eval_only else 'best weighted score on Val-sel; report only on Val-rep')
    if args.infer_ablations is not None:
        report['infer_ablations'] = args.infer_ablations
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print("FINAL_OVERALL_A " + json.dumps({key: row["scores"]["A"] for key, row in observed["groups"]["Overall"].items()}), flush=True)
    if args.infer_ablations is not None:
        for mode, evaluation in observed['ablations'].items():
            print('INFER_OVERALL_A ' + mode + ' ' + json.dumps({key: row['scores']['A'] for key, row in evaluation['groups']['Overall'].items()}), flush=True)
    print(f"Wrote {args.out_dir / 'metrics.json'} and summary.md", flush=True)


if __name__ == "__main__":
    main()
