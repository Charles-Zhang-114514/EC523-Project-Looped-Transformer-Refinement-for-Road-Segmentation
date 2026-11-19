#!/usr/bin/env python3
"""Evaluate frozen draft heads with D4 TTA and the e6 seed-0 refiner."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from analysis import draft_overfit_check as previous
from scripts import train_refiner as runner


DOMAINS = ("Overall", *runner.DOMAINS)
KS = (1, 2, 4, 8)
SCORE_KEYS = ("road_iou", "cldice", "boundary_iou", "quality")
BASELINE = ("A1-i", 0)
ARMS = ("A1-i", "v3a", "v3b")
MATRIX = [(arm, seed) for arm in ARMS for seed in (0, 1)]
SPLIT = ROOT / "configs/val_split_seed0.json"
E6_REPORT = ROOT / "results/exp/e6_convdraft_s0/metrics.json"


def d4(array, code, inverse=False):
    """Match training D4: rotate first, then optionally flip the width axis."""
    if code not in range(8):
        raise ValueError("D4 code must be in 0..7")
    if inverse:
        if code >= 4:
            array = array.flip((-1,))
        return array.rot90(-(code % 4), (-2, -1))
    rotated = array.rot90(code % 4, (-2, -1))
    return rotated.flip((-1,)) if code >= 4 else rotated


@torch.no_grad()
def average_d4_logits(predict, image, identity):
    """Reuse the original-view logits and accumulate aligned float32 logits."""
    total = identity.float().clone()
    for code in range(1, 8):
        logits = predict(d4(image, code).contiguous()).float()
        total.add_(d4(logits, code, inverse=True))
    return total.div_(8)


def binary_mask(logits):
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite inference logits")
    return (logits.sigmoid()[0, 0] >= .5).cpu().numpy().astype(np.uint8)


def evaluate(encoder, head, model, datasets, ks, device, loader_workers, metric_workers, tta):
    """Stream binary masks to official metrics; keep feature maps batch-local."""
    metrics = runner.probe.official_metrics()
    modes = ("none", "d4") if tta == "d4" else ("none",)
    keys = ("y0_draft", *map(str, ks))
    counts = {mode: {domain: {key: metrics.Counts() for key in keys}
                     for domain in runner.DOMAINS} for mode in modes}
    pending, images = deque(), []

    def consume():
        domain, values = pending.popleft().get()
        for mode, by_step in values.items():
            for key, value in by_step.items():
                counts[mode][domain][key] += metrics.Counts(**value)

    with mp.get_context("spawn").Pool(metric_workers, initializer=runner.initialize_eval_worker,
                                     initargs=(os.environ["EC523_STARTER_DIR"],)) as pool:
        for domain in runner.DOMAINS:
            loader = runner.make_loader(datasets[domain], 1, loader_workers)
            for images_np, _, _, ids, labels in loader:
                image = runner.probe.as_device_tensor(images_np, device)
                started = runner.stamp(device)
                features, draft = runner.frozen_draft(encoder, head, image)
                single_seconds = runner.stamp(device) - started
                logits = {"none": draft}
                times = {"none": single_seconds}
                if tta == "d4":
                    started = runner.stamp(device)
                    logits["d4"] = average_d4_logits(lambda view: head(encoder(view)), image, draft)
                    times["d4"] = single_seconds + runner.stamp(device) - started

                predictions = {}
                refiner_times = {}
                for mode, answer in logits.items():
                    predictions[mode] = {"y0_draft": binary_mask(answer)}
                    if ks:
                        refiner_times[mode] = {}
                        for key, refined, seconds, _ in runner.inference_steps(model, features, answer, ks, device):
                            predictions[mode][key] = binary_mask(refined)
                            refiner_times[mode][key] = seconds

                images.append(dict(domain=domain, image_id=ids[0], draft_seconds=times,
                                   refiner_seconds=refiner_times))
                raw = np.asarray(labels).squeeze(0).astype(np.uint8, copy=False)
                pending.append(pool.apply_async(previous.score_predictions, (domain, predictions, raw)))
                if len(pending) >= 2 * metric_workers:
                    consume()
            del loader
        while pending:
            consume()

    for by_domain in counts.values():
        by_domain["Overall"] = {key: by_domain["Urban"][key] + by_domain["Rural"][key] for key in keys}
    return {
        "images": images,
        "groups": {mode: {domain: {key: {"counts": asdict(value), "scores": metrics.summarize(value)}
                                   for key, value in by_key.items()}
                          for domain, by_key in by_domain.items()}
                   for mode, by_domain in counts.items()},
    }


def paths(arm, seed):
    if (arm, seed) == BASELINE:
        return ROOT / "results/draft_head/A1-i", ROOT / previous.PROBES["A1-i"]
    run = f"{arm}_s{seed}"
    return ROOT / "runs/draft_head_v3" / run / "report", ROOT / "runs/draft_head_v3" / run / "checkpoints/best.pt"


def selected_names(dataset):
    return {domain: [image.name for image, _ in dataset[domain].pairs] for domain in runner.DOMAINS}


def load_training(datasets, split_hash):
    reports = {}
    selection_files = selected_names(datasets["Val-sel"])
    report_files = selected_names(datasets["Val-rep"])
    for arm, seed in MATRIX:
        report_dir, checkpoint = paths(arm, seed)
        required = [report_dir / "summary.md", checkpoint]
        if (arm, seed) != BASELINE:
            required.append(checkpoint.with_name("last.pt"))
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed training artifact: {path.relative_to(ROOT)}")
        raw = json.loads((report_dir / "metrics.json").read_text())
        config = raw["config"]
        expected = dict(draft_layers="last", head="conv_up", seed=seed, epochs=10, batch_size=8,
                        lr=1e-3, pos_weight=4., weight_decay=0., select_per_domain=100,
                        limit_per_domain=None, threshold=.5, val_split_sha256=split_hash)
        if (any(config.get(k) != v for k, v in expected.items())
                or config.get("lr_schedule", "constant") != ("constant" if arm == "A1-i" else "cosine")
                or config.get("d4_aug", False) != (arm == "v3b")):
            raise ValueError(f"Unexpected training configuration for {arm} seed {seed}")
        epochs = raw["epochs"]
        if ([entry["epoch"] for entry in epochs] != list(range(1, 11))
                or any(entry["selection"]["files"] != selection_files for entry in epochs)
                or raw["evaluation"]["files"] != report_files):
            raise ValueError(f"Unexpected validation protocol for {arm} seed {seed}")
        best = max(epochs, key=lambda entry: entry["selection_A"])
        if (raw["best_epoch"], raw["best_selection_A"]) != (best["epoch"], best["selection_A"]):
            raise ValueError(f"Unexpected checkpoint selection for {arm} seed {seed}")
        reports[arm, seed] = dict(arm=arm, seed=seed, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                                  best_epoch=raw["best_epoch"], best_selection_A=raw["best_selection_A"],
                                  val_sel_curve=[entry["selection_A"] for entry in epochs],
                                  val_rep=raw["evaluation"]["groups"])
    return reports


def write_report(out, report, markdown):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (out / "summary.md").open("x") as stream:
        stream.write(markdown)


def compare_refiner(datasets, device, args):
    """Compare the non-TTA evaluator to the unchanged evaluation entry point."""
    _, checkpoint = paths(*BASELINE)
    encoder, head, _ = runner.y0.load_frozen_probe(checkpoint, device)
    refiner, _ = previous.shared.load_model(ROOT / "runs/exp/e6_convdraft_s0/best.pt", device,
                                            hashlib.sha256(SPLIT.read_bytes()).hexdigest())
    ks = (1, 2, 4)
    current = evaluate(encoder, head, refiner, datasets, ks, device, args.loader_workers,
                       args.metric_workers, args.tta)
    legacy = runner.evaluate(encoder, head, refiner, datasets, ks, device, args.loader_workers,
                             args.metric_workers, geometry=False)
    if [{"domain": row["domain"], "image_id": row["image_id"]}
            for row in current["images"]] != legacy["images"]:
        raise ValueError("Refiner comparison used different images")
    rows = []
    for domain in DOMAINS:
        for key in ("y0_draft", *map(str, ks)):
            left = current["groups"]["none"][domain][key]
            right = legacy["groups"][domain][key]
            differences = {name: left["scores"][name] - right["scores"][name] for name in SCORE_KEYS}
            rows.append(dict(domain=domain, step=key, differences=differences,
                             counts_equal=left["counts"] == right["scores"]["counts"]))
    maximum = max(abs(value) for row in rows for value in row["differences"].values())
    report = dict(smoke=True, maximum_absolute_difference=maximum, comparisons=rows,
                  images=current["images"], current=current["groups"], legacy=legacy["groups"])
    write_report(args.out_dir, report, "# Observed CPU refiner regression\n\n"
                 f"Maximum absolute score difference across {len(rows)} rows: {maximum:.12g}.\n")
    if maximum > 1e-5:
        raise ValueError(f"Refiner regression difference {maximum:.12g} exceeds 1e-5")
    print(f"PASS refiner regression: {len(rows)} rows, max absolute score difference {maximum:.12g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tta", choices=("none", "d4"), default="none")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--loader-workers", type=int, default=1)
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--refiner-regression", type=int, metavar="IMAGES_PER_DOMAIN")
    parser.add_argument("--ensemble-bias", action="store_true")
    parser.add_argument("--heads", nargs="+", type=Path)
    args = parser.parse_args()
    slots = int(os.environ.get("NSLOTS", "0"))
    if args.loader_workers < 0 or args.metric_workers < 1 or (slots and 1 + args.loader_workers + args.metric_workers > slots):
        parser.error("Invalid loader/metric worker budget")
    if not args.out_dir.resolve().is_relative_to(ROOT) or (args.out_dir.exists() and any(args.out_dir.iterdir())):
        parser.error("Choose a fresh output directory inside the repository")
    if args.refiner_regression is not None and (
        args.refiner_regression < 1 or args.device != "cpu" or not args.out_dir.resolve().is_relative_to(ROOT / "runs")
    ):
        parser.error("The regression requires a positive image limit, CPU and output under runs")
    if args.refiner_regression is None and args.tta != "d4":
        parser.error("The full comparison requires --tta d4")

    if args.ensemble_bias:
        return ensemble_bias_run(args)
    split_bytes = SPLIT.read_bytes()
    split, split_hash = json.loads(split_bytes), hashlib.sha256(split_bytes).hexdigest()
    runner.probe.initialize_data_worker(0)
    device = torch.device(args.device)
    if args.refiner_regression is not None:
        return compare_refiner(runner.protocol_datasets(split, "Val-rep", args.refiner_regression), device, args)
    from analysis import summarize_draft_head_v3 as summary
    datasets = {part: runner.protocol_datasets(split, part, limit)
                for part, limit in (("Val-sel", 100), ("Val-rep", None))}
    training = load_training(datasets, split_hash)
    reference = json.loads(E6_REPORT.read_text())
    if (reference["config"]["val_split_sha256"] != split_hash or reference["config"]["seed"] != 0
            or reference["config"]["probe_ckpt"] != previous.PROBES["A1-i"]):
        raise ValueError("Archived e6 report does not match the expected seed, draft or split")
    model, epoch = previous.shared.load_model(ROOT / "runs/exp/e6_convdraft_s0/best.pt", device, split_hash)
    if epoch != reference["best_epoch"]:
        raise ValueError("e6 checkpoint epoch differs from the archived report")

    # All six heads use the same frozen encoder. Each measurement runs a
    # standalone draft pipeline to retain real per-image timings.
    encoder = None
    for arm, seed in MATRIX:
        _, checkpoint = paths(arm, seed)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        row = training[arm, seed]
        if (saved["encoder"] != runner.y0.ENCODER or saved["head"] != "conv_up"
                or saved["draft_layers"] != "last" or saved["config"]["seed"] != seed
                or saved["epoch"] != row["best_epoch"]):
            raise ValueError(f"Unexpected best checkpoint for {arm} seed {seed}")
        if encoder is None:
            encoder, head, _ = runner.y0.load_frozen_probe(checkpoint, device)
        else:
            head.load_state_dict(saved["state_dict"], strict=True)
        row["tta"] = {}
        for part, dataset in datasets.items():
            use_refiner = (arm, seed) == BASELINE and part == "Val-rep"
            row["tta"][part] = evaluate(encoder, head, model if use_refiner else None, dataset,
                                         KS if use_refiner else (), device, args.loader_workers,
                                         args.metric_workers, args.tta)
            print(f"OBSERVED {arm} seed {seed} {part}: {len(row['tta'][part]['images'])} images", flush=True)
    report = summary.build_report(training, reference["evaluation"]["groups"], split_hash)
    write_report(args.out_dir, report, summary.markdown(report))




@torch.no_grad()
def ensemble_drafts(encoder, heads, image):
    """Encode eight views once each; retain each head's float32 aligned mean."""
    encoded = encoder(image)
    features = heads[0].refiner_features(encoded) if hasattr(heads[0], "refiner_features") else encoded
    totals = [head(encoded).float().clone() for head in heads]
    for code in range(1, 8):
        encoded = encoder(d4(image, code).contiguous())
        for total, head in zip(totals, heads):
            total.add_(d4(head(encoded).float(), code, inverse=True))
    return features, [total.div_(8) for total in totals]


def output_bias(logits, bias):
    """Keep the recurrent state untouched and calibrate only exported logits."""
    return logits if bias == 0 else logits + bias


def bias_counts(domain, cache_path, label_path, biases):
    labels = np.load(label_path, allow_pickle=False)
    logits = np.load(cache_path, allow_pickle=False)
    predictions = {}
    for bias in biases:
        if bias == 0:
            continue
        predictions[str(bias)] = {
            str(step): (torch.from_numpy(logits[index] + bias).sigmoid().numpy() >= .5).astype(np.uint8)
            for index, step in enumerate(KS)
        }
    return previous.score_predictions(domain, predictions, labels)


def proxy(groups):
    return sum(weight * groups[str(k)]["scores"]["quality"] for weight, k in ((.2, 1), (.2, 2), (.6, 4)))


def ensemble_bias_run(args):
    """Stream E0/E1/E2 inference and score cached output logits through official workers."""
    import tempfile
    metrics = runner.probe.official_metrics()
    device = torch.device(args.device)
    split = json.loads(SPLIT.read_text())
    checkpoint_paths = args.heads or [paths(arm, seed)[1] for arm, seed in MATRIX]
    if len(checkpoint_paths) < 2:
        raise ValueError("E0/E1/E2 comparison requires at least two heads")
    encoder, first, _ = runner.y0.load_frozen_probe(checkpoint_paths[0], device)
    heads = [first]
    import copy
    for path in checkpoint_paths[1:]:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["encoder"] != runner.y0.ENCODER or saved["head"] != "conv_up" or saved["draft_layers"] != "last":
            raise ValueError("All ensemble heads must use the same last-layer conv_up encoder")
        head = copy.deepcopy(first)
        head.load_state_dict(saved["state_dict"], strict=True)
        heads.append(head)
    model, _ = previous.shared.load_model(ROOT/"runs/exp/e6_convdraft_s0/best.pt", device,
                                          hashlib.sha256(SPLIT.read_bytes()).hexdigest())
    regression = []
    for domain, dataset in runner.protocol_datasets(split, "Val-sel", 1).items():
        image = runner.probe.as_device_tensor(dataset[0][0][None], device)
        with torch.no_grad():
            original = heads[0](encoder(image))
            reference = average_d4_logits(lambda view: heads[0](encoder(view)), image, original)
            _, actual = ensemble_drafts(encoder, heads[:1], image)
        difference = float((actual[0]-reference).abs().max())
        masks_equal = np.array_equal(binary_mask(actual[0]), binary_mask(reference))
        if difference > 1e-5 or not masks_equal:
            raise ValueError("Single-head uncalibrated TTA regression failed")
        regression.append(dict(domain=domain, maximum_logit_difference=difference, masks_equal=masks_equal))
    print("REGRESSION " + json.dumps(regression), flush=True)
    groups = {"E0": [0], "E1": [0, 1], "E2": list(range(len(heads)))}
    biases = [round(index / 10, 1) for index in range(-15, 16)]
    chosen = None
    selected_bias = None
    report = dict(heads=[str(path.resolve().relative_to(ROOT)) for path in checkpoint_paths],
                  split_sha256=hashlib.sha256(SPLIT.read_bytes()).hexdigest(), regression=regression, splits={})

    def buckets(names):
        return {name: {d: {str(k): metrics.Counts() for k in KS} for d in runner.DOMAINS} for name in names}

    def add(target, result):
        domain, values = result
        for name, steps in values.items():
            for k, counts in steps.items():
                target[name][domain][k] += metrics.Counts(**counts)

    def summarize(target):
        for name in target:
            target[name]["Overall"] = {k: target[name]["Urban"][k]+target[name]["Rural"][k] for k in map(str, KS)}
        return {name: {d: {k: dict(counts=asdict(c), scores=metrics.summarize(c)) for k,c in steps.items()}
                       for d,steps in domains.items()} for name,domains in target.items()}

    with tempfile.TemporaryDirectory(prefix="ec523-wave1-", dir=os.environ["TMPDIR"]) as temporary:
        cache = Path(temporary)
        for part in ("Val-sel", "Val-rep"):
            datasets = runner.protocol_datasets(split, part)
            counts = buckets(groups)
            pending, files, timings = deque(), [], []
            biased = buckets([str(selected_bias)]) if part == "Val-rep" else None
            cached = []
            with mp.get_context("spawn").Pool(args.metric_workers, initializer=runner.initialize_eval_worker,
                                             initargs=(os.environ["EC523_STARTER_DIR"],)) as pool:
                for domain in runner.DOMAINS:
                    for images, _, _, ids, labels in runner.make_loader(datasets[domain], 1, args.loader_workers):
                        image = runner.probe.as_device_tensor(images, device)
                        started = runner.stamp(device)
                        features, drafts = ensemble_drafts(encoder, heads, image)
                        predictions, final_bias = {}, {}
                        label = np.asarray(labels).squeeze(0).astype(np.uint8)
                        image_no = len(files)
                        label_path = cache/f"labels-{image_no}.npy"
                        if part == "Val-sel":
                            np.save(label_path, label)
                        for name, indices in groups.items():
                            draft = drafts[0] if len(indices) == 1 else torch.stack([drafts[i] for i in indices]).mean(0)
                            predictions[name] = {}
                            saved_logits = []
                            for k, logits, _, _ in runner.inference_steps(model, features, draft, KS, device):
                                predictions[name][k] = binary_mask(logits)
                                if part == "Val-sel":
                                    saved_logits.append(logits[0, 0].float().cpu().numpy())
                                elif name == chosen:
                                    final_bias[k] = binary_mask(output_bias(logits, selected_bias))
                            if part == "Val-sel":
                                path = cache/f"{name}-{image_no}.npy"
                                np.save(path, np.stack(saved_logits))
                                cached.append((name, domain, path, label_path))
                        files.append(dict(domain=domain, image_id=ids[0]))
                        timings.append(runner.stamp(device)-started)
                        pending.append(pool.apply_async(previous.score_predictions, (domain, predictions, label)))
                        if biased is not None:
                            pending.append(pool.apply_async(previous.score_predictions,
                                                            (domain, {str(selected_bias): final_bias}, label)))
                        while len(pending) >= 2*args.metric_workers:
                            result = pending.popleft().get()
                            add(biased if str(selected_bias) in result[1] else counts, result)
                while pending:
                    result = pending.popleft().get()
                    add(biased if str(selected_bias) in result[1] else counts, result)
                scored = summarize(counts)
                report["splits"][part] = dict(groups=scored, images=files, forward_seconds=timings)
                if part == "Val-sel":
                    scores = {name: proxy(domains["Overall"]) for name, domains in scored.items()}
                    best = max(scores, key=scores.get)
                    chosen = best if scores[best]-scores["E0"] >= .001 else "E0"
                    sweep = buckets(map(str, biases))
                    for name, domain, path, label_path in cached:
                        if name == chosen:
                            pending.append(pool.apply_async(bias_counts, (domain, path, label_path, biases)))
                        while len(pending) >= 2*args.metric_workers:
                            add(sweep, pending.popleft().get())
                    while pending:
                        add(sweep, pending.popleft().get())
                    sweep["0.0"] = counts[chosen]
                    sweep_scored = summarize(sweep)
                    bias_scores = {b: proxy(sweep_scored[str(b)]["Overall"]) for b in biases}
                    # Ties prefer zero/smaller absolute calibration.
                    selected_bias = max(biases, key=lambda b: (bias_scores[b], -abs(b), -b))
                    report.update(selected_ensemble=chosen, selection_L=scores,
                                  selected_bias=selected_bias, bias_selection_L=bias_scores,
                                  bias_selection_groups=sweep_scored)
                else:
                    report["biased_val_rep"] = summarize(biased)[str(selected_bias)]
            print(f"OBSERVED {part} {len(files)} images", flush=True)
        rep = report["splits"]["Val-rep"]["groups"]
        scores = {name: proxy(domains["Overall"]) for name, domains in rep.items()}
        gain = proxy(report["biased_val_rep"]["Overall"])-scores[chosen]
        report["observations"] = dict(rep_L=scores, bias_rep_gain=gain,
             A_P1=.001 <= scores["E1"]-scores["E0"] <= .004,
             A_P2=0 <= scores["E2"]-scores["E1"] <= .002,
             A_P3=-.6 <= selected_bias <= -.1 and .001 <= gain <= .003,
             adopted_bias=selected_bias if gain >= .001 else 0.)
    lines = ["# Observed ensemble and output-bias results", "",
             f"Observed selected ensemble {chosen}, Val-sel bias {selected_bias}, Val-rep gain {gain:.12g}.",
             "Observed inference adds bias only after each completed trajectory.", "",
             "| Split | Ensemble | Domain | K | IoU | clDice | BIoU | A |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for part, result in report["splits"].items():
        for name, domains in result["groups"].items():
            for domain, steps in domains.items():
                for k, value in steps.items():
                    values = " | ".join(f"{value['scores'][key]:.9f}" for key in SCORE_KEYS)
                    lines.append(f"| {part} | {name} | {domain} | {k} | {values} |")
    for domain, steps in report["biased_val_rep"].items():
        for k,value in steps.items():
            values = " | ".join(f"{value['scores'][key]:.9f}" for key in SCORE_KEYS)
            lines.append(f"| Val-rep | {chosen} biased | {domain} | {k} | {values} |")
    write_report(args.out_dir, report, "\n".join(lines)+"\n")



if __name__ == "__main__":
    main()
