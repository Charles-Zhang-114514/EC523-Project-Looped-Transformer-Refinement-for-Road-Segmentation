"""Summarize the fixed draft-head v3 matrix and apply its preregistered rules."""
import math
from statistics import mean, median


ARMS = ("A1-i", "v3a", "v3b")
SEEDS = (0, 1)
DOMAINS = ("Overall", "Urban", "Rural")
SCORE_KEYS = ("road_iou", "cldice", "boundary_iou", "quality")
KS = ("1", "2", "4", "8")


def metric_vector(scores):
    return {key: scores[key] for key in SCORE_KEYS}


def draft_scores(result, mode, domain):
    return result["groups"][mode][domain]["y0_draft"]["scores"]


def decide(training, averages, tta_rows, refiner_rows):
    baseline = averages["A1-i"]["Overall"]["quality"]
    gains = {arm: averages[arm]["Overall"]["quality"] - baseline for arm in ARMS[1:]}
    gaps = {f"{arm}_s{seed}": abs(row["val_sel_curve"][-1] - row["best_selection_A"])
            for (arm, seed), row in training.items() if arm != "A1-i"}
    p3 = all(gaps[f"v3a_s{seed}"] <= .002 for seed in SEEDS)
    tta_gains = {f"{row['arm']}_s{row['seed']}": row["delta"]["quality"] for row in tta_rows
                 if row["split"] == "Val-rep" and row["domain"] == "Overall"}
    y0_gain = tta_gains["A1-i_s0"]
    refiner_gains = {row["K"]: row["delta_vs_recorded"]["quality"] for row in refiner_rows
                     if row["domain"] == "Overall" and row["K"] != "8"}
    eligible = {arm: averages[arm]["Overall"]["quality"] >= baseline + .002
                and all(training[arm, seed]["val_rep"]["Overall"]["quality"] > baseline for seed in SEEDS)
                for arm in ARMS[1:]}
    eligible_arms = [arm for arm in ARMS[1:] if eligible[arm]]
    chosen = max(eligible_arms, key=lambda arm: (
        averages[arm]["Overall"]["quality"],
        mean(training[arm, seed]["best_selection_A"] for seed in SEEDS),
    ), default="A1-i")
    best_epochs = [training[chosen, seed]["best_epoch"] for seed in SEEDS]
    median_epoch = median(best_epochs)
    return {
        "P1": {"met": 0 <= gains["v3a"] <= .004, "v3a_mean_delta": gains["v3a"]},
        "P2": {"met": .002 <= gains["v3b"] <= .006 and gains["v3b"] > gains["v3a"],
               "v3b_mean_delta": gains["v3b"], "v3a_mean_delta": gains["v3a"]},
        "P3": {"met": p3, "last_to_best_val_sel_absolute_gaps": gaps},
        "P4": {"met": all(.003 <= gain <= .008 for gain in tta_gains.values()),
               "val_rep_Overall_y0_deltas": tta_gains},
        "P5": {"met": all(0 < gain < y0_gain for gain in refiner_gains.values()),
               "val_rep_Overall_deltas": refiner_gains, "A1-i_s0_y0_delta": y0_gain},
        "R1": {"met_by_arm": eligible, "baseline_mean_A": baseline, "minimum_arm_mean_A": baseline + .002,
               "seed_A": {arm: [training[arm, seed]["val_rep"]["Overall"]["quality"] for seed in SEEDS]
                          for arm in ARMS}},
        "R2": {"selected_arm": chosen},
        "R3": {"train_to_last": p3, "selected_arm": chosen, "best_epochs": best_epochs,
               "median_best_epoch": median_epoch,
               "fixed_epochs": None if p3 else math.floor(median_epoch + .5)},
        "R4": {"include_tta": all(gain >= .002 for gain in refiner_gains.values()),
               "val_rep_Overall_deltas": refiner_gains},
    }


def build_report(training, e6_reference, split_hash):
    averages = {arm: {domain: {key: mean(training[arm, seed]["val_rep"][domain][key] for seed in SEEDS)
                               for key in SCORE_KEYS} for domain in DOMAINS} for arm in ARMS}
    tta_rows = []
    for row in training.values():
        for split, result in row["tta"].items():
            for domain in DOMAINS:
                plain = draft_scores(result, "none", domain)
                tta = draft_scores(result, "d4", domain)
                images = [image for image in result["images"] if domain in ("Overall", image["domain"])]
                tta_rows.append(dict(arm=row["arm"], seed=row["seed"], split=split, domain=domain,
                                     none=metric_vector(plain), d4=metric_vector(tta),
                                     delta={key: tta[key] - plain[key] for key in SCORE_KEYS},
                                     mean_draft_seconds={mode: mean(image["draft_seconds"][mode] for image in images)
                                                         for mode in ("none", "d4")}))

    e6 = training["A1-i", 0]["tta"]["Val-rep"]["groups"]
    refiner_rows = []
    for domain in DOMAINS:
        for k in KS:
            recorded = metric_vector(e6_reference[domain][k]["scores"])
            plain = metric_vector(e6["none"][domain][k]["scores"])
            tta = metric_vector(e6["d4"][domain][k]["scores"])
            refiner_rows.append(dict(domain=domain, K=k, recorded_none=recorded, paired_none=plain, d4=tta,
                                     delta_vs_recorded={key: tta[key] - recorded[key] for key in SCORE_KEYS},
                                     delta_vs_paired={key: tta[key] - plain[key] for key in SCORE_KEYS}))

    return dict(val_split_sha256=split_hash, threshold=.5, training=list(training.values()),
                arm_means=averages, tta_comparisons=tta_rows, e6_comparisons=refiner_rows,
                decisions=decide(training, averages, tta_rows, refiner_rows))


def markdown(report):
    def vector(scores):
        return " / ".join(f"{scores[key]:.6f}" for key in SCORE_KEYS)

    lines = ["# Observed draft head v3 and D4 TTA results", "",
             "Observed scores use threshold 0.5 and official Counts pooled before summarization.",
             "Score cells list IoU / clDice / BIoU / A. Arm means average two seed scores.", "",
             "## Val-rep draft heads", "",
             "| Arm | Seed | Best epoch | Overall | Urban | Rural |",
             "|---|---|---:|---|---|---|"]
    for arm in ARMS:
        for row in report["training"]:
            if row["arm"] == arm:
                groups = " | ".join(vector(row["val_rep"][domain]) for domain in DOMAINS)
                lines.append(f"| {arm} | {row['seed']} | {row['best_epoch']} | {groups} |")
        groups = " | ".join(vector(report["arm_means"][arm][domain]) for domain in DOMAINS)
        lines.append(f"| {arm} | Mean |  | {groups} |")

    lines += ["", "## Observed Val-sel A by epoch", "",
              "| Arm | Seed | " + " | ".join(map(str, range(1, 11))) + " |",
              "|---|---|" + "---:|" * 10]
    for row in report["training"]:
        curve = " | ".join(f"{value:.6f}" for value in row["val_sel_curve"])
        lines.append(f"| {row['arm']} | {row['seed']} | {curve} |")

    lines += ["", "## Observed D4 draft TTA", "",
              "Times are mean inference seconds per image. D4 time includes the original view.",
              "The JSON contains timings for every image.", "",
              "| Arm | Seed | Split | Domain | Without TTA | D4 TTA | ΔA | Seconds without / with TTA |",
              "|---|---|---|---|---|---|---:|---|"]
    for row in report["tta_comparisons"]:
        times = row["mean_draft_seconds"]
        lines.append(f"| {row['arm']} | {row['seed']} | {row['split']} | {row['domain']} | "
                     f"{vector(row['none'])} | {vector(row['d4'])} | {row['delta']['quality']:+.6f} | "
                     f"{times['none']:.6f} / {times['d4']:.6f} |")

    lines += ["", "## Observed K-step e6 seed-0 validation", "",
              "The archived and paired runs both use the A1-i seed-0 draft with e6 seed 0.",
              "The D4 run gives e6 the original-view features and the eight-view mean draft logit.", "",
              "| Domain | K | Archived without TTA | Paired without TTA | D4 TTA | ΔA vs archived | ΔA vs paired |",
              "|---|---:|---|---|---|---:|---:|"]
    for row in report["e6_comparisons"]:
        lines.append(f"| {row['domain']} | {row['K']} | {vector(row['recorded_none'])} | "
                     f"{vector(row['paired_none'])} | {vector(row['d4'])} | "
                     f"{row['delta_vs_recorded']['quality']:+.6f} | {row['delta_vs_paired']['quality']:+.6f} |")

    lines += ["", "## Observed preregistered outcomes", ""]
    for rule, result in report["decisions"].items():
        fields = []
        for name, value in result.items():
            if isinstance(value, dict):
                value = ", ".join(f"{key}={item:.9g}" if isinstance(item, float)
                                  else f"{key}={item}" for key, item in value.items())
            elif isinstance(value, float):
                value = f"{value:.9g}"
            fields.append(f"{name.replace('_', ' ')} = {value}")
        lines.append(f"- Observed {rule}. " + ". ".join(fields) + ".")
    if "acceptance" in report:
        accepted = report["acceptance"]
        opening = ["# Observed draft head v3 acceptance", "",
                   "| Gate | Status | Maximum absolute difference | Comparisons |",
                   "|---|---|---:|---:|"]
        for name, gate in accepted["gates"].items():
            opening.append(f"| {name} | {gate['status']} | {gate['maximum_absolute_difference']:.12g} | {gate['comparisons']} |")
        opening += ["", "| Rule | Observed outcome | Evidence |", "|---|---|---|"]
        for name, result in accepted["rules"].items():
            opening.append(f"| {name} | {result['status']} | {result['evidence']} |")
        opening += ["", "Observed R2 retains A1-i. Observed R4 admits draft-only D4 TTA.", "",
                    "Observed R2-selected arm Val-sel last-minus-best differences are " +
                    ", ".join(f"{name}={value:.12g}" for name, value in
                              accepted["selected_arm_last_minus_best"].items()) + ".", "",
                    "Observed TTA job 8756257 used NVIDIA L40S for 2304 wallclock seconds. "
                    "Peak CUDA memory was not saved by this evaluator. Per-image timings remain in metrics.json.", ""]
        lines = opening + lines[2:]
    return "\n".join(lines) + "\n"
