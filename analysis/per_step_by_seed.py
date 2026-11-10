#!/usr/bin/env python3
"""Render per-seed tables from saved per-step statistics without inference."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("Overall", "Urban", "Rural")
REGIONS = ("R1", "R2", "R3", "R4")


def percentage(numerator, denominator):
    """Use NA for an undefined fraction instead of silently assigning zero."""
    return f"{100 * numerator / denominator:.6f}" if denominator else "NA"


def number(value):
    return f"{value:.12f}" if value is not None else "NA"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def render(report, source):
    """Read stored counts and scores; do not recompute official metrics."""
    lines = ["# Observed per-step results by seed", "", f"Source: `{source}`.", "",
             "Observed additions and deletions count consecutive-step flips, not cumulative flips from y0. "
             "Addition precision is added_correct / added. Each R1-R4 share uses that step's total additions "
             "as its denominator; the four shares sum to 100% before rounding. All flips are restricted "
             "to valid pixels. NA denotes an undefined fraction.", "",
             "Observed IoU, clDice, BIoU and A are the saved official aggregate scores, with A = quality. "
             "Gap repair and new break use the original fixed-y0 skeleton sets. Fraction columns marked % "
             "are percentages; metric columns use the 0-1 scale.", "",
             "JSON locations: seeds.<seed>.<domain>.steps.<t>, with added, deleted, added_correct, "
             "regions.<R1-R4>.added, scores.<road_iou|cldice|boundary_iou|quality>, gap_repair and new_break.", ""]
    best_rows = []
    for seed in sorted(report["seeds"], key=int):
        groups = report["seeds"][seed]
        lines += [f"## Observed seed {seed}", ""]
        for domain in DOMAINS:
            steps = groups[domain]["steps"]
            ordered = sorted(steps, key=int)
            rows = []
            for t in ordered:
                step = steps[t]
                assert step["added"] == step["added_correct"] + step["added_wrong"]
                assert sum(step["regions"][r]["added"] for r in REGIONS) == step["added"]
                scores = step["scores"]
                rows.append([t, step["added"], step["deleted"],
                             percentage(step["added_correct"], step["added"]),
                             *[percentage(step["regions"][r]["added"], step["added"]) for r in REGIONS],
                             *[number(scores[f]) for f in ("road_iou", "cldice", "boundary_iou", "quality")],
                             number(100 * step["gap_repair"]) if step["gap_repair"] is not None else "NA",
                             number(100 * step["new_break"]) if step["new_break"] is not None else "NA"])
            lines += [f"### Observed {domain}", "", table(
                ["t", "Added", "Deleted", "Addition precision %", *[f"{r} addition share %" for r in REGIONS],
                 "IoU", "clDice", "BIoU", "A", "Gap repair %", "New break %"], rows), ""]
            if domain in ("Urban", "Rural"):
                best = max(ordered, key=lambda t: steps[t]["scores"]["quality"])
                first, last = (steps[t]["scores"]["quality"] for t in ("1", "8"))
                best_rows.append([seed, domain, best, number(steps[best]["scores"]["quality"]),
                                  number(first), number(last), f"{last - first:+.12f}"])
    lines += ["## Observed Urban and Rural maxima over t=1-8", "",
              "Observed maxima exclude y0; ties select the earliest step. The final column subtracts "
              "the saved t=1 A from the saved t=8 A.", "",
              table(["Seed", "Domain", "Highest-A step", "Highest A", "A at t=1", "A at t=8", "A(t=8)-A(t=1)"], best_rows), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/per_step/v2/metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/per_step/v2/by_seed.md")
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    source = args.metrics.resolve()
    label = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.name
    args.output.write_text(render(report, label), encoding="utf-8")
    print(f"Wrote {args.output.name}; no model or original report was modified.")


if __name__ == "__main__":
    main()
