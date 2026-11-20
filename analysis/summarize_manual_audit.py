"""Describe one reviewer's 30 selected e0 v2 seed-0 Urban degradation cases."""
import argparse
import csv
from pathlib import Path
from statistics import median, correlation


def joined_rows(sheet, key):
    with Path(sheet).open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    with Path(key).open(newline="") as stream:
        keys = list(csv.DictReader(stream))
    numbers = [int(row["audit_no"]) for row in reviews]
    key_numbers = [int(row["audit_no"]) for row in keys]
    if len(numbers) != 30 or set(numbers) != set(range(1, 31)) or len(set(key_numbers)) != len(key_numbers):
        raise ValueError("Expected 30 unique review numbers 1 through 30 and unique key numbers")
    lookup = {int(row["audit_no"]): row for row in keys}
    if set(numbers) - lookup.keys():
        raise ValueError("Missing audit_no in key")
    return [{**lookup[int(row["audit_no"])], **row} for row in reviews]


def markdown(rows):
    lines = ["# Observed manual audit", "",
             "Observed one reviewer's judgments on 30 selected Urban degradation cases from e0 v2 seed 0 "
             "with the linear draft. The current e6 submission uses a different draft and refiner.",
             "The sample was selected for degradation and does not estimate dataset-wide category prevalence.", "",
             "A denotes unlabeled real roads. B denotes incomplete or truncated labels. C denotes incorrect model additions. "
             "D denotes almost no red. Mixed labels retain the reviewer's original ordering.",
             "Added precision is computed against the benchmark annotation. real_road_pct is the reviewer's percentage "
             "of red pixels judged to be real roads.", "",
             "| red_main | Cases | Median delta A4-y0 | Median added precision | Median real_road_pct | Rated cases |",
             "|---|---:|---:|---:|---:|---:|"]
    for category in sorted(set(row["red_main"] for row in rows)):
        group = [row for row in rows if row["red_main"] == category]
        rated = [float(row["real_road_pct"]) for row in group if row["real_road_pct"]]
        rating = f"{median(rated):.6f}" if rated else "unavailable"
        lines.append(f"| {category} | {len(group)} | {median(float(r['delta_A4_y0']) for r in group):.9f} | "
                     f"{median(float(r['added_precision']) for r in group):.9f} | {rating} | {len(rated)} |")
    rated = [row for row in rows if row["real_road_pct"]]
    human = [float(row["real_road_pct"]) / 100 for row in rated]
    annotated = [float(row["added_precision"]) for row in rated]
    lines += ["", f"Observed {len(rows)} joined rows with every audit_no from 1 through 30. "
              f"Observed {len(rated)} paired percentage/precision ratings; missing percentages are excluded.",
              f"Observed median human real-road fraction {median(human):.9f}, "
              f"median annotated added precision {median(annotated):.9f}, "
              f"and median paired human-minus-annotation difference {median(a-b for a,b in zip(human, annotated)):.9f}.",
              f"Observed Pearson correlation between human real-road fraction and annotated added precision "
              f"{correlation(human, annotated):.9f}.", "",
              "Observed the following descriptive groups by the supplied human percentage.", "",
              "| real_road_pct | Cases | Median added precision |",
              "|---|---:|---:|"]
    for pct in sorted(set(float(row["real_road_pct"]) for row in rated)):
        group = [float(row["added_precision"]) for row in rated if float(row["real_road_pct"]) == pct]
        lines.append(f"| {pct:g} | {len(group)} | {median(group):.9f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=Path("results/manual_audit/audit_sheet_dingning.csv"))
    parser.add_argument("--key", type=Path, default=Path("runs/manual_audit/key.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/manual_audit/summary.md"))
    args = parser.parse_args()
    args.out.write_text(markdown(joined_rows(args.sheet, args.key)))


if __name__ == "__main__":
    main()
