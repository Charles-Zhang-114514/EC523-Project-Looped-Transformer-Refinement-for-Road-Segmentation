"""A2 reporting and shell arguments without training or scheduler submission."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from analysis import aggregate_experiments as aggregate
from scripts.train_refiner import build_parser


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs/a2_refiner_conv_draft.qsub"
APPENDIX = "\n\n## Untied vs shared\n\nObserved old comparison.\n\n## No-add penalty (e5) vs shared (e0)\n\nObserved unchanged appendix.\n"


def fixture_report(seed, conv=False):
    groups = {}
    for domain, base in (("Overall", .5), ("Urban", .6), ("Rural", .4)):
        draft = base + .001 * seed + (.04 if conv else 0)
        groups[domain] = {"y0_draft": {"scores": {"A": draft}}}
        for k in (1, 2, 4, 6, 8):
            groups[domain][str(k)] = {"scores": {"A": draft + k * (.002 if conv else .001)},
                "extra": {"tracking": {"gap_repair_fraction": .1, "new_break_fraction": .01},
                          "changes": {"added_correct_fraction": .4, "deleted_correct_fraction": .5},
                          "new_fragments": {"count": 3}}}
    return {"protocol": "val_sel_rep", "config": {"seed": seed, "val_split_sha256": "fixture"},
            "best_epoch": seed + 1, "evaluation": {"groups": groups}}


class A2ConvDraftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_run(self, directory, experiment, seed):
        path = directory / f"{experiment}_s{seed}" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fixture_report(seed, experiment == "e6_convdraft")))

    def test_no_e6_outputs_match_executable_prechange_snapshot_byte_for_byte(self):
        directory = self.directory / "inputs"
        for experiment in aggregate.EXPERIMENTS:
            for seed in (0, 1, 2):
                self.write_run(directory, experiment, seed)
        (directory / "summary.md").write_text("# Previous report" + APPENDIX)
        original = subprocess.check_output(
            ["git", "show", "56a5cf9:analysis/aggregate_experiments.py"], cwd=ROOT, text=True)
        namespace = {"__name__": "legacy_aggregate", "__file__": str(ROOT / "analysis/aggregate_experiments.py")}
        exec(compile(original, namespace["__file__"], "exec"), namespace)
        for name, main in (("legacy", namespace["main"]), ("current", aggregate.main)):
            with patch.object(sys, "argv", ["aggregate", "--results-dir", str(directory),
                                            "--out-dir", str(self.directory / name)]), contextlib.redirect_stdout(io.StringIO()):
                main()
        for filename in ("summary.md", "summary.json"):
            self.assertEqual((self.directory / "legacy" / filename).read_bytes(),
                             (self.directory / "current" / filename).read_bytes())

    def test_e6_uses_own_draft_and_preserves_all_existing_rows(self):
        directory = self.directory / "inputs"
        for seed in (0, 1, 2):
            self.write_run(directory, "e0_v2", seed)
        before = aggregate.aggregate_directory(directory)
        for seed in (0, 1, 2):
            self.write_run(directory, "e6_convdraft", seed)
        after = aggregate.aggregate_directory(directory)
        for key in ("rows", "table1"):
            unchanged = [r for r in after[key] if not r["experiment"].startswith("e6_convdraft")]
            self.assertEqual(before[key], unchanged)
        rows = after["conv_draft_comparison"]["rows"]
        for domain in ("Overall", "Urban", "Rural"):
            for experiment, expected_gain in (("e0_v2", .004), ("e6_convdraft", .008)):
                row = next(r for r in rows if r["experiment"] == experiment and r["domain"] == domain and r["seed"] == 0)
                self.assertAlmostEqual(row["gain_K4"], expected_gain)
        mean = next(r for r in rows if r["experiment"] == "e6_convdraft mean ± std (3 seeds)" and r["domain"] == "Overall")
        self.assertAlmostEqual(mean["y0_A"]["mean"], .541)
        self.assertAlmostEqual(mean["A_K4"]["mean"], .549)
        self.assertAlmostEqual(mean["A_K4"]["std"], .001)
        self.assertAlmostEqual(mean["gain_K4"]["mean"], .008)
        (directory / "summary.md").write_text("# Previous report" + APPENDIX)
        for _ in range(2):
            with patch.object(sys, "argv", ["aggregate", "--results-dir", str(directory)]), contextlib.redirect_stdout(io.StringIO()):
                aggregate.main()
            text = (directory / "summary.md").read_text()
            self.assertEqual(text.count(aggregate.CONV_DRAFT_MARKER), 1)
            self.assertIn(APPENDIX, text)

    def test_partial_e6_is_incomplete_without_partial_mean(self):
        self.write_run(self.directory, "e6_convdraft", 1)
        report = aggregate.aggregate_directory(self.directory)
        mean = next(r for r in report["rows"] if r["experiment"] == "e6_convdraft mean ± std (3 seeds)")
        self.assertEqual(mean["status"], "incomplete")
        self.assertEqual(mean["completed_seeds"], 1)
        for row in report["conv_draft_comparison"]["rows"]:
            if row["experiment"] == "e6_convdraft mean ± std (3 seeds)":
                self.assertEqual(row["status"], "incomplete")
                self.assertNotIn("A_K4", row)

    def shell_run(self, arguments, score="0.538246"):
        capture = self.directory / "arguments.json"
        capture.unlink(missing_ok=True)
        fake_train = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["A2_CAPTURE"]).write_text(json.dumps(args))
out = Path(args[args.index("--out-dir")+1]); out.mkdir(parents=True, exist_ok=True)
report = {"evaluation": {"groups": {"Overall": {"y0_draft": {"scores": {"A": float(os.environ["A2_SCORE"])}}}}}}
(out/"metrics.json").write_text(json.dumps(report))
'''
        # Metadata is mocked; the final JSON-only regression runs for real.
        (self.directory / "env.sh").write_text(
            'python() {\nif [[ "$1" == -u ]]; then\n"$A2_TEST_PYTHON" -c ' + shlex.quote(fake_train) + ' "$@"\n'
            'elif [[ $# == 1 ]]; then cat >/dev/null; else "$A2_TEST_PYTHON" "$@"; fi\n}\n'
            'git() { :; }\n')
        env = dict(os.environ, SGE_O_WORKDIR=str(self.directory), A2_CAPTURE=str(capture),
                   A2_SCORE=score, A2_TEST_PYTHON=sys.executable)
        process = subprocess.run(["bash", str(JOB), *arguments], env=env, capture_output=True, text=True)
        return process, json.loads(capture.read_text()) if capture.exists() else None

    def test_qsub_valid_seeds_match_e0_except_checkpoint_and_output_paths(self):
        source = (ROOT / "scripts/submit_controlled_exp.sh").read_text().splitlines()
        for seed in (0, 1, 2):
            process, command = self.shell_run([str(seed)])
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("PASS e6 y0_draft", process.stdout)
            line = next(line for line in source if line.startswith(f"submit -N ec523-e0-s{seed} "))
            original = shlex.split(line)
            baseline = vars(build_parser().parse_args(original[original.index("jobs/exp_refiner.qsub") + 1:]))
            proposed = vars(build_parser().parse_args(command[2:]))
            changed = {key for key in baseline if baseline[key] != proposed[key]}
            self.assertEqual(changed, {"probe_ckpt", "out_dir", "ckpt_dir"})
            self.assertEqual(proposed["seed"], seed)
            self.assertEqual(proposed["probe_ckpt"], Path("runs/draft_head/A1-i/best.pt"))
            self.assertEqual(proposed["out_dir"], Path(f"results/exp/e6_convdraft_s{seed}"))
            self.assertEqual(proposed["ckpt_dir"], Path(f"runs/exp/e6_convdraft_s{seed}"))

    def test_qsub_rejects_anything_other_than_one_exact_seed(self):
        for arguments in ([], ["3"], ["-1"], ["01"], ["0.0"], ["seed"], ["0", "extra"]):
            with self.subTest(arguments=arguments):
                process, command = self.shell_run(arguments)
                self.assertEqual(process.returncode, 2)
                self.assertIsNone(command)

    def test_qsub_draft_regression_failure_is_nonfatal(self):
        process, _ = self.shell_run(["0"], score="0.0")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("FAIL e6 y0_draft", process.stdout)


if __name__ == "__main__":
    unittest.main()
