"""CLI wiring smoke tests for train_baseline.py.

Regression coverage for a real bug: train_baseline.py was written against a
transient, uncommitted config.py/AIGDataModule state (a configurable
`split_by` feature) that was never actually part of this branch's history --
it only exists in a separate commit on `main`. The argparse setup referenced
`config.SPLIT_BY`/`config.VALID_SPLIT_BY` (which don't exist here) and passed
`split_by=` to `AIGDataModule` (which doesn't accept it here), so simply
running `python -m train_baseline --help` crashed immediately. None of the
other baseline tests exercise this file's `__main__` block at all (they test
the regressor/lightning-wrapper classes directly), so this went undetected
until it was run for real. These tests invoke the actual CLI as a subprocess
to catch this class of bug going forward.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2])
_SHELL_DIR = Path(_SRC_DIR) / "shell"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR
    return subprocess.run(
        [sys.executable, "-m", "train_baseline", *args],
        cwd=_SRC_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestTrainBaselineCLI(unittest.TestCase):
    def test_help_does_not_crash(self):
        result = _run_cli("--help")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_split_by_flag_does_not_exist(self):
        # This branch's dataset hardcodes design-level splitting (see
        # data/dataset.py); --split_by isn't a real, wired-up flag here.
        result = _run_cli("--help")
        self.assertNotIn("--split_by", result.stdout)

    def test_missing_required_baseline_arg_fails_cleanly(self):
        result = _run_cli("--csv_paths", "dummy.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--baseline", result.stderr)

    def test_baseline_choices_are_synthnet_hoga_and_deepgate4(self):
        result = _run_cli("--help")
        self.assertIn("{synthnet,hoga,deepgate4}", result.stdout)

    def test_every_flag_used_by_a_job_script_exists(self):
        """Generalises the per-flag checks below.

        These SLURM scripts are the only way these baselines are ever launched,
        and a renamed or removed flag surfaces as an argparse error minutes into
        a queued GPU job rather than here. Parses each script for the
        `--flag` tokens it passes and requires all of them to appear in --help.
        """
        known = set(re.findall(r"--[A-Za-z0-9_]+", _run_cli("--help").stdout))
        self.assertIn("--baseline", known, "sanity: --help did not parse")

        scripts = sorted(_SHELL_DIR.glob("train_baseline_*.sh"))
        self.assertTrue(scripts, "no baseline job scripts found")
        for script in scripts:
            with self.subTest(script=script.name):
                used = set(
                    re.findall(r"^\s+(--[a-z0-9_]+)\s", script.read_text(), re.M)
                )
                self.assertTrue(used, f"{script.name} passes no flags -- parser changed?")
                self.assertEqual(
                    sorted(used - known), [], f"{script.name} passes unknown flags"
                )

    def test_max_epochs_is_resolved_per_baseline(self):
        """Each baseline paper publishes its own epoch count: SynthNet 80
        (models/qor/SynthNetV3/train.py), DeepGate4 200 (Zheng et al. ICLR'25
        Sec 4.1, "We train all models for 200 epochs", and upstream's
        run/train_large.sh --epoch 200). HOGA publishes none and keeps 80.
        A single shared default would silently give DeepGate4 SynthNet's number.
        """
        sys.path.insert(0, _SRC_DIR)
        try:
            import train_baseline
        finally:
            sys.path.remove(_SRC_DIR)

        self.assertEqual(train_baseline.DEEPGATE4_DEFAULTS["max_epochs"], 200)
        self.assertEqual(train_baseline.SYNTHNET_DEFAULTS["max_epochs"], 80)
        self.assertEqual(
            set(train_baseline._BASELINE_DEFAULTS), {"synthnet", "hoga", "deepgate4"}
        )
        for name, defaults in train_baseline._BASELINE_DEFAULTS.items():
            with self.subTest(baseline=name):
                self.assertEqual(
                    set(defaults), {"batch_size", "lr", "weight_decay", "max_epochs"}
                )

    def test_deepgate4_flags_are_wired(self):
        # train_baseline_deepgate4.sh passes all of these; a rename would break
        # the job script silently. --deepgate4_num_hops is the virtual-edge
        # radius k and the baseline's dominant cost knob, and
        # --deepgate4_gradient_checkpointing must stay reachable because the
        # model does not fit in GPU memory without it (see
        # src/baselines/deepgate4/regressor.py).
        result = _run_cli("--help")
        for flag in (
            "--deepgate4_num_hops",
            "--deepgate4_max_nodes_per_batch",
            "--deepgate4_symmetric_virtual_edges",
            "--deepgate4_gradient_checkpointing",
            "--deepgate4_pretrained_tokenizer",
        ):
            self.assertIn(flag, result.stdout)

    def test_hoga_node_budget_flag_is_wired(self):
        # HOGA batches by total node count, not graph count -- graphs here
        # average ~40k nodes and a fixed batch_size=32 OOMs (see
        # train_baseline.py's module docstring). train_baseline_hoga.sh passes
        # this flag, so a rename would break the job script silently.
        result = _run_cli("--help")
        self.assertIn("--hoga_max_nodes_per_batch", result.stdout)

    def test_accumulate_grad_batches_flag_is_wired(self):
        # train_baseline_hoga.sh passes this (20, to match the primary model's
        # ~75-graph effective batch); a rename would break the job script
        # silently. The default stays 1 so SynthNet keeps its published
        # batch_size=64 effective batch -- not asserted here because this
        # parser uses argparse's plain HelpFormatter, which never prints
        # defaults.
        result = _run_cli("--help")
        self.assertIn("--accumulate_grad_batches", result.stdout)

    def test_epoch_subsampling_flags_are_wired(self):
        # train_baseline_hoga.sh passes both; a rename would silently restore
        # the ~12h epochs these exist to cap.
        result = _run_cli("--help")
        self.assertIn("--limit_train_batches", result.stdout)
        self.assertIn("--limit_val_batches", result.stdout)

    def test_batch_limit_preserves_int_vs_float(self):
        # Lightning reads an int as an absolute batch count and a float as a
        # fraction of the epoch. It coerces any float > 1 with no fractional
        # part back to int, so the distinction bites at exactly one value:
        # `1` = one batch, `1.0` = the whole epoch. Under a plain type=float,
        # "--limit_val_batches 1" would silently run all 32k val batches.
        from train_baseline import _batch_limit

        self.assertIsInstance(_batch_limit("1"), int)
        self.assertIsInstance(_batch_limit("1.0"), float)
        self.assertEqual(_batch_limit("15000"), 15000)
        self.assertIsInstance(_batch_limit("15000"), int)
        self.assertEqual(_batch_limit("0.1"), 0.1)
        self.assertIsInstance(_batch_limit("0.1"), float)

    def test_hoga_hop_cache_dir_is_optional(self):
        # The on-disk hop cache is not size-viable at full scale, so omitting
        # --hoga_hop_cache_dir must be allowed (it used to raise).
        result = _run_cli("--help")
        self.assertIn("--hoga_hop_cache_dir", result.stdout)
        self.assertNotIn("required when --baseline hoga", result.stdout)


if __name__ == "__main__":
    unittest.main()
