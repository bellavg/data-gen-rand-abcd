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
import subprocess
import sys
import unittest
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2])


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

    def test_baseline_choices_are_synthnet_and_hoga(self):
        result = _run_cli("--help")
        self.assertIn("{synthnet,hoga}", result.stdout)


if __name__ == "__main__":
    unittest.main()
