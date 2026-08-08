"""CLI wiring smoke tests for train_baseline.py.

Regression coverage for a real bug: train_baseline.py was written against a
configurable `split_by` feature that had not reached this branch yet, so the
argparse setup referenced `config.SPLIT_BY`/`config.VALID_SPLIT_BY` and passed
`split_by=` to an `AIGDataModule` that accepted none of them, and simply
running `python -m train_baseline --help` crashed immediately. None of the
other baseline tests exercise this file's `__main__` block at all (they test
the regressor/lightning-wrapper classes directly), so this went undetected
until it was run for real. These tests invoke the actual CLI as a subprocess
to catch this class of bug going forward.

`main` has since been merged in, so `split_by` and `use_graph_cache` are both
real flags now. They stay under test because they are hashed into the dataset's
graph-cache signature: a value that disagrees with warmup_train_cache.sh's
renames the manifest, misses the warmed one, and silently re-derives all ~700k
train samples on the GPU node (~10 h) instead of loading them.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        # Each call spawns a fresh interpreter that imports torch/PyG: ~3 s
        # warm, but this file makes several such calls and the whole suite runs
        # them alongside everything else. At 30 s a loaded machine produced
        # TimeoutExpired failures that vanish on re-run -- a false red, not a
        # signal. This is a liveness guard against a hung CLI, so it only needs
        # to be well short of the suite's own patience.
        timeout=120,
    )


class TestTrainBaselineCLI(unittest.TestCase):
    def test_help_does_not_crash(self):
        result = _run_cli("--help")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_job_scripts_pin_the_warmup_cache_signature(self):
        """Both flags are hashed into _build_cache_signature().

        warmup_train_cache.sh builds the train+val manifest on a CPU node with
        use_graph_cache=False and its array slot's split_by. If a baseline job
        disagrees on either, it computes a different signature, finds no
        manifest under that name, and rebuilds the whole thing on the GPU node
        -- which is exactly how a SynthNet run burned ~10 h re-caching 707k
        graphs it already had.
        """
        # Anchored to the kwarg in the embedded Python, NOT a bare substring:
        # "use_graph_cache=False" also appears twice in that file's comments,
        # so an unanchored check stays green even if the real call flips to
        # True -- the one change this assertion exists to catch.
        warmup = (_SHELL_DIR / "warmup_train_cache.sh").read_text()
        self.assertRegex(
            warmup,
            r"(?m)^\s*use_graph_cache=False,",
            "warmup no longer disables the graph cache -- update the job scripts",
        )

        scripts = sorted(_SHELL_DIR.glob("train_baseline_*.sh"))
        self.assertTrue(scripts, "no baseline job scripts found")
        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text()
                # (?m)^\s*(?!#) so a commented-out flag in the srun block does
                # not satisfy these -- the point is what the job actually
                # passes, not what the file happens to mention.
                self.assertRegex(
                    text,
                    r'(?m)^\s*(?!#)--use_graph_cache\s+"false"',
                    f"{script.name} must pass --use_graph_cache false",
                )
                self.assertRegex(
                    text,
                    r'(?m)^\s*(?!#)--split_by\s+"\$SPLIT_BY"',
                    f"{script.name} must pass --split_by",
                )
                self.assertRegex(
                    text,
                    r'(?m)^\s*(?!#)SPLIT_BY="\$\{SPLIT_BY:-design\}"',
                    f"{script.name} must default SPLIT_BY to the warmed strategy",
                )

    def test_run_label_separates_the_two_synthnet_edge_directions(self):
        """DIAGNOSIS.md asks for both directions to be run and reported.

        The two runs share every other setting, so without a suffix they share
        one checkpoint dir, one log dir and one WandB name, and the second
        silently overwrites the first's last.ckpt. Upstream (True) stays
        untagged so existing run directories keep their names.
        """
        import config
        import train_baseline

        def label(**overrides):
            args = SimpleNamespace(
                baseline="synthnet",
                algorithm="Orchestrate",
                split_by=config.SPLIT_BY,
                synthnet_upstream_edge_direction=True,
                loss=train_baseline._BASELINE_DEFAULTS["synthnet"]["loss"],
            )
            for key, value in overrides.items():
                setattr(args, key, value)
            return train_baseline._run_label(args)

        self.assertEqual(label(), "synthnet_Orchestrate")
        self.assertEqual(
            label(synthnet_upstream_edge_direction=False),
            "synthnet_Orchestrate_nativeedges",
        )
        # Orthogonal to split_by, and both suffixes stack.
        self.assertEqual(
            label(split_by="recipe", synthnet_upstream_edge_direction=False),
            "synthnet_Orchestrate_recipe_nativeedges",
        )
        # The suffix is SynthNet-only: the other baselines have no such flag,
        # so their label must not depend on it.
        self.assertEqual(
            label(baseline="hoga", synthnet_upstream_edge_direction=False),
            "hoga_Orchestrate",
        )

    def test_run_label_separates_the_two_loss_recipes(self):
        """PolarGate is meant to be run under both upstream's own 'mae' and
        the primary model's SmoothL1 (train.py:151), and the two runs differ in
        nothing else -- so without a suffix the second overwrites the first's
        last.ckpt. Each baseline's own default stays untagged.
        """
        import config
        import train_baseline

        def label(baseline, loss):
            return train_baseline._run_label(
                SimpleNamespace(
                    baseline=baseline,
                    algorithm="Orchestrate",
                    split_by=config.SPLIT_BY,
                    synthnet_upstream_edge_direction=True,
                    loss=loss,
                    polargate_size_covariates=False,
                    polargate_pooling="mean",
                )
            )

        self.assertEqual(label("polargate", "mae"), "polargate_Orchestrate")
        self.assertEqual(
            label("polargate", "smooth_l1"), "polargate_Orchestrate_smooth_l1"
        )
        # Inverted for the other three, whose default is mse.
        self.assertEqual(label("hoga", "mse"), "hoga_Orchestrate")
        self.assertEqual(label("hoga", "smooth_l1"), "hoga_Orchestrate_smooth_l1")

    def test_run_label_separates_the_two_gamora_lr_recipes(self):
        """train_baseline_gamora.sh's GAMORA_LR deviation (0.0003, config.LR)
        is meant to run alongside the faithful 0.008 run, not replace it -- so
        without a suffix the two share algo_checkpoint_dir/log_dir and
        ModelCheckpoint(save_last=True) silently overwrites the first run's
        last.ckpt. Upstream's own 0.008 stays untagged so the existing
        wandb xfauwjcw checkpoint dir keeps its name.
        """
        import config
        import train_baseline

        def label(lr):
            return train_baseline._run_label(
                SimpleNamespace(
                    baseline="gamora",
                    algorithm="Orchestrate",
                    split_by=config.SPLIT_BY,
                    synthnet_upstream_edge_direction=True,
                    loss=train_baseline._BASELINE_DEFAULTS["gamora"]["loss"],
                    lr=lr,
                )
            )

        self.assertEqual(label(0.008), "gamora_Orchestrate")
        self.assertEqual(label(0.0003), "gamora_Orchestrate_lr0.0003")
        # A namespace with no lr attribute at all (older callers) must not
        # crash -- treated as "unset", same as the baseline's own default.
        self.assertEqual(
            train_baseline._run_label(
                SimpleNamespace(
                    baseline="gamora",
                    algorithm="Orchestrate",
                    split_by=config.SPLIT_BY,
                    synthnet_upstream_edge_direction=True,
                    loss=train_baseline._BASELINE_DEFAULTS["gamora"]["loss"],
                )
            ),
            "gamora_Orchestrate",
        )

    def test_polargate_defaults_to_upstreams_own_loss(self):
        """The defect this guards: train_baseline.py used to hardcode
        nn.MSELoss() for every baseline while train.py trains the primary model
        under SmoothL1(beta=0.01). The label is 48.8% exactly zero, so MSE
        through a terminal sigmoid collapses toward the mean and the comparison
        confounds objective with architecture. Changing the existing three
        (SynthNet/HOGA/DeepGate4) would invalidate runs already made. Gamora
        publishes no loss of its own, so it takes the primary model's; PolarGate
        defaults to upstream's own 'mae' (train.py's --loss_type default) as of
        2026-08-07, per the project author's no-unnecessary-variations directive
        -- see baselines/polargate/PROVENANCE.md.
        """
        import train_baseline

        self.assertEqual(
            train_baseline._BASELINE_DEFAULTS["polargate"]["loss"], "mae"
        )
        for baseline in ("synthnet", "hoga", "deepgate4"):
            with self.subTest(baseline=baseline):
                self.assertEqual(
                    train_baseline._BASELINE_DEFAULTS[baseline]["loss"], "mse"
                )
        # train.py:151 is the source of truth for beta; --loss_beta's argparse
        # default is the only other place the value appears (see
        # test_loss_is_resolved_per_baseline for the built-loss.beta check).
        train_baseline_py = Path(_SRC_DIR) / "train_baseline.py"
        self.assertIn(
            '"--loss_beta", type=float, default=0.01', train_baseline_py.read_text()
        )
        train_py = Path(_SRC_DIR) / "train.py"
        self.assertIn("nn.SmoothL1Loss(beta=0.01)", train_py.read_text())

    def test_missing_required_baseline_arg_fails_cleanly(self):
        result = _run_cli("--csv_paths", "dummy.csv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--baseline", result.stderr)

    def test_baseline_choices_cover_every_ported_baseline(self):
        result = _run_cli("--help")
        self.assertIn(
            "{synthnet,hoga,deepgate4,gamora,polargate}", result.stdout
        )

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
        Gamora's released code defaults to 100 (gnn_multitask.py:532).
        PolarGate publishes 500 (train.py argparse; TODAES Sec 6.2 confirms
        "a maximum of 500 epochs").
        A single shared default would silently give DeepGate4 SynthNet's number.
        """
        sys.path.insert(0, _SRC_DIR)
        try:
            import train_baseline
        finally:
            sys.path.remove(_SRC_DIR)

        self.assertEqual(train_baseline.DEEPGATE4_DEFAULTS["max_epochs"], 200)
        self.assertEqual(train_baseline.SYNTHNET_DEFAULTS["max_epochs"], 80)
        self.assertEqual(train_baseline.GAMORA_DEFAULTS["max_epochs"], 100)
        self.assertEqual(train_baseline.POLARGATE_DEFAULTS["max_epochs"], 500)
        self.assertEqual(
            set(train_baseline._BASELINE_DEFAULTS),
            {"synthnet", "hoga", "deepgate4", "gamora", "polargate"},
        )
        for name, defaults in train_baseline._BASELINE_DEFAULTS.items():
            with self.subTest(baseline=name):
                self.assertEqual(
                    set(defaults),
                    {"batch_size", "lr", "weight_decay", "max_epochs", "loss"},
                )

    def test_loss_is_resolved_per_baseline(self):
        """The three older baselines must keep MSE; Gamora and PolarGate change.

        train_baseline.py used to hardcode nn.MSELoss() for every baseline,
        while the primary model (train.py:151) trains under
        SmoothL1(beta=0.01) -- and those two weight a zero-inflated target very
        differently, so a baseline scored under the wrong one confounds
        architecture with loss choice. Gamora publishes
        no regression loss at all (its task is classification, F.nll_loss at
        gnn_multitask.py:183), so it takes the primary model's. PolarGate
        defaults to upstream's own 'mae' (train.py's --loss_type default) as
        of 2026-08-07 -- see baselines/polargate/PROVENANCE.md. The other
        three keep the loss their runs were made under, and this test is what
        stops that changing by accident.
        """
        import torch.nn as nn

        sys.path.insert(0, _SRC_DIR)
        try:
            import train_baseline
        finally:
            sys.path.remove(_SRC_DIR)

        self.assertEqual(train_baseline.SYNTHNET_DEFAULTS["loss"], "mse")
        self.assertEqual(train_baseline.HOGA_DEFAULTS["loss"], "mse")
        self.assertEqual(train_baseline.DEEPGATE4_DEFAULTS["loss"], "mse")
        self.assertEqual(train_baseline.GAMORA_DEFAULTS["loss"], "smooth_l1")
        self.assertEqual(train_baseline.POLARGATE_DEFAULTS["loss"], "mae")

        mse = train_baseline._build_loss(SimpleNamespace(loss="mse", loss_beta=0.01))
        self.assertIsInstance(mse, nn.MSELoss)

        smooth = train_baseline._build_loss(
            SimpleNamespace(loss="smooth_l1", loss_beta=0.01)
        )
        self.assertIsInstance(smooth, nn.SmoothL1Loss)
        # beta must match train.py's, or "the primary model's loss" is a
        # differently-shaped loss that merely shares a name.
        self.assertEqual(smooth.beta, 0.01)

        mae = train_baseline._build_loss(SimpleNamespace(loss="mae", loss_beta=0.01))
        self.assertIsInstance(mae, nn.L1Loss)

        with self.assertRaises(ValueError):
            train_baseline._build_loss(SimpleNamespace(loss="huber", loss_beta=0.01))

    def test_gamora_node_budget_reflects_the_documented_speed_deviation(self):
        """The Gamora baseline's node budget WAS primary-model parity.

        Originally the literal was pinned equal to config.MAX_TOTAL_NODES_PER_BATCH
        so the baseline trained at the same effective batch (~75 graphs/step)
        as the primary model it is compared against, with no gradient
        accumulation. Raised to 15,000,000 on 2026-08-06 as a deliberate,
        documented deviation from that parity for wall-clock speed -- see
        train_baseline_gamora.sh's own comment above the literal for the
        measured GPU-memory/utilization evidence. The two values are now
        EXPECTED to differ; assert the shell script's literal directly rather
        than against config.MAX_TOTAL_NODES_PER_BATCH, so a future retune of
        either one is a conscious edit here, not a silent drift back into (or
        further out of) parity.
        """
        script = (_SHELL_DIR / "train_baseline_gamora.sh").read_text()
        match = re.search(
            r'(?m)^GAMORA_MAX_NODES_PER_BATCH="\$\{GAMORA_MAX_NODES_PER_BATCH:-(\d+)\}"',
            script,
        )
        self.assertIsNotNone(match, "GAMORA_MAX_NODES_PER_BATCH default not found")
        self.assertEqual(int(match.group(1)), 15_000_000)

    def test_gamora_flags_are_wired(self):
        # train_baseline_gamora.sh passes all of these; a rename would break
        # the job script silently.
        result = _run_cli("--help")
        for flag in (
            "--loss",
            "--torch_compile",
            "--gamora_num_layers",
            "--gamora_hidden_dim",
            "--gamora_max_nodes_per_batch",
        ):
            self.assertIn(flag, result.stdout)

    def test_torch_compile_defaults_off(self):
        """Shared by all four baselines' wrapper; only Gamora's script turns it
        on. A flipped default would silently change what SynthNet/HOGA/
        DeepGate4 execute without any of them being verified under compile --
        see baselines/common/lightning_wrapper.py's module docstring.

        argparse's plain HelpFormatter never prints defaults, so --help can't
        show this, and train_baseline.py builds its parser only inside
        `if __name__ == "__main__":` rather than a reusable function, so it
        can't be imported and re-invoked either. Assert on the source directly:
        the default is a literal in the add_argument call, immediately after
        the flag name.
        """
        source = (Path(_SRC_DIR) / "train_baseline.py").read_text()
        flag_idx = source.find('"--torch_compile"')
        self.assertNotEqual(flag_idx, -1, "--torch_compile flag not found")
        # Scan forward a bounded window rather than matching to the next ")":
        # the type= lambda's own ("true", "1", "yes") tuple contains a ")"
        # first, which a naive [^)]* stops at before reaching default=.
        window = source[flag_idx : flag_idx + 300]
        match = re.search(r"default=(\w+)", window)
        self.assertIsNotNone(match, "--torch_compile default not found")
        self.assertEqual(match.group(1), "False")

    def test_torch_compile_arg_is_wired_to_the_lightning_module(self):
        """The one line connecting the CLI flag to the wrapper.

        Confirmed by mutation: deleting `compile_model=args.torch_compile,`
        from the `BaselineRegressionLightningModule(...)` call left the ENTIRE
        baseline test suite green (137 passed) -- every existing test either
        constructs the wrapper directly (bypassing this line) or never checks
        whether `--torch_compile true` actually reaches it. Source-level
        rather than a full CLI run, matching this file's own established
        pattern for pinning a literal (e.g.
        test_gamora_node_budget_reflects_the_documented_speed_deviation):
        `main()` loads
        the real dataset and touches wandb, which is too heavy to invoke here
        just to prove one kwarg is threaded through.
        """
        source = (Path(_SRC_DIR) / "train_baseline.py").read_text()
        call_start = source.index("model = BaselineRegressionLightningModule(")
        # Scan forward a bounded window rather than to the next ")": one of
        # this call's own kwargs is _build_loss(args), a nested call whose
        # closing paren comes first and would truncate the block early (bit
        # us once already on the --torch_compile default= regex above).
        call_block = source[call_start : call_start + 400]
        self.assertIn(
            "compile_model=args.torch_compile",
            call_block,
            "args.torch_compile is not passed to BaselineRegressionLightningModule",
        )

    def test_only_gamora_script_wires_torch_compile(self):
        """The opt-in is per-script, not per-baseline in train_baseline.py.

        Only Gamora's script passes --torch_compile at all; the other three
        must not, so they keep the parser's off default rather than someone
        copy-pasting Gamora's block into them. This does NOT pin whether
        Gamora's own TORCH_COMPILE defaults true or false -- that value has
        flipped three times in one day (2026-08-06: on for an A100 test that
        regressed, off, back on for a clean H100 test that then hung for over
        an hour on batch 0 and was killed, off again) and is expected to keep
        moving as it gets re-tested; asserting a specific literal here would
        just be testing the same commit that set it. What must stay true
        regardless of that value is that Gamora is the only script wiring the
        flag in the first place.
        """
        scripts = sorted(_SHELL_DIR.glob("train_baseline_*.sh"))
        self.assertTrue(scripts, "no baseline job scripts found")
        for script in scripts:
            with self.subTest(script=script.name):
                text = script.read_text()
                has_flag = bool(
                    re.search(r'(?m)^\s*(?!#)--torch_compile\s+"', text)
                )
                if script.name == "train_baseline_gamora.sh":
                    self.assertTrue(has_flag, "gamora script must pass --torch_compile")
                else:
                    self.assertFalse(
                        has_flag, f"{script.name} must not pass --torch_compile"
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
