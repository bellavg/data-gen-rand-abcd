import argparse
import itertools
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Batch, Data

# Mock Optuna if not installed
try:
    import optuna  # noqa: F401
except ModuleNotFoundError:
    optuna_stub = types.ModuleType("optuna")
    optuna_stub.Trial = object
    sys.modules["optuna"] = optuna_stub

# Import the module under the package namespace
import hp_tuning
import hp_tuning_utils
from models.lightning_model import AIGRegressionLightningModule


class _FakeScore:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeTrial:
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.number = 0  # Add this attribute
        self.requested_choices = {}  # Required for test_none_pe_logic

    def suggest_categorical(self, name, choices):
        self.requested_choices[name] = choices  # Track what was suggested
        val = self.params.get(name, choices[0])
        self.params[name] = val
        return val

    def suggest_int(self, name, low, high, step=1, log=False):
        self.requested_choices[name] = (low, high)
        val = self.params.get(name, low)
        self.params[name] = val
        return val

    def suggest_float(self, name, low, high, step=None, log=False):
        self.requested_choices[name] = (low, high)
        val = self.params.get(name, low)
        self.params[name] = val
        return val


def _run_objective_for_test(trial_values: dict, best_model_score: float | None = 0.25):
    args = argparse.Namespace(
        csv_paths=["algo_a.csv", "algo_b.csv"],
        checkpoint_dir="/tmp/ckpt",
        cache_dir="/tmp/hp_cache",
        num_workers=0,
        train_samples=25000,
        log_dir="/tmp",
    )
    trial = _FakeTrial(trial_values)

    checkpoint_cb = MagicMock()
    checkpoint_cb.best_model_score = (
        _FakeScore(best_model_score) if best_model_score is not None else None
    )

    with (
        patch("hp_tuning.AIGDataModule") as datamodule_cls,
        patch("hp_tuning.AIGRegressionLightningModule") as model_cls,
        patch(
            "pytorch_lightning.callbacks.ModelCheckpoint", return_value=checkpoint_cb
        ),
        patch("hp_tuning.pl.Trainer") as trainer_cls,
    ):
        trainer = MagicMock()
        trainer_cls.return_value = trainer
        result = hp_tuning.objective(trial, args)

    return {
        "result": result,
        "trial": trial,
        "datamodule_call": datamodule_cls.call_args,
        "model_call": model_cls.call_args,
    }


class TestHpTuningObjectiveWiring(unittest.TestCase):
    def test_none_pe_logic(self):
        out = _run_objective_for_test(
            {
                "batch_size": 16,
                "lr": 1e-3,
                "huber_delta": 1.1,
                "encoder_name": "gine",
                "hidden_dim": 64,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
            }
        )

        # Verify PE specific params were skipped
        self.assertNotIn("pos_enc_dim", out["trial"].requested_choices)
        self.assertNotIn("project_with_pos_enc", out["trial"].requested_choices)

        # Verify DataModule received None for positional_encoding
        self.assertIsNone(out["datamodule_call"].kwargs["positional_encoding"])

    def test_hidden_dim_choices_are_fixed_across_encoders(self):
        """hidden_dim must use the same choice set for all encoders.

        Using different sets per encoder caused an Optuna RDB error:
        'CategoricalDistribution does not support dynamic value space'.
        This ensures the bug cannot regress.
        """
        all_encoders = [
            "gine",
            "transformer_conv",
            "graphgps",
            "egin",
            "gcn",
            "vanilla_mpnn",
        ]
        seen_choices = None
        for encoder in all_encoders:
            params = {
                "batch_size": 4,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": encoder,
                "hidden_dim": 32,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 2,
                "dropout": 0.0,
                "norm_type": "batch",
                "jk_mode": "last",
            }
            if encoder in ("transformer_conv", "graphgps"):
                params["heads"] = 1
            if encoder == "egin":
                params.update(
                    {
                        "num_mlp_layers": 2,
                        "egin_dot_update": False,
                        "egin_edge_mlp": False,
                        "edge_hidden_dim": 32,
                    }
                )
            out = _run_objective_for_test(params)
            choices = tuple(out["trial"].requested_choices["hidden_dim"])
            if seen_choices is None:
                seen_choices = choices
            else:
                self.assertEqual(
                    choices,
                    seen_choices,
                    f"hidden_dim choices differ for encoder '{encoder}': "
                    f"{choices} != {seen_choices}",
                )
        # 512 must be reachable from every encoder
        self.assertIn(512, seen_choices)

    def test_jk_mode_choices_are_fixed_across_encoders(self):
        """jk_mode must include 'cat' for all encoders (same distribution set).

        Previously attention encoders excluded 'cat', causing dynamic value space errors.
        """
        all_encoders = [
            "gine",
            "transformer_conv",
            "graphgps",
            "egin",
            "gcn",
            "vanilla_mpnn",
        ]
        for encoder in all_encoders:
            params = {
                "batch_size": 4,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": encoder,
                "hidden_dim": 32,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 2,
                "dropout": 0.0,
                "norm_type": "batch",
                "jk_mode": "last",
            }
            if encoder in ("transformer_conv", "graphgps"):
                params["heads"] = 1
            if encoder == "egin":
                params.update(
                    {
                        "num_mlp_layers": 2,
                        "egin_dot_update": False,
                        "egin_edge_mlp": False,
                        "edge_hidden_dim": 32,
                    }
                )
            with self.subTest(encoder=encoder):
                out = _run_objective_for_test(params)
                jk_choices = out["trial"].requested_choices["jk_mode"]
                self.assertIn(
                    "cat",
                    jk_choices,
                    f"'cat' missing from jk_mode choices for encoder '{encoder}'",
                )

    def test_hidden_dim_filtering(self):
        """Verify hidden_dim choices contain the expected values for any encoder."""
        out = _run_objective_for_test(
            {
                "batch_size": 16,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": "gine",
                "hidden_dim": 256,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
            }
        )

        hid_choices = out["trial"].requested_choices["hidden_dim"]
        self.assertIn(32, hid_choices)
        self.assertIn(128, hid_choices)
        self.assertIn(256, hid_choices)
        # 512 must also be present (unified choice set)
        self.assertIn(512, hid_choices)

    def test_egin_mapping_correctly(self):
        out = _run_objective_for_test(
            {
                "batch_size": 8,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": "egin",
                "hidden_dim": 64,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
                "num_mlp_layers": 2,
                "egin_dot_update": True,
                "egin_edge_mlp": True,
                "edge_hidden_dim": 16,
            }
        )

        enc_kwargs = out["model_call"].kwargs["encoder_kwargs"]
        self.assertTrue(enc_kwargs["dot_update"])
        self.assertTrue(enc_kwargs["edge_mlp"])
        self.assertEqual(enc_kwargs["edge_hidden_dim"], 16)


class TestTrialOptionModelInstantiation(unittest.TestCase):
    def test_all_combinations_instantiate(self):
        encoders = ["gine", "egin"]
        pe_types = ["none", "pi_paths"]  # Test path-based PE
        projections = [True, False]

        for enc, pe, proj in itertools.product(encoders, pe_types, projections):
            with self.subTest(enc=enc, pe=pe, proj=proj):
                enc_kwargs = {
                    "num_layers": 2,
                    "norm_type": "batch",
                    "dropout": 0.1,
                    "hid_dim": 64,
                    "jk_mode": "last",
                }
                # FIXED: Removed 'hid_dim' from direct args
                model = AIGRegressionLightningModule(
                    encoder_name=enc,
                    hidden_dim=64,
                    pe_type=pe,
                    pos_enc_dim=16 if pe != "none" else 0,
                    encoder_kwargs=enc_kwargs,
                )
                self.assertIsNotNone(model)


class TestComprehensiveCombinations(unittest.TestCase):
    def test_all_hyperparameter_combinations_and_forward_pass(self):
        encoders = ["gine", "transformer_conv", "graphgps", "gcn", "vanilla_mpnn"]
        pe_types = ["none", "level"]
        jk_modes = ["last", "max"]
        projections = [True, False]

        for encoder, pe_type, jk_mode, proj in itertools.product(
            encoders, pe_types, jk_modes, projections
        ):
            with self.subTest(encoder=encoder, pe=pe_type, jk=jk_mode, proj=proj):
                pos_enc_dim = 16 if pe_type != "none" else 0
                encoder_kwargs = {
                    "num_layers": 2,
                    "norm_type": "batch",
                    "dropout": 0.0,
                    "jk_mode": jk_mode,
                    "hid_dim": 32,
                }
                if encoder in ["transformer_conv", "graphgps"]:
                    encoder_kwargs["heads"] = 4

                # FIXED: Removed 'num_edge_types' and 'hid_dim'
                lm = AIGRegressionLightningModule(
                    encoder_name=encoder,
                    hidden_dim=32,
                    node_input_dim=4,
                    task_out_dim=1,
                    pe_type=pe_type,
                    pos_enc_dim=pos_enc_dim,
                    encoder_kwargs=encoder_kwargs,
                )

                # Dummy Data Forward Pass
                num_nodes = 5
                data = Data(
                    x=torch.zeros((num_nodes, 4)),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
                    edge_attr=torch.zeros((4, 2)),
                )

                if pe_type == "level":
                    data.pos_enc = torch.randint(0, 5, (num_nodes, 1))

                batch = Batch.from_data_list([data])
                out = lm(batch)
                self.assertEqual(out.shape, (1, 1))


class TestOOMTrialCleanup(unittest.TestCase):
    def _make_trial(self, number: int) -> _FakeTrial:
        trial = _FakeTrial(
            {
                "batch_size": 4,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": "gine",
                "hidden_dim": 32,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 2,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
            }
        )
        trial.number = number
        return trial

    def test_cleanup_runs_before_third_trial_after_oom(self):
        args = argparse.Namespace(
            csv_paths=["algo_a.csv", "algo_b.csv"],
            checkpoint_dir="/tmp/ckpt",
            cache_dir="/tmp/hp_cache",
            num_workers=0,
            train_samples=20000,
            log_dir="/tmp",
            hard_prune_risk=999999.0,
        )

        events: list[str] = []
        fit_counter = {"value": 0}

        class _FakeTrainer:
            def __init__(self):
                self.callbacks = []
                self.loggers = []
                self.optimizers = [object()]

            def fit(self, *args, **kwargs):
                fit_counter["value"] += 1
                trial_idx = fit_counter["value"]
                events.append(f"fit_{trial_idx}_start")
                if trial_idx == 2:
                    raise RuntimeError("CUDA out of memory")
                events.append(f"fit_{trial_idx}_end")

        fake_datamodule = MagicMock()
        fake_datamodule.train_dataloader.return_value = object()
        fake_datamodule.val_dataloader.return_value = object()

        with (
            patch("hp_tuning.AIGDataModule", return_value=fake_datamodule),
            patch("hp_tuning.AIGRegressionLightningModule", return_value=MagicMock()),
            patch("hp_tuning.pl.Trainer", side_effect=lambda *a, **k: _FakeTrainer()),
            patch(
                "hp_tuning_utils.gc.collect", side_effect=lambda: events.append("gc")
            ),
            patch("hp_tuning_utils.torch.cuda.is_available", return_value=True),
            patch(
                "hp_tuning_utils.torch.cuda.empty_cache",
                side_effect=lambda: events.append("empty_cache"),
            ),
            patch("hp_tuning_utils.torch.cuda.synchronize"),
        ):
            _ = hp_tuning.objective(self._make_trial(0), args)
            with self.assertRaises(hp_tuning.optuna.TrialPruned):
                hp_tuning.objective(self._make_trial(1), args)
            _ = hp_tuning.objective(self._make_trial(2), args)

        fit2_start = events.index("fit_2_start")
        fit3_start = events.index("fit_3_start")
        between = events[fit2_start + 1 : fit3_start]

        self.assertIn("gc", between)
        self.assertIn("empty_cache", between)


class TestRuntimeOOMClassification(unittest.TestCase):
    def test_classifies_mmap_enomem_as_host_oom(self):
        exc = RuntimeError(
            "unable to mmap 1012833 bytes from file </tmp/a.pt>: "
            "Cannot allocate memory (12)"
        )
        kind = hp_tuning_utils._classify_oom_runtime_error(exc)
        self.assertIsNotNone(kind)
        self.assertEqual(kind, "host")

    def test_classifies_cuda_oom_as_cuda(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        kind = hp_tuning_utils._classify_oom_runtime_error(exc)
        self.assertIsNotNone(kind)
        self.assertEqual(kind, "cuda")


class TestHardPruneRiskScope(unittest.TestCase):
    def test_hard_prune_applies_to_large_batch_trials(self):
        args = argparse.Namespace(
            csv_paths=["algo_a.csv", "algo_b.csv"],
            checkpoint_dir="/tmp/ckpt",
            cache_dir="/tmp/hp_cache",
            num_workers=0,
            train_samples=20000,
            log_dir="/tmp",
            hard_prune=True,
            hard_prune_risk=1.0,
        )

        trial = _FakeTrial(
            {
                "batch_size": 32,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": "gcn",
                "hidden_dim": 32,
                "pe_type": "none",
                "pooling_type": "mean",
                "num_layers": 2,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
            }
        )

        with (
            patch(
                "hp_tuning.AIGDataModule",
                side_effect=AssertionError(
                    "hard prune should trigger before datamodule construction"
                ),
            ),
            patch("hp_tuning.torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaises(hp_tuning.optuna.TrialPruned):
                hp_tuning.objective(trial, args)


class TestSeedStudyFromBest(unittest.TestCase):
    """Unit tests for _seed_study_from_best."""

    def _make_frozen_trial(self, number, value, eligible, params):
        t = MagicMock()
        t.number = number
        t.value = value
        t.state = optuna.trial.TrialState.COMPLETE
        t.user_attrs = {"selection_eligible": eligible}
        t.params = params
        t.distributions = {k: MagicMock() for k in params.keys()}
        return t

    def test_enqueues_top_n_eligible_trials(self):
        params_a = {"lr": 1e-3, "hidden_dim": 64}
        params_b = {"lr": 5e-4, "hidden_dim": 128}
        params_c = {"lr": 1e-2, "hidden_dim": 32}

        trials = [
            self._make_frozen_trial(0, 0.5, True, params_a),
            self._make_frozen_trial(1, 0.3, True, params_b),  # best eligible
            self._make_frozen_trial(2, 0.1, False, params_c),  # not eligible
            self._make_frozen_trial(3, 0.4, True, params_a),
        ]

        fake_source_study = MagicMock()
        fake_source_study.trials = trials

        dest_study = MagicMock()
        dest_study.get_trials.return_value = []
        enqueued = []

        def _enqueue(params, **kwargs):
            enqueued.append((params, kwargs))

        dest_study.enqueue_trial.side_effect = _enqueue

        with (
            patch("hp_tuning.RDBStorage"),
            patch("hp_tuning.optuna.load_study", return_value=fake_source_study),
        ):
            hp_tuning._seed_study_from_best(
                dest_study,
                source_db_url="sqlite:///fake.db",
                source_study_name="stage1",
                top_n=2,
                seed_mode="enqueue",
            )

        # Only eligible trials, sorted by value asc, top 2
        self.assertEqual(len(enqueued), 2)
        self.assertEqual(enqueued[0][0], params_b)  # value=0.3 (best eligible)
        self.assertEqual(enqueued[1][0], params_a)  # value=0.4 (next best; trial 3)
        self.assertTrue(enqueued[0][1]["skip_if_exists"])
        self.assertEqual(enqueued[0][1]["user_attrs"]["seed_mode"], "enqueue")

    def test_imports_top_n_eligible_trials(self):
        params_a = {"lr": 1e-3, "hidden_dim": 64}
        params_b = {"lr": 5e-4, "hidden_dim": 128}

        trials = [
            self._make_frozen_trial(0, 0.5, True, params_a),
            self._make_frozen_trial(1, 0.3, True, params_b),
        ]

        fake_source_study = MagicMock()
        fake_source_study.trials = trials

        dest_study = MagicMock()
        dest_study.get_trials.return_value = []
        created_trials = []
        imported = []

        def _create_trial(**kwargs):
            created_trials.append(kwargs)
            return kwargs

        dest_study.add_trial.side_effect = lambda trial: imported.append(trial)

        with (
            patch("hp_tuning.RDBStorage"),
            patch("hp_tuning.optuna.load_study", return_value=fake_source_study),
            patch("hp_tuning.optuna.trial.create_trial", side_effect=_create_trial),
        ):
            hp_tuning._seed_study_from_best(
                dest_study,
                source_db_url="sqlite:///fake.db",
                source_study_name="stage1",
                top_n=2,
                seed_mode="import",
            )

        self.assertEqual(len(imported), 2)
        self.assertEqual(created_trials[0]["params"], params_b)
        self.assertEqual(created_trials[1]["params"], params_a)
        self.assertEqual(created_trials[0]["user_attrs"]["seed_mode"], "import")

    def test_skips_ineligible_trials(self):
        trials = [
            self._make_frozen_trial(0, 0.2, False, {"lr": 1e-3}),
            self._make_frozen_trial(1, 0.9, False, {"lr": 5e-3}),
        ]
        fake_source_study = MagicMock()
        fake_source_study.trials = trials

        dest_study = MagicMock()
        dest_study.get_trials.return_value = []
        with (
            patch("hp_tuning.RDBStorage"),
            patch("hp_tuning.optuna.load_study", return_value=fake_source_study),
        ):
            hp_tuning._seed_study_from_best(
                dest_study,
                source_db_url="sqlite:///fake.db",
                source_study_name="stage1",
                top_n=5,
                seed_mode="enqueue",
            )
        dest_study.enqueue_trial.assert_not_called()
        dest_study.add_trial.assert_not_called()

    def test_top_n_zero_is_noop(self):
        dest_study = MagicMock()
        with patch("hp_tuning.RDBStorage"):
            hp_tuning._seed_study_from_best(
                dest_study,
                source_db_url="sqlite:///fake.db",
                source_study_name="stage1",
                top_n=0,
                seed_mode="enqueue",
            )
        dest_study.enqueue_trial.assert_not_called()
        dest_study.add_trial.assert_not_called()

    def test_enqueue_error_is_raised(self):
        """A broken param set should propagate instead of being swallowed."""
        params_good = {"lr": 1e-3}
        params_bad = {"lr": 999}  # will raise when enqueued
        trials = [
            self._make_frozen_trial(0, 0.1, True, params_bad),
            self._make_frozen_trial(1, 0.2, True, params_good),
        ]
        fake_source_study = MagicMock()
        fake_source_study.trials = trials

        dest_study = MagicMock()
        dest_study.get_trials.return_value = []

        enqueued = []

        def _enqueue(params, **kwargs):
            if params is params_bad:
                raise ValueError("bad param")
            enqueued.append(params)

        dest_study.enqueue_trial.side_effect = _enqueue

        with (
            patch("hp_tuning.RDBStorage"),
            patch("hp_tuning.optuna.load_study", return_value=fake_source_study),
        ):
            with self.assertRaises(ValueError):
                hp_tuning._seed_study_from_best(
                    dest_study,
                    source_db_url="sqlite:///fake.db",
                    source_study_name="stage1",
                    top_n=2,
                    seed_mode="enqueue",
                )

    def test_import_includes_params_even_if_present_in_destination_study(self):
        params_a = {"lr": 1e-3, "hidden_dim": 64}
        params_b = {"lr": 5e-4, "hidden_dim": 128}
        trials = [
            self._make_frozen_trial(10, 0.1, True, params_a),
            self._make_frozen_trial(11, 0.2, True, params_b),
        ]
        fake_source_study = MagicMock()
        fake_source_study.trials = trials

        existing = MagicMock()
        existing.params = params_a
        existing.user_attrs = {}

        dest_study = MagicMock()
        dest_study.get_trials.return_value = [existing]
        imported = []
        dest_study.add_trial.side_effect = lambda trial: imported.append(trial)

        with (
            patch("hp_tuning.RDBStorage"),
            patch("hp_tuning.optuna.load_study", return_value=fake_source_study),
            patch(
                "hp_tuning.optuna.trial.create_trial",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            hp_tuning._seed_study_from_best(
                dest_study,
                source_db_url="sqlite:///fake.db",
                source_study_name="stage1",
                top_n=2,
                seed_mode="import",
            )

        self.assertEqual(len(imported), 2)
        self.assertEqual(imported[0]["params"], params_a)
        self.assertEqual(imported[1]["params"], params_b)


class TestMemoryGuardBatchAccumulation(unittest.TestCase):
    @staticmethod
    def _make_graph(num_nodes: int, num_edges: int) -> Data:
        src = torch.arange(num_edges) % max(1, num_nodes)
        dst = (torch.arange(num_edges) + 1) % max(1, num_nodes)
        edge_index = torch.stack([src, dst], dim=0)
        return Data(x=torch.zeros((num_nodes, 4)), edge_index=edge_index)

    def test_guard_prunes_when_batch_total_exceeds_limit(self):
        guarded_collate = hp_tuning_utils._build_guarded_collate(
            {
                "max_tokens": 100.0,
                "hidden_dim": 1,
                "num_layers": 1,
                "jk_mode": "last",
                "encoder_name": "gine",
                "heads": 1,
                "expansion_factor": 1.0,
            }
        )

        # Each graph is 80 tokens (40 nodes + 40 edges) so each one is safe alone,
        # but the two-graph batch totals 160 and must be pruned.
        graphs = [self._make_graph(40, 40), self._make_graph(40, 40)]
        with self.assertRaises(hp_tuning_utils.HPMemoryGuardError) as caught:
            guarded_collate(graphs)

        self.assertIn("batch_tokens", str(caught.exception))

    def test_guard_allows_batch_under_limit(self):
        guarded_collate = hp_tuning_utils._build_guarded_collate(
            {
                "max_tokens": 100.0,
                "hidden_dim": 1,
                "num_layers": 1,
                "jk_mode": "last",
                "encoder_name": "gine",
                "heads": 1,
                "expansion_factor": 1.0,
            }
        )

        batch = guarded_collate([self._make_graph(20, 20), self._make_graph(20, 20)])
        self.assertEqual(batch.num_graphs, 2)

    def test_error_message_contains_total_batch_tokens(self):
        """Regression: error message must name total_batch_tokens, not total_tokens."""
        guarded_collate = hp_tuning_utils._build_guarded_collate(
            {
                "max_tokens": 50.0,
                "hidden_dim": 1,
                "num_layers": 1,
                "jk_mode": "last",
                "encoder_name": "gine",
                "heads": 1,
                "expansion_factor": 1.0,
            }
        )
        with self.assertRaises(hp_tuning_utils.HPMemoryGuardError) as ctx:
            guarded_collate([self._make_graph(30, 30), self._make_graph(30, 30)])
        msg = str(ctx.exception)
        self.assertIn("batch_tokens=", msg)
        self.assertNotIn("total_tokens=", msg)


class TestPeriodicMemoryReleaseCallback(unittest.TestCase):
    def test_releases_memory_for_train_and_validation_hooks(self):
        callback = hp_tuning.PeriodicMemoryReleaseCallback(
            every_train_steps=500,
            every_val_batches=2,
        )
        trainer = types.SimpleNamespace(global_step=1000)
        pl_module = MagicMock()

        with patch.object(
            hp_tuning.PeriodicMemoryReleaseCallback,
            "_release_memory",
        ) as release_mock:
            callback.on_train_batch_end(
                trainer,
                pl_module,
                outputs=None,
                batch=None,
                batch_idx=0,
            )
            callback.on_validation_start(trainer, pl_module)
            callback.on_validation_batch_end(
                trainer,
                pl_module,
                outputs=None,
                batch=None,
                batch_idx=0,
                dataloader_idx=0,
            )
            callback.on_validation_batch_end(
                trainer,
                pl_module,
                outputs=None,
                batch=None,
                batch_idx=1,
                dataloader_idx=0,
            )
            callback.on_validation_end(trainer, pl_module)

        # train-end + val-start + val-batch(2nd) + val-end
        self.assertEqual(release_mock.call_count, 4)


class TestPruningCallbackStepSemantics(unittest.TestCase):
    def test_skips_sanity_and_deduplicates_same_step(self):
        trial = MagicMock()
        trial.should_prune.return_value = False
        callback = hp_tuning.PyTorchLightningPruningCallback(
            trial,
            monitor="val/mae_node",
        )

        trainer = types.SimpleNamespace(
            callback_metrics={"val/mae_node": torch.tensor(0.123)},
            current_epoch=0,
            global_step=10000,
            sanity_checking=True,
        )
        pl_module = MagicMock()

        # Sanity validation must not report to Optuna.
        callback.on_validation_epoch_end(trainer, pl_module)
        trial.report.assert_not_called()

        # First real validation reports at aligned validation-check index 0.
        trainer.sanity_checking = False
        callback.on_validation_epoch_end(trainer, pl_module)
        self.assertEqual(trial.report.call_count, 1)
        self.assertEqual(trial.report.call_args.kwargs.get("step"), 0)

        # Repeated callback at same step should be ignored.
        callback.on_validation_epoch_end(trainer, pl_module)
        self.assertEqual(trial.report.call_count, 1)

        # Next distinct validation step increments the aligned index.
        trainer.global_step = 12000
        callback.on_validation_epoch_end(trainer, pl_module)
        self.assertEqual(trial.report.call_count, 2)
        self.assertEqual(trial.report.call_args.kwargs.get("step"), 1)
