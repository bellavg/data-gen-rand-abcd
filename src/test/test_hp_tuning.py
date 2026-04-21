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
    optuna_integration_stub = types.ModuleType("optuna.integration")

    class _NoOpPruningCallback:
        def __init__(self, *args, **kwargs):
            pass

    optuna_integration_stub.PyTorchLightningPruningCallback = _NoOpPruningCallback
    optuna_stub.integration = optuna_integration_stub
    sys.modules["optuna"] = optuna_stub
    sys.modules["optuna.integration"] = optuna_integration_stub

# Import the module under the package namespace
import hp_tuning
from models.lightning_model import AIGRegressionLightningModule


class _FakeScore:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeTrial:
    """Mock Optuna Trial to capture requested params and choices."""

    def __init__(self, values: dict, number: int = 3):
        self.values = values
        self.number = number
        self.requested_choices = {}  # Map name -> choices provided by objective

    def suggest_categorical(self, name, choices):
        if name not in self.values:
            raise AssertionError(f"Missing categorical value for '{name}'")
        value = self.values[name]
        if value not in choices:
            raise AssertionError(
                f"Value '{value}' not in categorical choices for '{name}'. "
                f"Choices were: {choices}"
            )
        self.requested_choices[name] = choices
        return value

    def suggest_float(self, name, low, high, log=False):
        if name not in self.values:
            raise AssertionError(f"Missing float value for '{name}'")
        value = self.values[name]
        if not (low <= value <= high):
            raise AssertionError(f"Float value '{value}' out of range for '{name}'")
        return value

    def suggest_int(self, name, low, high):
        if name not in self.values:
            raise AssertionError(f"Missing int value for '{name}'")
        value = self.values[name]
        if not isinstance(value, int):
            raise AssertionError(f"Expected int value for '{name}'")
        if not (low <= value <= high):
            raise AssertionError(f"Int value '{value}' out of range for '{name}'")
        return value

    def suggest_bool(self, name):
        if name not in self.values:
            raise AssertionError(f"Missing bool value for '{name}'")
        value = self.values[name]
        return value


def _run_objective_for_test(trial_values: dict, best_model_score: float | None = 0.25):
    args = argparse.Namespace(
        csv_paths=["algo_a.csv", "algo_b.csv"],
        checkpoint_dir="/tmp/ckpt",
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

    def test_hidden_dim_filtering(self):
        """Verify hidden_dim choices are exposed as expected."""
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
