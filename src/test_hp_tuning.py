import argparse
import itertools
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Batch, Data

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

import hp_tuning
from models.lightning_model import AIGRegressionLightningModule


class _FakeScore:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeTrial:
    def __init__(self, values: dict, number: int = 3):
        self.values = values
        self.number = number
        self.requested = []

    def suggest_categorical(self, name, choices):
        if name not in self.values:
            raise AssertionError(f"Missing categorical value for '{name}'")
        value = self.values[name]
        if value not in choices:
            raise AssertionError(
                f"Value '{value}' not in categorical choices for '{name}'"
            )
        self.requested.append(name)
        return value

    def suggest_float(self, name, low, high, log=False):
        if name not in self.values:
            raise AssertionError(f"Missing float value for '{name}'")
        value = self.values[name]
        if not (low <= value <= high):
            raise AssertionError(f"Float value '{value}' out of range for '{name}'")
        if name == "lr" and not log:
            raise AssertionError("Expected log=True for lr suggestion")
        self.requested.append(name)
        return value

    def suggest_int(self, name, low, high):
        if name not in self.values:
            raise AssertionError(f"Missing int value for '{name}'")
        value = self.values[name]
        if not isinstance(value, int):
            raise AssertionError(f"Expected int value for '{name}'")
        if not (low <= value <= high):
            raise AssertionError(f"Int value '{value}' out of range for '{name}'")
        self.requested.append(name)
        return value

    def suggest_bool(self, name):
        if name not in self.values:
            raise AssertionError(f"Missing bool value for '{name}'")
        value = self.values[name]
        if not isinstance(value, bool):
            raise AssertionError(f"Expected bool value for '{name}'")
        self.requested.append(name)
        return value


def _run_objective_for_test(trial_values: dict, best_model_score: float | None = 0.25):
    args = argparse.Namespace(
        csv_paths=["algo_a.csv", "algo_b.csv"],
        checkpoint_dir="/tmp/ckpt",
        num_workers=0,
    )
    trial = _FakeTrial(trial_values)

    checkpoint_cb = MagicMock()
    checkpoint_cb.best_model_score = (
        _FakeScore(best_model_score) if best_model_score is not None else None
    )

    with (
        patch("hp_tuning.AIGDataModule") as datamodule_cls,
        patch("hp_tuning.AIGRegressionLightningModule") as model_cls,
        patch("hp_tuning.ModelCheckpoint", return_value=checkpoint_cb),
        patch("hp_tuning.EarlyStopping") as early_stopping_cls,
        patch("hp_tuning.PyTorchLightningPruningCallback") as pruning_cb_cls,
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
        "model_instance": model_cls.return_value,
        "trainer": trainer,
        "trainer_cls": trainer_cls,
        "early_stopping_cls": early_stopping_cls,
        "pruning_cb_cls": pruning_cb_cls,
    }


class TestHpTuningObjectiveWiring(unittest.TestCase):
    def test_none_pe_skips_pos_enc_dim_heads_and_projection(self):
        out = _run_objective_for_test(
            {
                "batch_size": 16,
                "lr": 1e-3,
                "huber_delta": 1.1,
                "encoder_name": "gine",
                "embed_dim": 32,
                "hid_dim": 64,
                "pe_type": "none",
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "last",
            },
            best_model_score=0.73,
        )

        self.assertAlmostEqual(out["result"], 0.73)
        self.assertNotIn("pos_enc_dim", out["trial"].requested)
        self.assertNotIn("heads", out["trial"].requested)
        self.assertNotIn("project_with_pos_enc", out["trial"].requested)

        model_kwargs = out["model_call"].kwargs
        self.assertEqual(model_kwargs["encoder_name"], "gine")
        self.assertEqual(model_kwargs["embed_dim"], 32)
        self.assertEqual(model_kwargs["pe_type"], "none")
        self.assertEqual(model_kwargs["pos_enc_dim"], 0)
        self.assertFalse(model_kwargs["project_with_pos_enc"])
        self.assertEqual(model_kwargs["encoder_kwargs"]["jk_mode"], "last")

    def test_pe_requests_projection_and_dim(self):
        out = _run_objective_for_test(
            {
                "batch_size": 16,
                "lr": 1e-3,
                "huber_delta": 1.1,
                "encoder_name": "gine",
                "embed_dim": 64,
                "hid_dim": 128,
                "pe_type": "level",
                "pos_enc_dim": 32,
                "project_with_pos_enc": True,
                "num_layers": 3,
                "dropout": 0.1,
                "norm_type": "batch",
                "jk_mode": "max",
            }
        )
        self.assertIn("pos_enc_dim", out["trial"].requested)
        self.assertIn("project_with_pos_enc", out["trial"].requested)

        model_kwargs = out["model_call"].kwargs
        self.assertEqual(model_kwargs["pos_enc_dim"], 32)
        self.assertTrue(model_kwargs["project_with_pos_enc"])

    def test_egin_mapping_correctly(self):
        out = _run_objective_for_test(
            {
                "batch_size": 8,
                "lr": 1e-3,
                "huber_delta": 1.0,
                "encoder_name": "egin",
                "embed_dim": 32,
                "hid_dim": 64,
                "pe_type": "none",
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
        # Verify mapping from trial name to internal keyword
        self.assertTrue(enc_kwargs["dot_update"])
        self.assertTrue(enc_kwargs["edge_mlp"])
        self.assertEqual(enc_kwargs["edge_hidden_dim"], 16)
        self.assertEqual(enc_kwargs["num_mlp_layers"], 2)


class TestTrialOptionModelInstantiation(unittest.TestCase):
    def _make_encoder_kwargs(self, encoder_name: str, norm_type: str = "batch"):
        kwargs = {
            "num_layers": 3,
            "norm_type": norm_type,
            "dropout": 0.1,
            "hid_dim": 128,
            "jk_mode": "last",
        }
        if encoder_name in {"transformer_conv", "graphgps"}:
            kwargs["heads"] = 4
        return kwargs

    def test_all_combinations_instantiate(self):
        encoders = ["gine", "transformer_conv", "egin"]
        pe_types = ["none", "level"]
        projections = [True, False]

        for enc, pe, proj in itertools.product(encoders, pe_types, projections):
            with self.subTest(enc=enc, pe=pe, proj=proj):
                enc_kwargs = self._make_encoder_kwargs(enc)
                model = AIGRegressionLightningModule(
                    encoder_name=enc,
                    embed_dim=128,
                    pe_type=pe,
                    pos_enc_dim=16 if pe != "none" else 0,
                    project_with_pos_enc=proj,
                    encoder_kwargs=enc_kwargs,
                    hid_dim=enc_kwargs["hid_dim"],
                )
                self.assertEqual(model.model.project_with_pos_enc, proj)


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

                lm = AIGRegressionLightningModule(
                    encoder_name=encoder,
                    embed_dim=32,
                    node_input_dim=4,
                    num_edge_types=2,
                    task_out_dim=1,
                    pe_type=pe_type,
                    pos_enc_dim=pos_enc_dim,
                    project_with_pos_enc=proj,
                    encoder_kwargs=encoder_kwargs,
                    hid_dim=32,
                )

                # Dummy Data Forward Pass
                num_nodes = 5
                x = torch.zeros((num_nodes, 4), dtype=torch.float32)
                edge_index = torch.tensor(
                    [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
                )
                edge_attr = torch.zeros((4, 2), dtype=torch.float32)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                if pe_type == "level":
                    data.pos_enc = torch.randint(0, 5, (num_nodes, 1), dtype=torch.long)

                batch = Batch.from_data_list([data])
                out = lm(batch)
                self.assertEqual(out.shape, (1, 1))
