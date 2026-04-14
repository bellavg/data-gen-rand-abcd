import argparse
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

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
			raise AssertionError(f"Value '{value}' not in categorical choices for '{name}'")
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
	def test_none_pe_skips_pos_enc_dim_and_heads(self):
		out = _run_objective_for_test(
			{
				"batch_size": 16,
				"lr": 1e-3,
				"huber_delta": 1.1,
				"encoder_name": "gine",
				"embed_dim": 256,
				"pe_type": "none",
				"num_layers": 3,
				"dropout": 0.1,
				"norm_type": "batch",
			},
			best_model_score=0.73,
		)

		self.assertAlmostEqual(out["result"], 0.73)
		self.assertNotIn("pos_enc_dim", out["trial"].requested)
		self.assertNotIn("heads", out["trial"].requested)

		dm_kwargs = out["datamodule_call"].kwargs
		self.assertEqual(dm_kwargs["csv_paths"], ["algo_a.csv", "algo_b.csv"])
		self.assertIsNone(dm_kwargs["positional_encoding"])
		self.assertEqual(dm_kwargs["batch_size"], 16)

		model_kwargs = out["model_call"].kwargs
		self.assertEqual(model_kwargs["encoder_name"], "gine")
		self.assertEqual(model_kwargs["embed_dim"], 256)
		self.assertEqual(model_kwargs["pe_type"], "none")
		self.assertEqual(model_kwargs["pos_enc_dim"], 0)
		self.assertEqual(model_kwargs["lr"], 1e-3)
		self.assertEqual(model_kwargs["huber_delta"], 1.1)
		self.assertNotIn("heads", model_kwargs["encoder_kwargs"])

		out["trainer"].fit.assert_called_once()
		fit_args = out["trainer"].fit.call_args.args
		self.assertEqual(len(fit_args), 1)
		self.assertIs(fit_args[0], out["model_instance"])
		fit_kwargs = out["trainer"].fit.call_args.kwargs
		self.assertIn("datamodule", fit_kwargs)

	def test_transformer_and_graphgps_request_heads_and_positional_dim(self):
		for encoder_name in ["transformer_conv", "graphgps"]:
			with self.subTest(encoder_name=encoder_name):
				out = _run_objective_for_test(
					{
						"batch_size": 32,
						"lr": 5e-4,
						"huber_delta": 1.5,
						"encoder_name": encoder_name,
						"embed_dim": 384,
						"pe_type": "level",
						"pos_enc_dim": 64,
						"num_layers": 4,
						"dropout": 0.2,
						"norm_type": "layer",
						"heads": 8,
					}
				)

				self.assertIn("pos_enc_dim", out["trial"].requested)
				self.assertIn("heads", out["trial"].requested)

				dm_kwargs = out["datamodule_call"].kwargs
				self.assertEqual(dm_kwargs["positional_encoding"], "level")
				self.assertEqual(dm_kwargs["batch_size"], 32)

				model_kwargs = out["model_call"].kwargs
				self.assertEqual(model_kwargs["pos_enc_dim"], 64)
				self.assertEqual(model_kwargs["encoder_kwargs"]["heads"], 8)
				self.assertEqual(model_kwargs["encoder_kwargs"]["norm_type"], "layer")

	def test_all_batch_size_options_are_passed_to_datamodule(self):
		for batch_size in [8, 16, 32, 64, 128]:
			with self.subTest(batch_size=batch_size):
				out = _run_objective_for_test(
					{
						"batch_size": batch_size,
						"lr": 8e-4,
						"huber_delta": 1.0,
						"encoder_name": "egin",
						"embed_dim": 128,
						"pe_type": "none",
						"num_layers": 3,
						"dropout": 0.0,
						"norm_type": "none",
					}
				)
				self.assertEqual(out["datamodule_call"].kwargs["batch_size"], batch_size)

	def test_returns_fallback_when_checkpoint_score_missing(self):
		out = _run_objective_for_test(
			{
				"batch_size": 8,
				"lr": 1e-3,
				"huber_delta": 1.0,
				"encoder_name": "gine",
				"embed_dim": 128,
				"pe_type": "none",
				"num_layers": 3,
				"dropout": 0.1,
				"norm_type": "graph",
			},
			best_model_score=None,
		)
		self.assertEqual(out["result"], 1.0)


class TestTrialOptionModelInstantiation(unittest.TestCase):
	def _make_encoder_kwargs(self, encoder_name: str, norm_type: str = "batch"):
		kwargs = {
			"num_layers": 3,
			"norm_type": norm_type,
			"dropout": 0.1,
		}
		if encoder_name in {"transformer_conv", "graphgps"}:
			kwargs["heads"] = 4
		return kwargs

	def test_encoder_pe_combinations_instantiate(self):
		encoders = ["gine", "transformer_conv", "graphgps", "egin"]
		pe_types = ["none", "level", "pi_paths", "local_sp_sum"]

		for encoder_name in encoders:
			for pe_type in pe_types:
				with self.subTest(encoder_name=encoder_name, pe_type=pe_type):
					model = AIGRegressionLightningModule(
						encoder_name=encoder_name,
						embed_dim=128,
						pe_type=pe_type,
						pos_enc_dim=0 if pe_type == "none" else 16,
						encoder_kwargs=self._make_encoder_kwargs(encoder_name),
						huber_delta=1.2,
					)
					self.assertEqual(model.hparams.encoder_name, encoder_name)
					self.assertEqual(model.hparams.pe_type, pe_type)

	def test_all_embed_dim_choices_instantiate(self):
		for embed_dim in [128, 256, 384, 512, 768]:
			with self.subTest(embed_dim=embed_dim):
				model = AIGRegressionLightningModule(
					encoder_name="gine",
					embed_dim=embed_dim,
					pe_type="none",
					pos_enc_dim=0,
					encoder_kwargs=self._make_encoder_kwargs("gine"),
				)
				self.assertEqual(model.hparams.embed_dim, embed_dim)

	def test_all_norm_type_choices_instantiate(self):
		norm_types = ["batch", "layer", "graph", "none"]
		for encoder_name in ["gine", "transformer_conv", "graphgps", "egin"]:
			for norm_type in norm_types:
				with self.subTest(encoder_name=encoder_name, norm_type=norm_type):
					model = AIGRegressionLightningModule(
						encoder_name=encoder_name,
						embed_dim=128,
						pe_type="none",
						pos_enc_dim=0,
						encoder_kwargs=self._make_encoder_kwargs(encoder_name, norm_type),
					)
					self.assertEqual(model.hparams.encoder_kwargs["norm_type"], norm_type)
