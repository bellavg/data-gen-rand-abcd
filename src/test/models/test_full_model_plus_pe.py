from __future__ import annotations

import unittest

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import your models and layers
from src.models.base_model import UnifiedGraphBaseModel
from src.models.layers.positional_encodings import get_pe_transform
from src.models.lightning_model import AIGRegressionLightningModule
from src.models.model_utils import get_batch_positional_encoding

# ---------------------------------------------------------------------------
# Shared constants & dummy data generators for AIGs
# ---------------------------------------------------------------------------

NUM_NODES = 12
NUM_EDGES = 16
IN_DIM = 4  # 4 AIG node types [Const, PI, Gate, PO]
EDGE_DIM = 2  # e.g., standard vs. inverted edges
HIDDEN_DIM = 16
OUT_DIM = 1  # Single-target regression [Node Opt]
PE_DIM = 8


def _make_aig_data(seed: int = 42) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(NUM_NODES, IN_DIM, generator=g)
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(NUM_EDGES, EDGE_DIM, generator=g)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.rand(1, OUT_DIM, generator=g),
        level=torch.randint(0, 10, (NUM_NODES, 1), generator=g).float(),
        pi_paths=torch.rand(NUM_NODES, 1, generator=g),
    )


# ===========================================================================
# UnifiedGraphBaseModel Tests
# ===========================================================================


class TestUnifiedGraphBaseModel(unittest.TestCase):
    def test_forward_pass_no_pe(self):
        """Test base model with no positional encoding."""
        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM},
            pe_type="none",  # Required positional
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        model.eval()
        data = _make_aig_data()
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=None,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))

    def test_forward_pass_learned_level(self):
        """Test base model with discrete depth embedding."""
        model = UnifiedGraphBaseModel(
            encoder_name="gine",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM},
            pe_type="level",
            pos_enc_dim=PE_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        model.eval()
        data = _make_aig_data()

        # Apply PE Transform to simulate dataset extraction
        transform = get_pe_transform("level")
        data = transform(data)

        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=data.pos_enc,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))

    def test_forward_pass_continuous_pe(self):
        """Test base model with continuous positional encodings (pi_paths)."""
        model = UnifiedGraphBaseModel(
            encoder_name="vanilla_mpnn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM},
            pe_type="pi_paths",
            pos_enc_dim=PE_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        model.eval()
        data = _make_aig_data()

        transform = get_pe_transform("pi_paths")
        data = transform(data)

        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))
        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=data.pos_enc,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))

    def test_egin_encoder_bypass(self):
        """Test EGIN encoder bypasses the LazyLinear head."""
        model = UnifiedGraphBaseModel(
            encoder_name="egin",
            hidden_dim=HIDDEN_DIM,
            pe_type="none",
            encoder_kwargs={"num_layers": 3, "hid_dim": HIDDEN_DIM},
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        model.eval()
        data = _make_aig_data()
        out = model.forward_batch(data)
        # EGIN handles its own output dim; verify it reaches the task dim
        self.assertEqual(out.shape, (1, OUT_DIM))

    def test_gradient_flows(self):
        encoders = ["gine", "egin", "graphgps"]
        for enc in encoders:
            with self.subTest(encoder=enc):
                model = UnifiedGraphBaseModel(
                    encoder_name=enc,
                    hidden_dim=HIDDEN_DIM,
                    pe_type="level",
                    pos_enc_dim=PE_DIM,
                    encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM, "heads": 2},
                )
                data = get_pe_transform("level")(_make_aig_data())
                out = model.forward_batch(data)
                out.mean().backward()
                for name, p in model.named_parameters():
                    # EGIN does not use the global head, so skip it
                    if enc == "egin" and "head" in name:
                        continue
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, f"[{enc}] Broken graph at {name}")

    def test_batch_independence_for_all_encoders(self):
        """Ensures processing graphs in a batch yields identical results to processing them individually."""
        from torch_geometric.data import Batch

        encoders = [
            "gine",
            "gcn",
            "vanilla_mpnn",
            "egin",
            "transformer_conv",
            "graphgps",
        ]

        for encoder_name in encoders:
            with self.subTest(encoder_name=encoder_name):
                encoder_kwargs = {"num_layers": 2, "hid_dim": HIDDEN_DIM}
                if encoder_name in ["transformer_conv", "graphgps"]:
                    encoder_kwargs["heads"] = 2

                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    hidden_dim=HIDDEN_DIM,
                    encoder_kwargs=encoder_kwargs,
                    pe_type="none",  # Fixed missing pe_type
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                )
                model.eval()

                data1 = _make_aig_data(seed=1)
                data2 = _make_aig_data(seed=2)

                out1_alone = model.forward_batch(Batch.from_data_list([data1]))
                out2_alone = model.forward_batch(Batch.from_data_list([data2]))

                combined_batch = Batch.from_data_list([data1, data2])
                out_combined = model.forward_batch(combined_batch)

                self.assertTrue(
                    torch.allclose(out1_alone[0], out_combined[0], atol=1e-4)
                )
                self.assertTrue(
                    torch.allclose(out2_alone[0], out_combined[1], atol=1e-4)
                )


# ===========================================================================
# Lightning Module Training & Testing Loop
# ===========================================================================


class TestLightningModelTraining(unittest.TestCase):
    def setUp(self):
        """Create a synthetic dataset and DataLoaders of fake AIG data."""
        self.dataset = []
        for i in range(10):
            data = _make_aig_data(seed=i)
            transform = get_pe_transform("level")
            data = transform(data)
            self.dataset.append(data)

        self.train_loader = DataLoader(self.dataset[:6], batch_size=2)
        self.val_loader = DataLoader(self.dataset[6:8], batch_size=2)
        self.test_loader = DataLoader(self.dataset[8:], batch_size=2)

    def test_training_and_testing_loop(self):
        """Runs a mock training, validation, and testing loop using PyTorch Lightning."""
        # Fixed: Removed hid_dim and num_edge_types from direct args
        model = AIGRegressionLightningModule(
            encoder_name="gcn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM},
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="level",
            pos_enc_dim=PE_DIM,
            lr=1e-3,
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(
            model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader
        )
        trainer.test(model, dataloaders=self.test_loader)


class TestPositionalEncodingCompatibility(unittest.TestCase):
    def test_invalid_encoder_raises(self):
        # Must provide required encoder_kwargs and pe_type for UnifiedGraphBaseModel
        with self.assertRaises(KeyError):  # Registry lookup error
            UnifiedGraphBaseModel(
                encoder_name="nope", hidden_dim=8, encoder_kwargs={}, pe_type="none"
            )

    def test_get_batch_positional_encoding_present_and_absent(self):
        data = _make_aig_data(seed=0)
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader))

        self.assertIsNone(get_batch_positional_encoding(batch))

        pe = torch.randn(batch.x.size(0), 1)
        batch.pos_enc = pe
        out = get_batch_positional_encoding(batch)
        self.assertEqual(out.shape, (batch.x.size(0), 1))

    def test_forward_batch_and_forward_equivalence_with_pos_enc(self):
        data = _make_aig_data(seed=1)
        transform = get_pe_transform("pi_paths")
        data = transform(data)
        batch = next(iter(DataLoader([data], batch_size=1)))

        # Fixed: Passed required encoder_kwargs
        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": 32},
            pe_type="pi_paths",
            pos_enc_dim=PE_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        model.eval()

        out_via_forward_batch = model.forward_batch(batch)
        out_via_forward = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
            pos_enc=getattr(batch, "pos_enc", None),
        )

        self.assertTrue(
            torch.allclose(out_via_forward, out_via_forward_batch, atol=1e-6)
        )

    def test_lightning_training_step_accepts_pos_enc_and_returns_tensor(self):
        data = _make_aig_data(seed=2)
        transform = get_pe_transform("level")
        data = transform(data)
        batch = next(iter(DataLoader([data, data], batch_size=2)))

        # Fixed: Corrected keyword arguments
        lm = AIGRegressionLightningModule(
            encoder_name="gcn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": HIDDEN_DIM},
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="level",
            pos_enc_dim=PE_DIM,
            lr=1e-3,
        )

        loss = lm.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)


class TestEncoderKwargsPropagation(unittest.TestCase):
    def test_base_model_respects_non_egin_encoder_kwargs(self):

        test_cases = [
            ("gcn", "layers", 3, 20),
            ("gine", "layers", 4, 24),
            ("vanilla_mpnn", "convs", 2, 12),
        ]

        for encoder_name, layer_attr, num_layers, hid_dim in test_cases:
            with self.subTest(encoder=encoder_name):
                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    hidden_dim=HIDDEN_DIM,
                    encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
                    pe_type="none",
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                )
                self.assertEqual(model.encoder.num_layers, num_layers)

    def test_base_model_respects_egin_encoder_kwargs(self):
        model = UnifiedGraphBaseModel(
            encoder_name="egin",
            hidden_dim=HIDDEN_DIM,
            pe_type="none",
            encoder_kwargs={"num_layers": 5, "hid_dim": 18},
        )
        # Verify layer count in EGIN
        self.assertEqual(model.encoder.num_layers, 5)
        self.assertEqual(len(model.encoder.mlps), 4)

    def test_lightning_passes_encoder_kwargs_to_base_model(self):
        lm = AIGRegressionLightningModule(
            encoder_name="gcn",
            hidden_dim=HIDDEN_DIM,
            encoder_kwargs={"num_layers": 4, "hid_dim": 14},
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
        )
        self.assertEqual(lm.model.encoder.num_layers, 4)


if __name__ == "__main__":
    unittest.main()
