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
EMBED_DIM = 16
OUT_DIM = 1  # Single-target regression [Node Opt]
PE_DIM = 8


def _make_aig_data(
    num_nodes: int = NUM_NODES,
    in_dim: int = IN_DIM,
    edge_dim: int = EDGE_DIM,
    num_edges: int = NUM_EDGES,
    seed: int = 42,
) -> Data:
    """
    Return a PyG Data object simulating a synthetic AIG graph.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # Base Graph features
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, edge_dim, generator=g)

    # Targets for single-target regression (Node Opt)
    y = torch.rand(1, OUT_DIM, generator=g)

    # Pre-computed Positional Encodings
    level = torch.randint(0, 10, (num_nodes, 1), generator=g).float()
    pi_paths = torch.rand(num_nodes, 1, generator=g)
    local_sp_sum = torch.rand(num_nodes, 1, generator=g)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        level=level,
        pi_paths=pi_paths,
        local_sp_sum=local_sp_sum,
    )


# ===========================================================================
# UnifiedGraphBaseModel Tests
# ===========================================================================


class TestUnifiedGraphBaseModel(unittest.TestCase):
    def test_forward_pass_no_pe(self):
        """Test base model with no positional encoding."""
        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="none",
            encoder_kwargs={"num_layers": 2, "hid_dim": EMBED_DIM},
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
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="level",
            pos_enc_dim=PE_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": EMBED_DIM},
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
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="pi_paths",
            pos_enc_dim=PE_DIM,
            encoder_kwargs={"num_layers": 2, "hid_dim": EMBED_DIM},
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
        """Test EGIN encoder, which handles task_out_dim natively and uses nn.Identity for the head."""
        model = UnifiedGraphBaseModel(
            encoder_name="egin",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": 3, "hid_dim": EMBED_DIM},
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
        self.assertIsInstance(model.head, torch.nn.Identity)

    def test_gradient_flows_through_all_parameters(self):
        """Ensures the computational graph is fully connected from loss to inputs for ALL encoders."""
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
                # 1. Provide necessary kwargs for attention-based models
                encoder_kwargs = {"num_layers": 2, "hid_dim": EMBED_DIM}
                if encoder_name in ["transformer_conv", "graphgps"]:
                    encoder_kwargs["heads"] = 2

                # 2. Instantiate Model
                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    embed_dim=EMBED_DIM,
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                    pe_type="level",
                    pos_enc_dim=PE_DIM,
                    encoder_kwargs=encoder_kwargs,
                )
                model.train()

                # 3. Create Data
                data = _make_aig_data()
                transform = get_pe_transform("level")
                data = transform(data)
                batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

                # 4. Forward Pass
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=batch.batch,
                    edge_attr=data.edge_attr,
                    pos_enc=data.pos_enc,
                )

                # 5. Fake Loss and Backward Pass
                loss = out.mean()
                loss.backward()

                # 6. Verify Gradients
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertIsNotNone(
                            param.grad,
                            f"[{encoder_name}] Parameter '{name}' has no gradient. Computational graph broken.",
                        )
                        self.assertFalse(
                            torch.all(param.grad == 0),
                            f"[{encoder_name}] Parameter '{name}' has zero gradient. It is not learning.",
                        )

    def test_batch_independence_for_all_encoders(self):
        """Ensures processing graphs in a batch yields identical results to processing them individually for ALL encoders."""
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
                # 1. Provide necessary kwargs for attention-based models
                encoder_kwargs = {"num_layers": 2, "hid_dim": EMBED_DIM}
                if encoder_name in ["transformer_conv", "graphgps"]:
                    encoder_kwargs["heads"] = 2

                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    embed_dim=EMBED_DIM,
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                    encoder_kwargs=encoder_kwargs,
                )

                # CRITICAL: Must be in eval() mode. If in train() mode, Batch Normalization
                # layers will calculate statistics across the whole combined batch and deliberately
                # change the outputs, causing this test to fail.
                model.eval()

                # Create two distinct graphs
                data1 = _make_aig_data(seed=1)
                data2 = _make_aig_data(seed=2)

                # 1. Forward individually
                out1_alone = model.forward_batch(Batch.from_data_list([data1]))
                out2_alone = model.forward_batch(Batch.from_data_list([data2]))

                # 2. Forward together as a batch
                combined_batch = Batch.from_data_list([data1, data2])
                out_combined = model.forward_batch(combined_batch)

                # 3. Assert exact match
                # Using 1e-4 / 1e-5 atol because floating point math order of operations
                # changes slightly during batched matrix multiplication.
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
            # Create data and attach the discrete 'level' positional encoding
            data = _make_aig_data(seed=i)
            transform = get_pe_transform("level")
            data = transform(data)
            self.dataset.append(data)

        self.train_loader = DataLoader(self.dataset[:6], batch_size=2)
        self.val_loader = DataLoader(self.dataset[6:8], batch_size=2)
        self.test_loader = DataLoader(self.dataset[8:], batch_size=2)

    def test_training_and_testing_loop(self):
        """
        Runs a mock training, validation, and testing loop using PyTorch Lightning.
        Ensures loss calculation and metric logging (Huber, MAE Node) executes without errors.
        """
        # Initialize Lightning wrapper
        model = AIGRegressionLightningModule(
            encoder_name="gcn",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            num_edge_types=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="level",
            pos_enc_dim=PE_DIM,
            lr=1e-3,
            hid_dim=EMBED_DIM,
        )

        # fast_dev_run=True will run exactly 1 batch of train, val, and test
        # to ensure the loop works and prevents large CI test times.
        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )

        # Run Train/Val
        try:
            trainer.fit(
                model,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            fit_success = True
        except Exception as e:
            fit_success = False
            self.fail(f"Trainer.fit() failed with exception: {e}")

        self.assertTrue(fit_success)

        # Run Test
        try:
            trainer.test(model, dataloaders=self.test_loader)
            test_success = True
        except Exception as e:
            test_success = False
            self.fail(f"Trainer.test() failed with exception: {e}")

        self.assertTrue(test_success)


class TestPositionalEncodingCompatibility(unittest.TestCase):
    def test_invalid_encoder_raises(self):
        with self.assertRaises(ValueError):
            UnifiedGraphBaseModel(encoder_name="nope", embed_dim=8)

    def test_get_batch_positional_encoding_present_and_absent(self):
        data = _make_aig_data(seed=0)
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader))

        # No pos_enc by default
        self.assertIsNone(get_batch_positional_encoding(batch))

        # Attach a pos_enc and verify it's returned and shaped correctly
        pe = torch.randn(batch.x.size(0), 1)
        batch.pos_enc = pe
        out = get_batch_positional_encoding(batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (batch.x.size(0), 1))

    def test_forward_batch_and_forward_equivalence_with_pos_enc(self):
        data = _make_aig_data(seed=1)
        transform = get_pe_transform("pi_paths")
        data = transform(data)

        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader))

        model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="pi_paths",
            pos_enc_dim=PE_DIM,
            hid_dim=32,
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

        self.assertEqual(out_via_forward.shape, out_via_forward_batch.shape)
        self.assertTrue(
            torch.allclose(out_via_forward, out_via_forward_batch, atol=1e-6)
        )

    def test_lightning_training_step_accepts_pos_enc_and_returns_tensor(self):
        data = _make_aig_data(seed=2)
        transform = get_pe_transform("level")
        data = transform(data)

        loader = DataLoader([data, data], batch_size=2)
        batch = next(iter(loader))

        lm = AIGRegressionLightningModule(
            encoder_name="gcn",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            num_edge_types=EDGE_DIM,
            task_out_dim=OUT_DIM,
            pe_type="level",
            pos_enc_dim=PE_DIM,
            num_layers=2,
            lr=1e-3,
            hid_dim=EMBED_DIM,
        )

        loss = lm.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)


class TestEncoderKwargsPropagation(unittest.TestCase):
    def test_base_model_respects_non_egin_encoder_kwargs(self):
        data = _make_aig_data(seed=7)
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        test_cases = [
            ("gcn", "layers", 3, 20),
            ("gine", "layers", 4, 24),
            ("vanilla_mpnn", "convs", 2, 12),
        ]

        for encoder_name, layer_attr, num_layers, hid_dim in test_cases:
            with self.subTest(
                encoder=encoder_name, num_layers=num_layers, hid_dim=hid_dim
            ):
                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    embed_dim=EMBED_DIM,
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                    encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
                )
                model.eval()

                self.assertEqual(model.encoder.num_layers, num_layers)
                self.assertEqual(len(getattr(model.encoder, layer_attr)), num_layers)

                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=batch.batch,
                    edge_attr=data.edge_attr,
                    pos_enc=None,
                )
                self.assertEqual(out.shape, (1, OUT_DIM))
                self.assertEqual(model.head.in_features, hid_dim * (num_layers + 1))

    def test_base_model_respects_egin_encoder_kwargs(self):
        data = _make_aig_data(seed=8)
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        num_layers = 5
        hid_dim = 18

        model = UnifiedGraphBaseModel(
            encoder_name="egin",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
        )
        model.eval()

        self.assertEqual(model.encoder.num_layers, num_layers)
        self.assertEqual(len(model.encoder.mlps), num_layers - 1)
        self.assertEqual(len(model.encoder.linears_prediction), num_layers)
        self.assertEqual(model.encoder.edge_attr_dim, EMBED_DIM)
        self.assertIsInstance(model.head, torch.nn.Identity)

        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=None,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))

    def test_lightning_passes_encoder_kwargs_to_base_model(self):
        num_layers = 4
        hid_dim = 14

        lm = AIGRegressionLightningModule(
            encoder_name="gcn",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            num_edge_types=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
            hid_dim=hid_dim,
        )

        self.assertEqual(lm.model.encoder.num_layers, num_layers)

    def test_base_model_respects_transformer_conv_kwargs(self):
        data = _make_aig_data(seed=9)
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        num_layers = 3
        hid_dim = 10
        model = UnifiedGraphBaseModel(
            encoder_name="transformer_conv",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim, "heads": 2},
        )
        model.eval()

        self.assertEqual(model.encoder.num_layers, num_layers)
        self.assertEqual(len(model.encoder.layers), num_layers)

        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=None,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))
        self.assertEqual(model.head.in_features, hid_dim * (num_layers + 1))

    def test_base_model_respects_graphgps_kwargs(self):
        data = _make_aig_data(seed=10)
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        num_layers = 2
        hid_dim = 12
        model = UnifiedGraphBaseModel(
            encoder_name="graphgps",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            edge_attr_dim=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
        )
        model.eval()

        self.assertEqual(model.encoder.num_layers, num_layers)
        self.assertEqual(len(model.encoder.layers), num_layers)

        out = model(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch.batch,
            edge_attr=data.edge_attr,
            pos_enc=None,
        )
        self.assertEqual(out.shape, (1, OUT_DIM))
        self.assertEqual(model.head.in_features, hid_dim * (num_layers + 1))

    def test_lightning_passes_graphgps_kwargs_to_base_model(self):
        num_layers = 2
        hid_dim = 12
        lm = AIGRegressionLightningModule(
            encoder_name="graphgps",
            embed_dim=EMBED_DIM,
            node_input_dim=IN_DIM,
            num_edge_types=EDGE_DIM,
            task_out_dim=OUT_DIM,
            encoder_kwargs={"num_layers": num_layers, "hid_dim": hid_dim},
            hid_dim=hid_dim,
        )

        self.assertEqual(lm.model.encoder.num_layers, num_layers)
        self.assertEqual(len(lm.model.encoder.layers), num_layers)

    def test_lightning_example_input_runs_for_edge_aware_encoders(self):
        for encoder_name in ["transformer_conv", "graphgps"]:
            with self.subTest(encoder_name=encoder_name):
                lm = AIGRegressionLightningModule(
                    encoder_name=encoder_name,
                    embed_dim=EMBED_DIM,
                    node_input_dim=IN_DIM,
                    num_edge_types=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                    encoder_kwargs={"num_layers": 2, "hid_dim": 12, "heads": 2},
                    hid_dim=12,
                )
                self.assertTrue(hasattr(lm.example_input_array, "edge_attr"))
                self.assertIsNotNone(lm.example_input_array.edge_attr)
                out = lm.forward(lm.example_input_array)
                self.assertEqual(out.shape[-1], OUT_DIM)

    def test_edge_aware_encoders_handle_missing_edge_attr(self):
        data = _make_aig_data(seed=11)
        batch = Data(batch=torch.zeros(data.num_nodes, dtype=torch.long))

        for encoder_name in ["gine", "transformer_conv", "graphgps", "egin"]:
            with self.subTest(encoder_name=encoder_name):
                kwargs = {"num_layers": 2, "hid_dim": 12}
                if encoder_name in {"transformer_conv", "graphgps"}:
                    kwargs["heads"] = 2
                if encoder_name == "egin":
                    kwargs["num_layers"] = 3

                model = UnifiedGraphBaseModel(
                    encoder_name=encoder_name,
                    embed_dim=EMBED_DIM,
                    node_input_dim=IN_DIM,
                    edge_attr_dim=EDGE_DIM,
                    task_out_dim=OUT_DIM,
                    encoder_kwargs=kwargs,
                )
                model.eval()

                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=batch.batch,
                    edge_attr=None,
                    pos_enc=None,
                )
                self.assertEqual(out.shape, (1, OUT_DIM))


if __name__ == "__main__":
    unittest.main()
