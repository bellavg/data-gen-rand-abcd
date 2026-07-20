from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from models.layers.positional_encodings import get_pos_enc_layer

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NUM_NODES = 8
NUM_EDGES = 12
IN_DIM = 4
EDGE_DIM = 2
HID_DIM = 16
NUM_LAYERS = 2
PE_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    num_nodes: int = NUM_NODES,
    in_dim: int = IN_DIM,
    edge_dim: int = EDGE_DIM,
    num_edges: int = NUM_EDGES,
    seed: int = 0,
):
    """Return (x, edge_index, edge_attr, batch) for a tiny synthetic graph."""
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, edge_dim, generator=g)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return x, edge_index, edge_attr, batch


def _make_data(**kwargs) -> Data:
    """Return a PyG Data object for transform tests with default dimensions."""
    x, edge_index, edge_attr, _ = _make_graph(**kwargs)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ==========================================================================
# Positional Encoding — transforms
# ==========================================================================


class TestExtractPrecomputedPE(unittest.TestCase):
    def test_discrete_casts_to_long(self):
        from models.layers.positional_encodings import ExtractPrecomputedPE

        data = _make_data()
        data.level = torch.randint(0, 20, (NUM_NODES, 1)).float()
        t = ExtractPrecomputedPE(source_key="level", attr_name="pos_enc", discrete=True)
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.long)

    def test_continuous_casts_to_float(self):
        from models.layers.positional_encodings import ExtractPrecomputedPE

        data = _make_data()
        data.pi_paths = torch.rand(NUM_NODES, 1)
        t = ExtractPrecomputedPE(
            source_key="pi_paths", attr_name="pos_enc", discrete=False
        )
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_missing_key_is_noop(self):
        from models.layers.positional_encodings import ExtractPrecomputedPE

        data = _make_data()
        t = ExtractPrecomputedPE(source_key="nonexistent_key", attr_name="pos_enc")
        out = t(data)
        self.assertIsNone(getattr(out, "pos_enc", None))

    def test_custom_attr_name(self):
        from models.layers.positional_encodings import ExtractPrecomputedPE

        data = _make_data()
        data.local_sp_sum = torch.rand(NUM_NODES, 1)
        t = ExtractPrecomputedPE(
            source_key="local_sp_sum", attr_name="my_pe", discrete=False
        )
        out = t(data)
        self.assertIsNotNone(getattr(out, "my_pe", None))
        self.assertEqual(out.my_pe.dtype, torch.float32)


class TestAddSinusoidalPE(unittest.TestCase):
    def test_output_shape(self):
        from models.layers.positional_encodings import AddSinusoidalPE

        data = _make_data()
        t = AddSinusoidalPE(dim=PE_DIM, attr_name="pos_enc")
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_custom_attr_name(self):
        from models.layers.positional_encodings import AddSinusoidalPE

        data = _make_data()
        t = AddSinusoidalPE(dim=PE_DIM, attr_name="sinusoidal_pe")
        out = t(data)
        self.assertEqual(out.sinusoidal_pe.shape, (NUM_NODES, PE_DIM))

    def test_different_dims(self):
        from models.layers.positional_encodings import AddSinusoidalPE

        data = _make_data()
        for dim in [4, 16, 32]:
            t = AddSinusoidalPE(dim=dim)
            out = t(data)
            self.assertEqual(out.pos_enc.shape, (NUM_NODES, dim))


# ==========================================================================
# Positional Encoding — learned modules
# ==========================================================================


class TestLearnedDepthEmbedding(unittest.TestCase):
    def setUp(self):
        from models.layers.positional_encodings import LearnedDepthEmbedding

        self.model = LearnedDepthEmbedding(max_depth=100, embed_dim=PE_DIM)

    def test_output_shape(self):
        depth = torch.randint(0, 50, (NUM_NODES, 1))
        out = self.model(depth)
        self.assertEqual(out.shape, (NUM_NODES, PE_DIM))

    def test_flat_input(self):
        depth = torch.randint(0, 50, (NUM_NODES,))
        out = self.model(depth)
        self.assertEqual(out.shape, (NUM_NODES, PE_DIM))

    def test_clamping_out_of_range(self):
        # Should not crash on out-of-range integer indices
        depth = torch.tensor([[200], [300], [-5]])
        out = self.model(depth)
        self.assertEqual(out.shape, (3, PE_DIM))

    def test_embed_dim_respected(self):
        from models.layers.positional_encodings import LearnedDepthEmbedding

        for dim in [4, 32]:
            m = LearnedDepthEmbedding(max_depth=50, embed_dim=dim)
            out = m(torch.zeros(5, 1, dtype=torch.long))
            self.assertEqual(out.shape[1], dim)


# ==========================================================================
# Positional Encoding — factory functions
# ==========================================================================


class TestGetPeTransform(unittest.TestCase):
    def test_none_is_identity(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        t = get_pe_transform(None)
        self.assertIs(t(data), data)

    def test_none_string_is_identity(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        t = get_pe_transform("none")
        self.assertIs(t(data), data)

    def test_level_discrete(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        data.level = torch.randint(0, 10, (NUM_NODES, 1)).float()
        t = get_pe_transform("level")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_learned_level_strips_prefix(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        data.level = torch.randint(0, 10, (NUM_NODES, 1)).float()
        t = get_pe_transform("learned_level")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_pi_paths_continuous(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        data.pi_paths = torch.rand(NUM_NODES, 1)
        t = get_pe_transform("pi_paths")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_local_sp_sum_continuous(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        data.local_sp_sum = torch.rand(NUM_NODES, 1)
        t = get_pe_transform("local_sp_sum")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_sinusoidal(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        t = get_pe_transform("sinusoidal", dim=PE_DIM)
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_sine_alias(self):
        from models.layers.positional_encodings import get_pe_transform

        data = _make_data()
        t = get_pe_transform("sine", dim=PE_DIM)
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_unknown_raises(self):
        from models.layers.positional_encodings import get_pe_transform

        with self.assertRaises(ValueError):
            get_pe_transform("unknown_pe")


class TestGetPosEncLayer(unittest.TestCase):
    def setUp(self):
        self.pe_dim = 16
        self.num_nodes = 10

    def test_pi_paths_sequential(self):
        layer = get_pos_enc_layer("pi_paths", pos_enc_dim=self.pe_dim)
        # We now expect a Sequential block containing Linear + LeakyReLU
        self.assertIsInstance(layer, nn.Sequential)
        self.assertIsInstance(layer[0], gnn.Linear)
        self.assertIsInstance(layer[1], nn.LeakyReLU)

        # Verify forward pass
        dummy_input = torch.rand(self.num_nodes, 1)
        out = layer(dummy_input)
        self.assertEqual(out.shape, (self.num_nodes, self.pe_dim))

    def test_local_sp_sum_sequential(self):
        layer = get_pos_enc_layer("local_sp_sum", pos_enc_dim=self.pe_dim)
        self.assertIsInstance(layer, nn.Sequential)
        self.assertIsInstance(layer[0], gnn.Linear)

    def test_sinusoidal_sequential(self):
        layer = get_pos_enc_layer("sinusoidal", pos_enc_dim=self.pe_dim)
        self.assertIsInstance(layer, nn.Sequential)
        self.assertIsInstance(layer[0], gnn.Linear)

    def test_level_embedding(self):
        layer = get_pos_enc_layer("level", pos_enc_dim=self.pe_dim)
        self.assertIsInstance(layer, nn.Sequential)
        self.assertIsInstance(layer[0], gnn.Linear)


# GINE Conv layers tests removed since GINE was deleted.


# ==========================================================================
# GCN layers
# ==========================================================================


class TestGCNConvLayer(unittest.TestCase):
    def setUp(self):
        from models.layers.gcn import GCNConvLayer

        self.layer = GCNConvLayer(
            dim_in=HID_DIM,
            dim_out=HID_DIM,
            edge_dim=EDGE_DIM,
            dropout=0.0,
            norm_type="batch",
        )
        self.layer.eval()

    def test_output_shape(self):
        _, edge_index, edge_attr, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_no_edge_attr(self):
        # GCN now always requires edge attributes (edge_attr passed through
        # propagate, not via instance variable).  Confirm a TypeError/similar
        # is raised when edge_attr is None rather than silently producing zeros.
        _, edge_index, _, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        with self.assertRaises(Exception):
            self.layer(x=x, edge_index=edge_index, edge_attr=None)

    def test_layer_norm_type(self):
        from models.layers.gcn import GCNConvLayer

        layer = GCNConvLayer(
            dim_in=HID_DIM,
            dim_out=HID_DIM,
            edge_dim=EDGE_DIM,
            dropout=0.0,
            norm_type="layer",
        )
        layer.eval()
        _, edge_index, edge_attr, _ = _make_graph()
        out = layer(
            x=torch.randn(NUM_NODES, HID_DIM),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))


class TestGCNEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from models.layers.gcn import GCNEncoder

        num_layers = kwargs.get("num_layers", NUM_LAYERS)
        defaults = dict(
            node_input_dim=HID_DIM,
            hid_dim=HID_DIM,
            num_layers=num_layers,
            edge_attr_dim=HID_DIM,
            output_dim=HID_DIM * (num_layers + 1),
        )
        defaults.update(kwargs)
        return GCNEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph(in_dim=HID_DIM, edge_dim=HID_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))


# Unused MPNN, TransformerConv, and GraphGPS layer tests removed.


# ==========================================================================
# model_utils — get_norm_layer & apply_norm
# ==========================================================================


class TestGetNormLayer(unittest.TestCase):
    def test_none_returns_identity(self):
        from models.model_utils import get_norm_layer

        layer = get_norm_layer(None, HID_DIM)
        self.assertIsInstance(layer, nn.Identity)

    def test_none_string_returns_identity(self):
        from models.model_utils import get_norm_layer

        layer = get_norm_layer("none", HID_DIM)
        self.assertIsInstance(layer, nn.Identity)

    def test_batch_norm(self):
        from torch_geometric.nn import BatchNorm

        from models.model_utils import get_norm_layer

        layer = get_norm_layer("batch", HID_DIM)
        self.assertIsInstance(layer, BatchNorm)

    def test_layer_norm(self):
        from torch_geometric.nn import LayerNorm

        from models.model_utils import get_norm_layer

        layer = get_norm_layer("layer", HID_DIM)
        self.assertIsInstance(layer, LayerNorm)

    def test_graph_norm(self):
        from torch_geometric.nn import GraphNorm

        from models.model_utils import get_norm_layer

        layer = get_norm_layer("graph", HID_DIM)
        self.assertIsInstance(layer, GraphNorm)
        # Aliases
        self.assertIsInstance(get_norm_layer("graphnorm", HID_DIM), GraphNorm)
        self.assertIsInstance(get_norm_layer("gn", HID_DIM), GraphNorm)

    def test_instance_norm(self):
        from torch_geometric.nn import InstanceNorm

        from models.model_utils import get_norm_layer

        layer = get_norm_layer("instance", HID_DIM)
        self.assertIsInstance(layer, InstanceNorm)
        # Aliases
        self.assertIsInstance(get_norm_layer("instancenorm", HID_DIM), InstanceNorm)
        self.assertIsInstance(get_norm_layer("in", HID_DIM), InstanceNorm)

    def test_unknown_raises_value_error(self):
        from models.model_utils import get_norm_layer

        with self.assertRaises(ValueError):
            get_norm_layer("unknown_norm", HID_DIM)


class TestApplyNorm(unittest.TestCase):
    def test_identity_no_batch(self):
        from models.model_utils import apply_norm

        x = torch.randn(NUM_NODES, HID_DIM)
        out = apply_norm(nn.Identity(), x)
        self.assertTrue(torch.equal(out, x))

    def test_batch_norm_no_batch_arg(self):
        from torch_geometric.nn import BatchNorm

        from models.model_utils import apply_norm

        norm = BatchNorm(HID_DIM)
        norm.eval()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = apply_norm(norm, x, batch=None)
        self.assertEqual(out.shape, x.shape)

    def test_layer_norm_receives_batch_arg(self):
        from torch_geometric.nn import LayerNorm

        from models.model_utils import apply_norm

        norm = LayerNorm(HID_DIM)
        norm.eval()
        x = torch.randn(NUM_NODES, HID_DIM)
        batch = torch.zeros(NUM_NODES, dtype=torch.long)
        out = apply_norm(norm, x, batch=batch)
        self.assertEqual(out.shape, x.shape)

    def test_graph_norm_receives_batch_arg(self):
        from torch_geometric.nn import GraphNorm

        from models.model_utils import apply_norm

        norm = GraphNorm(HID_DIM)
        norm.eval()
        x = torch.randn(NUM_NODES, HID_DIM)
        batch = torch.zeros(NUM_NODES, dtype=torch.long)
        out = apply_norm(norm, x, batch=batch)
        self.assertEqual(out.shape, x.shape)


# ==========================================================================
# constants — get_output_dim_for_encoder
# ==========================================================================


class TestGetOutputDimForEncoder(unittest.TestCase):
    def test_jk_cat_multiplies_layers(self):
        from config import get_output_dim_for_encoder

        # cat mode: hid_dim * (num_layers + 1)
        result = get_output_dim_for_encoder(
            "gcn", {"hid_dim": 32, "num_layers": 3, "jk_mode": "cat"}
        )
        self.assertEqual(result, 32 * 4)

    def test_jk_last_returns_hid_dim(self):
        from config import get_output_dim_for_encoder

        result = get_output_dim_for_encoder(
            "gcn", {"hid_dim": 64, "num_layers": 3, "jk_mode": "last"}
        )
        self.assertEqual(result, 64)

    def test_jk_mean_returns_hid_dim(self):
        from config import get_output_dim_for_encoder

        result = get_output_dim_for_encoder(
            "gcn", {"hid_dim": 128, "num_layers": 2, "jk_mode": "mean"}
        )
        self.assertEqual(result, 128)

    def test_default_jk_mode_is_cat(self):
        """No jk_mode key defaults to 'cat'."""
        from config import get_output_dim_for_encoder

        result = get_output_dim_for_encoder("gcn", {"hid_dim": 16, "num_layers": 2})
        self.assertEqual(result, 16 * 3)
