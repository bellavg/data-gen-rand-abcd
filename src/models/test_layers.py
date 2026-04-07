from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

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


# ===========================================================================
# Positional Encoding — transforms
# ===========================================================================


class TestExtractPrecomputedPE(unittest.TestCase):
    def test_discrete_casts_to_long(self):
        from src.models.layers.positional_encodigs import ExtractPrecomputedPE

        data = _make_data()
        data.level = torch.randint(0, 20, (NUM_NODES, 1)).float()
        t = ExtractPrecomputedPE(source_key="level", attr_name="pos_enc", discrete=True)
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.long)

    def test_continuous_casts_to_float(self):
        from src.models.layers.positional_encodigs import ExtractPrecomputedPE

        data = _make_data()
        data.pi_paths = torch.rand(NUM_NODES, 1)
        t = ExtractPrecomputedPE(source_key="pi_paths", attr_name="pos_enc", discrete=False)
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_missing_key_is_noop(self):
        from src.models.layers.positional_encodigs import ExtractPrecomputedPE

        data = _make_data()
        t = ExtractPrecomputedPE(source_key="nonexistent_key", attr_name="pos_enc")
        out = t(data)
        self.assertIsNone(getattr(out, "pos_enc", None))

    def test_custom_attr_name(self):
        from src.models.layers.positional_encodigs import ExtractPrecomputedPE

        data = _make_data()
        data.local_sp_sum = torch.rand(NUM_NODES, 1)
        t = ExtractPrecomputedPE(source_key="local_sp_sum", attr_name="my_pe", discrete=False)
        out = t(data)
        self.assertIsNotNone(getattr(out, "my_pe", None))
        self.assertEqual(out.my_pe.dtype, torch.float32)


class TestAddSinusoidalPE(unittest.TestCase):
    def test_output_shape(self):
        from src.models.layers.positional_encodigs import AddSinusoidalPE

        data = _make_data()
        t = AddSinusoidalPE(dim=PE_DIM, attr_name="pos_enc")
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_custom_attr_name(self):
        from src.models.layers.positional_encodigs import AddSinusoidalPE

        data = _make_data()
        t = AddSinusoidalPE(dim=PE_DIM, attr_name="sinusoidal_pe")
        out = t(data)
        self.assertEqual(out.sinusoidal_pe.shape, (NUM_NODES, PE_DIM))

    def test_different_dims(self):
        from src.models.layers.positional_encodigs import AddSinusoidalPE

        data = _make_data()
        for dim in [4, 16, 32]:
            t = AddSinusoidalPE(dim=dim)
            out = t(data)
            self.assertEqual(out.pos_enc.shape, (NUM_NODES, dim))


# ===========================================================================
# Positional Encoding — learned modules
# ===========================================================================


class TestLearnedDepthEmbedding(unittest.TestCase):
    def setUp(self):
        from src.models.layers.positional_encodigs import LearnedDepthEmbedding

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
        from src.models.layers.positional_encodigs import LearnedDepthEmbedding

        for dim in [4, 32]:
            m = LearnedDepthEmbedding(max_depth=50, embed_dim=dim)
            out = m(torch.zeros(5, 1, dtype=torch.long))
            self.assertEqual(out.shape[1], dim)


class TestLearnedRelativeDistanceEmbedding(unittest.TestCase):
    def setUp(self):
        from src.models.layers.positional_encodigs import LearnedRelativeDistanceEmbedding

        self.model = LearnedRelativeDistanceEmbedding(max_hops=10, embed_dim=PE_DIM)

    def test_output_shape(self):
        dists = torch.randint(0, 5, (NUM_EDGES, 1))
        out = self.model(dists)
        self.assertEqual(out.shape, (NUM_EDGES, PE_DIM))

    def test_flat_input(self):
        dists = torch.randint(0, 5, (NUM_EDGES,))
        out = self.model(dists)
        self.assertEqual(out.shape, (NUM_EDGES, PE_DIM))

    def test_clamping_exceeds_max_hops(self):
        dists = torch.tensor([[100], [200]])
        out = self.model(dists)
        self.assertEqual(out.shape, (2, PE_DIM))

    def test_embed_dim_respected(self):
        from src.models.layers.positional_encodigs import LearnedRelativeDistanceEmbedding

        for dim in [4, 32]:
            m = LearnedRelativeDistanceEmbedding(max_hops=5, embed_dim=dim)
            out = m(torch.zeros(3, 1, dtype=torch.long))
            self.assertEqual(out.shape[1], dim)


# ===========================================================================
# Positional Encoding — factory functions
# ===========================================================================


class TestGetPeTransform(unittest.TestCase):
    def test_none_is_identity(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        t = get_pe_transform(None)
        self.assertIs(t(data), data)

    def test_none_string_is_identity(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        t = get_pe_transform("none")
        self.assertIs(t(data), data)

    def test_level_discrete(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        data.level = torch.randint(0, 10, (NUM_NODES, 1)).float()
        t = get_pe_transform("level")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.long)

    def test_learned_level_strips_prefix(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        data.level = torch.randint(0, 10, (NUM_NODES, 1)).float()
        t = get_pe_transform("learned_level")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.long)

    def test_pi_paths_continuous(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        data.pi_paths = torch.rand(NUM_NODES, 1)
        t = get_pe_transform("pi_paths")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_local_sp_sum_continuous(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        data.local_sp_sum = torch.rand(NUM_NODES, 1)
        t = get_pe_transform("local_sp_sum")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.float32)

    def test_edge_rel_dist_discrete(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        data.edge_rel_dist = torch.randint(0, 5, (NUM_EDGES, 1)).float()
        t = get_pe_transform("edge_rel_dist")
        out = t(data)
        self.assertEqual(out.pos_enc.dtype, torch.long)

    def test_sinusoidal(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        t = get_pe_transform("sinusoidal", dim=PE_DIM)
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_sine_alias(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        data = _make_data()
        t = get_pe_transform("sine", dim=PE_DIM)
        out = t(data)
        self.assertEqual(out.pos_enc.shape, (NUM_NODES, PE_DIM))

    def test_unknown_raises(self):
        from src.models.layers.positional_encodigs import get_pe_transform

        with self.assertRaises(ValueError):
            get_pe_transform("unknown_pe")


class TestGetPosEncLayer(unittest.TestCase):
    def test_none_returns_identity(self):
        import torch.nn as nn

        from src.models.layers.positional_encodigs import get_pos_enc_layer

        layer = get_pos_enc_layer(None)
        self.assertIsInstance(layer, nn.Identity)

    def test_none_string_returns_identity(self):
        import torch.nn as nn

        from src.models.layers.positional_encodigs import get_pos_enc_layer

        layer = get_pos_enc_layer("none")
        self.assertIsInstance(layer, nn.Identity)

    def test_learned_level(self):
        from src.models.layers.positional_encodigs import (
            LearnedDepthEmbedding,
            get_pos_enc_layer,
        )

        layer = get_pos_enc_layer("learned_level", pos_enc_dim=PE_DIM, max_depth=100)
        self.assertIsInstance(layer, LearnedDepthEmbedding)
        out = layer(torch.randint(0, 50, (NUM_NODES, 1)))
        self.assertEqual(out.shape, (NUM_NODES, PE_DIM))

    def test_level_alias(self):
        from src.models.layers.positional_encodigs import (
            LearnedDepthEmbedding,
            get_pos_enc_layer,
        )

        layer = get_pos_enc_layer("level", pos_enc_dim=PE_DIM, max_depth=100)
        self.assertIsInstance(layer, LearnedDepthEmbedding)

    def test_learned_edge_rel_dist(self):
        from src.models.layers.positional_encodigs import (
            LearnedRelativeDistanceEmbedding,
            get_pos_enc_layer,
        )

        layer = get_pos_enc_layer("learned_edge_rel_dist", pos_enc_dim=PE_DIM, max_hops=10)
        self.assertIsInstance(layer, LearnedRelativeDistanceEmbedding)
        out = layer(torch.randint(0, 5, (NUM_EDGES, 1)))
        self.assertEqual(out.shape, (NUM_EDGES, PE_DIM))

    def test_pi_paths_linear(self):
        import torch.nn as nn

        from src.models.layers.positional_encodigs import get_pos_enc_layer

        layer = get_pos_enc_layer("pi_paths", pos_enc_dim=PE_DIM)
        self.assertIsInstance(layer, nn.Linear)
        out = layer(torch.rand(NUM_NODES, 1))
        self.assertEqual(out.shape, (NUM_NODES, PE_DIM))

    def test_local_sp_sum_linear(self):
        import torch.nn as nn

        from src.models.layers.positional_encodigs import get_pos_enc_layer

        layer = get_pos_enc_layer("local_sp_sum", pos_enc_dim=PE_DIM)
        self.assertIsInstance(layer, nn.Linear)

    def test_sinusoidal_linear(self):
        import torch.nn as nn

        from src.models.layers.positional_encodigs import get_pos_enc_layer

        layer = get_pos_enc_layer("sinusoidal", pos_enc_dim=PE_DIM)
        self.assertIsInstance(layer, nn.Linear)

    def test_unknown_raises(self):
        from src.models.layers.positional_encodigs import get_pos_enc_layer

        with self.assertRaises(ValueError):
            get_pos_enc_layer("unknown_pe")


# ===========================================================================
# GINE layers
# ===========================================================================


class TestGINEConvLayer(unittest.TestCase):
    def setUp(self):
        from src.models.layers.gine import GINEConvLayer

        self.layer = GINEConvLayer(
            hid_dim=HID_DIM, edge_dim=EDGE_DIM, dropout=0.0, norm_type="batch"
        )
        self.layer.eval()

    def test_output_shape(self):
        _, edge_index, edge_attr, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_output_dtype_float(self):
        _, edge_index, edge_attr, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.dtype, torch.float32)

    def test_layer_norm_type(self):
        from src.models.layers.gine import GINEConvLayer

        layer = GINEConvLayer(hid_dim=HID_DIM, edge_dim=EDGE_DIM, dropout=0.0, norm_type="layer")
        layer.eval()
        _, edge_index, edge_attr, _ = _make_graph()
        out = layer(x=torch.randn(NUM_NODES, HID_DIM), edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))


class TestGINEEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from src.models.layers.gine import GINEEncoder

        defaults = dict(in_dim=IN_DIM, hid_dim=HID_DIM, num_layers=NUM_LAYERS, edge_dim=EDGE_DIM)
        defaults.update(kwargs)
        return GINEEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_out_dim_attribute(self):
        enc = self._make_enc()
        self.assertEqual(enc.out_dim, HID_DIM * (NUM_LAYERS + 1))

    def test_with_pos_enc(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_wrong_pos_enc_dim_raises(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_single_layer(self):
        enc = self._make_enc(num_layers=1)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * 2))

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(ValueError):
            self._make_enc(num_layers=0)

    def test_edge_type_arg_ignored(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))


# ===========================================================================
# GCN layers
# ===========================================================================


class TestGCNConvLayer(unittest.TestCase):
    def setUp(self):
        from src.models.layers.gcn import GCNConvLayer

        self.layer = GCNConvLayer(
            dim_in=HID_DIM, dim_out=HID_DIM, edge_dim=EDGE_DIM, dropout=0.0, norm_type="batch"
        )
        self.layer.eval()

    def test_output_shape(self):
        _, edge_index, edge_attr, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_no_edge_attr(self):
        _, edge_index, _, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=None)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_layer_norm_type(self):
        from src.models.layers.gcn import GCNConvLayer

        layer = GCNConvLayer(
            dim_in=HID_DIM, dim_out=HID_DIM, edge_dim=EDGE_DIM, dropout=0.0, norm_type="layer"
        )
        layer.eval()
        _, edge_index, edge_attr, _ = _make_graph()
        out = layer(x=torch.randn(NUM_NODES, HID_DIM), edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))


class TestGCNEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from src.models.layers.gcn import GCNEncoder

        defaults = dict(in_dim=IN_DIM, hid_dim=HID_DIM, num_layers=NUM_LAYERS, edge_dim=EDGE_DIM)
        defaults.update(kwargs)
        return GCNEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_out_dim_attribute(self):
        enc = self._make_enc()
        self.assertEqual(enc.out_dim, HID_DIM * (NUM_LAYERS + 1))

    def test_with_pos_enc(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_wrong_pos_enc_dim_raises(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_single_layer(self):
        enc = self._make_enc(num_layers=1)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * 2))

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(ValueError):
            self._make_enc(num_layers=0)

    def test_no_edge_attr(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, _, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=None)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))


# ===========================================================================
# TransformerConv layers
# ===========================================================================


class TestTransformerConvLayer(unittest.TestCase):
    def setUp(self):
        from src.models.layers.transformer_conv import TransformerConvLayer

        self.layer = TransformerConvLayer(
            hid_dim=HID_DIM, edge_dim=EDGE_DIM, heads=4, concat=False, norm_type="batch"
        )
        self.layer.eval()

    def test_output_shape(self):
        _, edge_index, edge_attr, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_output_dtype_float(self):
        _, edge_index, edge_attr, _ = _make_graph()
        out = self.layer(x=torch.randn(NUM_NODES, HID_DIM), edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.dtype, torch.float32)

    def test_concat_heads_preserves_dim(self):
        from src.models.layers.transformer_conv import TransformerConvLayer

        # HID_DIM=16 divisible by heads=4
        layer = TransformerConvLayer(
            hid_dim=HID_DIM, edge_dim=EDGE_DIM, heads=4, concat=True, norm_type="batch"
        )
        layer.eval()
        _, edge_index, edge_attr, _ = _make_graph()
        out = layer(x=torch.randn(NUM_NODES, HID_DIM), edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_indivisible_hid_dim_concat_raises(self):
        from src.models.layers.transformer_conv import TransformerConvLayer

        with self.assertRaises(ValueError):
            TransformerConvLayer(hid_dim=9, heads=4, concat=True)

    def test_layer_norm_type(self):
        from src.models.layers.transformer_conv import TransformerConvLayer

        layer = TransformerConvLayer(
            hid_dim=HID_DIM, edge_dim=EDGE_DIM, heads=4, concat=False, norm_type="layer"
        )
        layer.eval()
        _, edge_index, edge_attr, _ = _make_graph()
        out = layer(x=torch.randn(NUM_NODES, HID_DIM), edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))


class TestTransformerConvEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from src.models.layers.transformer_conv import TransformerConvEncoder

        defaults = dict(in_dim=IN_DIM, hid_dim=HID_DIM, num_layers=NUM_LAYERS, edge_dim=EDGE_DIM)
        defaults.update(kwargs)
        return TransformerConvEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_out_dim_attribute(self):
        enc = self._make_enc()
        self.assertEqual(enc.out_dim, HID_DIM * (NUM_LAYERS + 1))

    def test_with_pos_enc(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_wrong_pos_enc_dim_raises(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_single_layer(self):
        enc = self._make_enc(num_layers=1)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * 2))

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(ValueError):
            self._make_enc(num_layers=0)

    def test_edge_type_arg_ignored(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))


# ===========================================================================
# Vanilla MPNN layers
# ===========================================================================


class TestVanillaMPNNConv(unittest.TestCase):
    def setUp(self):
        from src.models.layers.vanilla_mpnn import VanillaMPNNConv

        # edge_attr passed to VanillaMPNNConv must already be HID_DIM (projected by MPNNEncoder)
        self.layer = VanillaMPNNConv(hid_dim=HID_DIM, norm_type="batch")
        self.layer.eval()

    def test_output_shape(self):
        _, edge_index, _, _ = _make_graph()
        x = torch.randn(NUM_NODES, HID_DIM)
        edge_attr = torch.randn(NUM_EDGES, HID_DIM)
        out = self.layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_output_dtype_float(self):
        _, edge_index, _, _ = _make_graph()
        out = self.layer(
            x=torch.randn(NUM_NODES, HID_DIM),
            edge_index=edge_index,
            edge_attr=torch.randn(NUM_EDGES, HID_DIM),
        )
        self.assertEqual(out.dtype, torch.float32)

    def test_layer_norm_type(self):
        from src.models.layers.vanilla_mpnn import VanillaMPNNConv

        layer = VanillaMPNNConv(hid_dim=HID_DIM, norm_type="layer")
        layer.eval()
        _, edge_index, _, _ = _make_graph()
        out = layer(
            x=torch.randn(NUM_NODES, HID_DIM),
            edge_index=edge_index,
            edge_attr=torch.randn(NUM_EDGES, HID_DIM),
        )
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))


class TestMPNNEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from src.models.layers.vanilla_mpnn import MPNNEncoder

        defaults = dict(in_dim=IN_DIM, hid_dim=HID_DIM, num_layers=NUM_LAYERS, edge_dim=EDGE_DIM)
        defaults.update(kwargs)
        return MPNNEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_out_dim_attribute(self):
        enc = self._make_enc()
        self.assertEqual(enc.out_dim, HID_DIM * (NUM_LAYERS + 1))

    def test_no_edge_attr_raises(self):
        enc = self._make_enc()
        x, edge_index, _, batch = _make_graph()
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=None)

    def test_with_pos_enc(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_wrong_pos_enc_dim_raises(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_single_layer(self):
        enc = self._make_enc(num_layers=1)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * 2))


# ===========================================================================
# EGIN layers
# ===========================================================================


class TestEGINMLP(unittest.TestCase):
    def test_single_layer_output_shape(self):
        from src.models.layers.egin import MLP

        mlp = MLP(num_layers=1, input_dim=IN_DIM, hidden_dim=HID_DIM, output_dim=HID_DIM)
        out = mlp(torch.randn(NUM_NODES, IN_DIM))
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_two_layer_output_shape(self):
        from src.models.layers.egin import MLP

        mlp = MLP(num_layers=2, input_dim=IN_DIM, hidden_dim=HID_DIM, output_dim=HID_DIM)
        mlp.eval()
        out = mlp(torch.randn(NUM_NODES, IN_DIM))
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_three_layer_output_shape(self):
        from src.models.layers.egin import MLP

        mlp = MLP(num_layers=3, input_dim=IN_DIM, hidden_dim=HID_DIM, output_dim=HID_DIM)
        mlp.eval()
        out = mlp(torch.randn(NUM_NODES, IN_DIM))
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM))

    def test_different_output_dim(self):
        from src.models.layers.egin import MLP

        mlp = MLP(num_layers=2, input_dim=IN_DIM, hidden_dim=HID_DIM, output_dim=4)
        mlp.eval()
        out = mlp(torch.randn(NUM_NODES, IN_DIM))
        self.assertEqual(out.shape, (NUM_NODES, 4))

    def test_invalid_num_layers_raises(self):
        from src.models.layers.egin import MLP

        with self.assertRaises(ValueError):
            MLP(num_layers=0, input_dim=IN_DIM, hidden_dim=HID_DIM, output_dim=HID_DIM)


class TestGraphEGIN(unittest.TestCase):
    def _make_model(self, **kwargs):
        from src.models.layers.egin import GraphEGIN

        defaults = dict(
            num_layers=2,
            num_edge_feat=EDGE_DIM,
            input_dim=IN_DIM,
            hidden_dim=HID_DIM,
            output_dim=2,
        )
        defaults.update(kwargs)
        return GraphEGIN(**defaults)

    def test_output_shape_single_graph(self):
        model = self._make_model()
        model.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        # EGIN pools to graph level: [num_graphs, output_dim]
        self.assertEqual(out.shape, (1, 2))

    def test_output_shape_multi_graph_batch(self):
        model = self._make_model()
        model.eval()
        x = torch.randn(NUM_NODES * 2, IN_DIM)
        src = torch.randint(0, NUM_NODES * 2, (NUM_EDGES * 2,))
        dst = torch.randint(0, NUM_NODES * 2, (NUM_EDGES * 2,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.randn(NUM_EDGES * 2, EDGE_DIM)
        batch = torch.cat(
            [torch.zeros(NUM_NODES, dtype=torch.long), torch.ones(NUM_NODES, dtype=torch.long)]
        )
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (2, 2))

    def test_with_pos_enc(self):
        model = self._make_model(pos_enc_dim=PE_DIM)
        model.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (1, 2))

    def test_wrong_pos_enc_dim_raises(self):
        model = self._make_model(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_no_edge_attr_raises(self):
        model = self._make_model()
        x, edge_index, _, batch = _make_graph()
        with self.assertRaises((ValueError, AttributeError)):
            model(x=x, edge_index=edge_index, batch=batch, edge_attr=None)

    def test_wrong_edge_feat_size_raises(self):
        model = self._make_model()
        x, edge_index, _, batch = _make_graph()
        wrong_edge_attr = torch.randn(NUM_EDGES, EDGE_DIM + 1)
        with self.assertRaises(ValueError):
            model(x=x, edge_index=edge_index, batch=batch, edge_attr=wrong_edge_attr)

    def test_dot_update_mode(self):
        model = self._make_model(dot_update=True)
        model.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (1, 2))

    def test_edge_mlp_mode(self):
        model = self._make_model(edge_mlp=True)
        model.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (1, 2))

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(ValueError):
            self._make_model(num_layers=1)

    def test_invalid_num_edge_feat_raises(self):
        with self.assertRaises(ValueError):
            self._make_model(num_edge_feat=0)

    def test_edge_type_arg_ignored(self):
        model = self._make_model()
        model.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = model(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (1, 2))


# ===========================================================================
# GraphGPS layers
# ===========================================================================


class TestGraphGPSEncoder(unittest.TestCase):
    def _make_enc(self, **kwargs):
        from src.models.layers.graphgps import GraphGPSEncoder

        defaults = dict(
            in_dim=IN_DIM,
            edge_dim=EDGE_DIM,
            hidden_dim=HID_DIM,
            num_layers=NUM_LAYERS,
        )
        defaults.update(kwargs)
        return GraphGPSEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_out_dim_attribute(self):
        enc = self._make_enc()
        self.assertEqual(enc.out_dim, HID_DIM * (NUM_LAYERS + 1))

    def test_with_pos_enc(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        pos_enc = torch.randn(NUM_NODES, PE_DIM)
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=pos_enc)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_wrong_pos_enc_dim_raises(self):
        enc = self._make_enc(pos_enc_dim=PE_DIM)
        x, edge_index, edge_attr, batch = _make_graph()
        wrong_pe = torch.randn(NUM_NODES, PE_DIM + 1)
        with self.assertRaises(ValueError):
            enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, pos_enc=wrong_pe)

    def test_single_layer(self):
        enc = self._make_enc(num_layers=1)
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * 2))

    def test_invalid_num_layers_raises(self):
        with self.assertRaises(ValueError):
            self._make_enc(num_layers=0)

    def test_layer_norm_type(self):
        enc = self._make_enc(norm_type="layer")
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))

    def test_edge_type_arg_ignored(self):
        enc = self._make_enc()
        enc.eval()
        x, edge_index, edge_attr, batch = _make_graph()
        out = enc(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        self.assertEqual(out.shape, (NUM_NODES, HID_DIM * (NUM_LAYERS + 1)))


if __name__ == "__main__":
    unittest.main()
