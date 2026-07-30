"""Tests for the S2 exact-compression model track.

Tests cover:
- Shape/plumbing sanity for GCNConvExact / GCNEncoderExact / ExactGraphBaseModel
- THE critical test: does the exact-schema pipeline (fold_inversions_into_x
  -> color_refinement -> apply_exact_merge_map) reproduce, through a REAL
  ExactGraphBaseModel with default (non-zeroed, trained-style) weights, the
  same graph-level output as running the model on the uncoarsened graph?
  This is what the general track's TestEncoderInvariance test in
  test_summarization.py is xfailed for; this is the corrected, passing
  version for the model built specifically to satisfy the property.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from data.exact_graph import apply_exact_merge_map, fold_inversions_into_x
from data.summarization import color_refinement
from models.base_model_exact import ExactGraphBaseModel
from models.layers.gcn_exact import GCNConvExact, GCNConvLayerExact, GCNEncoderExact


def _symmetric_graph() -> Data:
    """Two structurally identical PI-pair->AND cones feeding one PO."""
    x = torch.zeros(8, 4, dtype=torch.float32)
    x[0, 0] = 1.0
    x[[1, 2, 4, 5], 1] = 1.0
    x[[3, 6], 2] = 1.0
    x[7, 3] = 1.0
    edge_index = torch.tensor(
        [[1, 2, 4, 5, 3, 6], [3, 3, 6, 6, 7, 7]], dtype=torch.long
    )
    edge_attr = torch.tensor([[1, 0]] * 6, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 8
    return data


def _asymmetric_polarity_graph() -> Data:
    """Same shape as _symmetric_graph, but one cone's fanins are inverted.

    Forces color_refinement to keep the two AND-gate cones apart (different
    inverted-fanin counts after folding), so the exactness check exercises
    a case where inversion actually blocks a merge, not just multiplicity.
    """
    data = _symmetric_graph()
    data.edge_attr = data.edge_attr.clone()
    data.edge_attr[0] = torch.tensor([0.0, 1.0])  # 1->3 inverted
    data.edge_attr[1] = torch.tensor([0.0, 1.0])  # 2->3 inverted
    return data


class TestGCNConvExact:
    def test_forward_shape(self) -> None:
        conv = GCNConvExact(4, 8)
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 4]])
        edge_weight = torch.ones(3)
        out = conv(x, edge_index, edge_weight)
        assert out.shape == (5, 8)

    def test_no_incoming_edges_yields_bias_only(self) -> None:
        conv = GCNConvExact(4, 8)
        with torch.no_grad():
            conv.bias_param.fill_(1.0)
        x = torch.randn(3, 4)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,))
        out = conv(x, edge_index, edge_weight)
        assert torch.allclose(out, torch.ones(3, 8))


class TestGCNConvLayerExact:
    def test_forward_matches_hand_computation(self) -> None:
        """Direct check of the formula, independent of the coarsening-
        equality tests: those can't distinguish a correct residual/FFN from
        an incorrect one, since any deterministic elementwise function
        trivially preserves already-equal inputs (see the class docstring).
        """
        torch.manual_seed(0)
        layer = GCNConvLayerExact(dim_in=3, dim_out=3, dropout=0.0).eval()

        x = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
        edge_index = torch.tensor([[0], [1]])
        edge_weight = torch.tensor([2.0])

        with torch.no_grad():
            conv_out = layer.model(x, edge_index=edge_index, edge_weight=edge_weight)
            expected = x + layer.drop(layer.act(conv_out))
            expected = expected + layer._ff_block(expected)

            actual = layer(x, edge_index=edge_index, edge_weight=edge_weight)

        assert torch.allclose(actual, expected)


class TestGCNEncoderExact:
    def test_cat_jk_output_dim(self) -> None:
        enc = GCNEncoderExact(hid_dim=8, num_layers=3, node_input_dim=8, jk_mode="cat")
        x = torch.randn(4, 8)
        edge_index = torch.tensor([[0, 1], [2, 3]])
        edge_weight = torch.ones(2)
        out = enc(x, edge_index, edge_weight)
        assert out.shape == (4, 8 * 4)

    def test_last_jk_output_dim(self) -> None:
        enc = GCNEncoderExact(hid_dim=8, num_layers=2, node_input_dim=8, jk_mode="last")
        x = torch.randn(4, 8)
        edge_index = torch.tensor([[0], [1]])
        edge_weight = torch.ones(1)
        out = enc(x, edge_index, edge_weight)
        assert out.shape == (4, 8)


class TestExactGraphBaseModel:
    def test_forward_batch_shape(self) -> None:
        model = ExactGraphBaseModel(
            hidden_dim=8, num_layers=2, node_input_dim=5, jk_mode="cat"
        ).eval()
        data = fold_inversions_into_x(_symmetric_graph())
        batch = Batch.from_data_list([data, data])
        with torch.no_grad():
            out = model.forward_batch(batch)
        assert out.shape == (2, 1)

    def test_size_weighted_pool_matches_mean_when_node_size_absent(self) -> None:
        # No node_size on the batch -> defaults to all-ones -> size-weighted
        # mean must equal plain mean pooling exactly.
        model = ExactGraphBaseModel(
            hidden_dim=8, num_layers=1, node_input_dim=5, jk_mode="last"
        ).eval()
        data = fold_inversions_into_x(_symmetric_graph())
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            enc_out = model.encoder(
                model.encode_and_integrate(batch.x), batch.edge_index, batch.edge_weight
            )
            from torch_geometric.nn import global_mean_pool

            plain_mean = global_mean_pool(enc_out, batch.batch, size=1)
            node_size = torch.ones(enc_out.size(0), 1)
            weighted = model._pool_size_weighted(enc_out, batch.batch, node_size, size=1)
        assert torch.allclose(plain_mean, weighted)

    def test_missing_edge_weight_raises(self) -> None:
        model = ExactGraphBaseModel(hidden_dim=8, num_layers=1, node_input_dim=4)
        data = Data(x=torch.eye(4), edge_index=torch.empty((2, 0), dtype=torch.long))
        data.num_nodes = 4
        batch = Batch.from_data_list([data])
        try:
            model.forward_batch(batch)
            raised = False
        except ValueError:
            raised = True
        assert raised


class TestExactnessThroughRealModel:
    """The property the whole exact-compression track exists to provide."""

    @staticmethod
    def _run(model: ExactGraphBaseModel, data: Data) -> torch.Tensor:
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            return model.forward_batch(batch)

    def test_exact_refinement_preserves_graph_output(self) -> None:
        torch.manual_seed(0)
        model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=2, node_input_dim=5, jk_mode="cat"
        ).eval()
        # Default construction: real (non-zeroed) biases, standard init --
        # nothing about the weights is special-cased for this property.

        folded = fold_inversions_into_x(_symmetric_graph())
        cluster = color_refinement(folded, depth=2, pe_aware=False)
        assert len(cluster.unique()) < folded.x.size(0), "no compression happened"
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        original_out = self._run(model, folded)
        coarse_out = self._run(model, coarse)

        assert torch.allclose(original_out, coarse_out, atol=1e-5), (
            f"graph-level output changed: {original_out.tolist()} vs "
            f"{coarse_out.tolist()}"
        )

    def test_exactness_survives_a_blocked_merge(self) -> None:
        # The two cones must NOT merge here (different inversion patterns),
        # so this exercises real per-target-member multiplicity (>1) on a
        # partial, not total, coarsening.
        torch.manual_seed(1)
        model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=2, node_input_dim=5, jk_mode="cat"
        ).eval()

        folded = fold_inversions_into_x(_asymmetric_polarity_graph())
        cluster = color_refinement(folded, depth=2, pe_aware=False)
        assert cluster[3] != cluster[6], "inversion difference should block the merge"
        assert len(cluster.unique()) < folded.x.size(0), "no compression happened"
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        original_out = self._run(model, folded)
        coarse_out = self._run(model, coarse)
        assert torch.allclose(original_out, coarse_out, atol=1e-5)

    def test_exactness_holds_with_alternate_jk_mode(self) -> None:
        torch.manual_seed(2)
        model = ExactGraphBaseModel(
            hidden_dim=12, num_layers=3, node_input_dim=5, jk_mode="sum"
        ).eval()

        folded = fold_inversions_into_x(_symmetric_graph())
        cluster = color_refinement(folded, depth=3, pe_aware=False)
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        original_out = self._run(model, folded)
        coarse_out = self._run(model, coarse)
        assert torch.allclose(original_out, coarse_out, atol=1e-5)

    def test_exactness_holds_inside_a_mixed_batch(self) -> None:
        # Batch.from_data_list drops an attribute from the WHOLE batch if
        # even one graph lacks it. Before fold_inversions_into_x always set
        # node_size, batching an uncoarsened graph next to a coarsened one
        # silently dropped node_size everywhere, and every existing test
        # used single-graph batches so this went uncaught.
        torch.manual_seed(4)
        model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=2, node_input_dim=5, jk_mode="cat"
        ).eval()

        folded = fold_inversions_into_x(_symmetric_graph())
        cluster = color_refinement(folded, depth=2, pe_aware=False)
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        solo_original = self._run(model, folded)
        solo_coarse = self._run(model, coarse)

        mixed_batch = Batch.from_data_list([folded, coarse])
        with torch.no_grad():
            mixed_out = model.forward_batch(mixed_batch)

        assert torch.allclose(mixed_out[0:1], solo_original, atol=1e-5)
        assert torch.allclose(mixed_out[1:2], solo_coarse, atol=1e-5)

    def test_missing_node_size_in_mixed_batch_raises_not_silently_defaults(
        self,
    ) -> None:
        model = ExactGraphBaseModel(hidden_dim=8, num_layers=1, node_input_dim=5)
        coarse_like = Data(
            x=torch.eye(5)[:2],
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_weight=torch.empty((0,)),
        )  # deliberately no node_size, unlike real apply_exact_merge_map output
        coarse_like.num_nodes = 2
        try:
            model.forward_batch(Batch.from_data_list([coarse_like]))
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_dropout_disabled_at_eval_does_not_break_exactness(self) -> None:
        # A nonzero dropout hyperparameter must not reintroduce randomness
        # once .eval() is called -- confirms the property holds for a
        # config matched to a tuned model (which will have dropout > 0).
        torch.manual_seed(3)
        model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=2, node_input_dim=5, dropout=0.3, jk_mode="cat"
        ).eval()

        folded = fold_inversions_into_x(_symmetric_graph())
        cluster = color_refinement(folded, depth=2, pe_aware=False)
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        original_out = self._run(model, folded)
        coarse_out = self._run(model, coarse)
        assert torch.allclose(original_out, coarse_out, atol=1e-5)
