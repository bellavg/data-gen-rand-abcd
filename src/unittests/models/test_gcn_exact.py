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

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data

import config
from data.data_utils import aig_to_pytorch_geometric
from data.exact_graph import apply_exact_merge_map, fold_inversions_into_x
from data.summarization import color_refinement, summarize_graph
from models.base_model_exact import ExactGraphBaseModel
from models.layers.gcn_exact import GCNConvExact, GCNConvLayerExact, GCNEncoderExact
from models.lightning_model import AIGRegressionLightningModule

# The one real AIG in the test tree, shared with the dataset tests.
_AIG_PATH = Path(__file__).resolve().parent.parent / "data" / "adder.aig"


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


def _intra_cluster_edge_graph() -> Data:
    """A graph whose depth-1 WL classes contain an edge between members.

    p1, p2 are PIs; t = AND(p1, p2); u = AND(t, p1); v = AND(u, p1); o is a
    PO on v.  After one refinement round u and v share a class (both are
    ANDs with fanin colours {AND, PI}) *and* u is a fanin of v — so the
    reduct has a real self-loop, which is the case the removed
    intra-cluster guard used to reject.
    """
    x = torch.zeros(6, 4, dtype=torch.float32)
    x[[0, 1], 1] = 1.0  # p1, p2: PI
    x[[2, 3, 4], 2] = 1.0  # t, u, v: AND
    x[5, 3] = 1.0  # o: PO
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 3, 0, 4], [2, 2, 3, 3, 4, 4, 5]], dtype=torch.long
    )
    edge_attr = torch.tensor([[1, 0]] * 7, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6
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
            weighted = model._pool_size_weighted(
                enc_out, batch.batch, node_size, size=1
            )
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


class TestModelSelector:
    """The `--model exact` wiring, which is what actually reaches training.

    Everything else in this file builds ExactGraphBaseModel directly, so it
    would all keep passing if the Lightning module quietly went on building
    UnifiedGraphBaseModel — which is the state this whole track was in.
    """

    @staticmethod
    def _kwargs(**overrides):
        base = {
            "encoder_name": "gcn",
            "hidden_dim": 16,
            "encoder_kwargs": {"num_layers": 2, "hid_dim": 16, "dropout": 0.0},
            "pe_type": "none",
            "pooling_type": "mean",
            "compile_model": False,
        }
        base.update(overrides)
        return base

    def test_exact_model_type_builds_the_exact_model(self) -> None:
        module = AIGRegressionLightningModule(
            **self._kwargs(
                model_type="exact", node_input_dim=config.EXACT_NODE_INPUT_DIM
            )
        )
        assert isinstance(module.model, ExactGraphBaseModel)

    def test_default_model_type_stays_on_the_production_model(self) -> None:
        module = AIGRegressionLightningModule(**self._kwargs())
        assert not isinstance(module.model, ExactGraphBaseModel)

    def test_exact_model_refuses_a_positional_encoding(self) -> None:
        # Silently ignoring it would report a PE the model never read.
        with pytest.raises(ValueError, match="requires pe_type='none'"):
            AIGRegressionLightningModule(
                **self._kwargs(model_type="exact", pe_type="level")
            )

    def test_exact_model_refuses_non_mean_pooling(self) -> None:
        with pytest.raises(ValueError, match="pooling_type must be 'mean'"):
            AIGRegressionLightningModule(
                **self._kwargs(model_type="exact", pooling_type="sum")
            )

    def test_unknown_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            AIGRegressionLightningModule(**self._kwargs(model_type="approximate"))

    def test_model_type_survives_a_checkpoint_round_trip(self, tmp_path) -> None:
        # verify_exact_rq5.py reads hparams.model_type off a loaded
        # checkpoint to refuse non-exact ones; that only works if
        # save_hyperparameters captured it.
        module = AIGRegressionLightningModule(
            **self._kwargs(
                model_type="exact", node_input_dim=config.EXACT_NODE_INPUT_DIM
            )
        )
        ckpt = tmp_path / "m.ckpt"
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            ckpt,
        )
        restored = AIGRegressionLightningModule.load_from_checkpoint(
            ckpt, map_location="cpu"
        )
        assert restored.hparams.model_type == "exact"
        assert isinstance(restored.model, ExactGraphBaseModel)


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

    def test_exactness_holds_with_intra_cluster_edges(self) -> None:
        # The case apply_exact_merge_map used to refuse.  An intra-cluster
        # edge becomes a self-loop of weight intra_edges / class_size, and
        # that weight is exactly what makes the sum come out right: members
        # of the class disagree about *which* class each fanin came from,
        # but agree on the multiset of fanin colours, so splitting the count
        # across the classes and re-averaging preserves the total.
        torch.manual_seed(5)
        model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=1, node_input_dim=5, jk_mode="cat"
        ).eval()

        folded = fold_inversions_into_x(_intra_cluster_edge_graph())
        cluster = color_refinement(folded, depth=1, pe_aware=False)
        assert cluster[3] == cluster[4], "u and v should share a depth-1 class"
        coarse = apply_exact_merge_map(folded, cluster, int(cluster.max()) + 1)

        loops = coarse.edge_index[0] == coarse.edge_index[1]
        assert int(loops.sum()) >= 1, "fixture produced no intra-cluster edge"

        assert torch.allclose(
            self._run(model, folded), self._run(model, coarse), atol=1e-5
        )

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
        # Both concatenate along dim 0 and carry no node indices, so PyG
        # needs no __inc__/__cat_dim__ override -- but assert it rather than
        # relying on the output comparison to notice, because a dropped
        # attribute fails as a wrong number, not an error.
        assert getattr(mixed_batch, "node_size", None) is not None
        assert getattr(mixed_batch, "edge_weight", None) is not None
        assert mixed_batch.node_size.size(0) == folded.x.size(0) + coarse.x.size(0)
        assert mixed_batch.edge_weight.size(0) == (
            folded.edge_index.size(1) + coarse.edge_index.size(1)
        )

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

    @pytest.mark.parametrize("depth", [1, 2, config.NUM_LAYERS])
    def test_real_aig_end_to_end_through_the_production_entry_point(
        self, depth: int
    ) -> None:
        """The single most valuable test here: a real AIG, the real
        ``summarize_graph`` dispatch, the real model.

        Everything above builds the reduct by calling the three stages by
        hand, which would keep passing if the precompute driver stopped
        routing ``wl_exact`` through them.  This goes through
        ``summarize_graph``, i.e. exactly what ``data.summarize_graphs``
        calls, with config's own parameters apart from ``depth``.

        Parametrised over depth because the exactness argument does not
        require the refinement to have converged: layer-*l* messages depend
        only on round-(*l*-1) colours, so a reduct built at depth *d* is
        exact for every model depth *l* <= *d* — which is precisely the
        invariant ``assert_exact_depth_supports_model`` protects.
        """
        raw = aig_to_pytorch_geometric(_AIG_PATH)
        params = {**config.SUMMARIZATION_PARAMS["wl_exact"], "depth": depth}
        coarse = summarize_graph(raw, "wl_exact", **params)

        assert coarse.x.size(1) == config.EXACT_NODE_INPUT_DIM
        assert getattr(coarse, "edge_attr", None) is None
        assert coarse.x.size(0) < raw.x.size(0), "no compression happened"
        assert int(coarse.node_size.sum()) == raw.x.size(0)

        torch.manual_seed(6)
        model = ExactGraphBaseModel(
            hidden_dim=16,
            num_layers=depth,
            node_input_dim=config.EXACT_NODE_INPUT_DIM,
            jk_mode="cat",
        ).eval()

        original_out = self._run(model, fold_inversions_into_x(raw))
        coarse_out = self._run(model, coarse)

        assert torch.allclose(original_out, coarse_out, rtol=1e-5, atol=1e-6), (
            f"depth={depth}: {original_out.tolist()} vs {coarse_out.tolist()}"
        )

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


class TestExactVsGeneralTrackMargin:
    """Quantifies the gap the general track's ``TestEncoderInvariance``
    (test_summarization.py) only xfails: on the same fixture graph, the
    exact track reproduces the full-graph output and the general
    (production) track does not, by a real, non-floating-point-noise
    margin. If ``general_residual`` ever drops near zero, the production
    architecture changed and that xfail should be revisited.
    """

    def test_exact_track_matches_general_track_does_not(self) -> None:
        from data.summarization import apply_merge_map
        from models.base_model import UnifiedGraphBaseModel

        # The two tracks use different native preprocessing (folded,
        # edge_attr-free schema + apply_exact_merge_map for exact; raw
        # one-hot schema + apply_merge_map for general) so their colour
        # refinements cluster differently -- that's each track's real
        # pipeline, not an unfair comparison. What's compared is each
        # track's own full-graph-vs-coarsened-graph residual.
        graph = _symmetric_graph()

        torch.manual_seed(0)
        exact_model = ExactGraphBaseModel(
            hidden_dim=16, num_layers=2, node_input_dim=5, jk_mode="cat"
        ).eval()
        folded = fold_inversions_into_x(graph)
        exact_cluster = color_refinement(folded, depth=2, pe_aware=False)
        exact_coarse = apply_exact_merge_map(
            folded, exact_cluster, int(exact_cluster.max()) + 1
        )
        exact_residual = (
            (self._run(exact_model, folded) - self._run(exact_model, exact_coarse))
            .abs()
            .max()
        )

        torch.manual_seed(0)
        general_model = UnifiedGraphBaseModel(
            encoder_name="gcn",
            hidden_dim=16,
            encoder_kwargs={"num_layers": 2, "hid_dim": 16},
            pe_type="none",
            task_out_dim=1,
            pooling_type="mean",
        ).eval()
        # apply_merge_map requires 'level' or 'pos_enc' to derive the merged
        # graph's positional attribute; the exact track doesn't need either.
        graph.level = torch.tensor(
            [[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [2.0]]
        )
        general_cluster = color_refinement(graph, depth=2, count_cap=None)
        general_coarse = apply_merge_map(
            graph, general_cluster, int(general_cluster.max()) + 1
        )
        with torch.no_grad():
            general_full_out = general_model.forward_batch(
                Batch.from_data_list([graph])
            )
            general_coarse_out = general_model.forward_batch(
                Batch.from_data_list([general_coarse])
            )
        general_residual = (general_full_out - general_coarse_out).abs().max()

        assert exact_residual < 1e-5, (
            "exact track should reproduce the full-graph output"
        )
        # Observed ~5.6e-3 on this fixture/seed; 1e-3 leaves a ~5x margin so
        # this isn't tripped by ordinary float rounding, only by the
        # architecture actually regaining the quotient property.
        assert general_residual > 1e-3, (
            "general track should NOT reproduce the full-graph output "
            "(GraphNorm + unweighted mean pooling break the quotient "
            "property)"
        )

    @staticmethod
    def _run(model: ExactGraphBaseModel, data: Data) -> torch.Tensor:
        batch = Batch.from_data_list([data])
        with torch.no_grad():
            return model.forward_batch(batch)
