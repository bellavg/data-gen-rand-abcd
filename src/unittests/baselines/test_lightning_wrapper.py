"""Tests for BaselineRegressionLightningModule's own behaviour.

Deliberately architecture-agnostic (a tiny dummy `nn.Module`, not any real
baseline) because these tests are about the WRAPPER, shared by all four
baselines -- SynthNet, HOGA, DeepGate4 and Gamora each get their own
Lightning-training tests in their own test file for that reason. The one
exception is `compile_model`, covered here at the wrapper level plus one
Gamora-specific end-to-end check in test_gamora.py, since Gamora is the only
baseline actually turned on under compile (see
baselines/common/lightning_wrapper.py's module docstring for why).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from baselines.common.lightning_wrapper import BaselineRegressionLightningModule


class _TinyGraphModel(nn.Module):
    """Smallest possible stand-in for a baseline model: `forward(batch) ->
    (num_graphs, 1)`. Deliberately includes a pooling-style reduction over the
    batch vector so the shape genuinely varies with graph/node count, the same
    way every real baseline's forward does."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, batch) -> torch.Tensor:
        h = torch.relu(self.lin(batch.x))
        num_graphs = int(batch.batch.max()) + 1 if batch.batch.numel() else 1
        pooled = torch.zeros(num_graphs, h.size(1), dtype=h.dtype, device=h.device)
        pooled.index_add_(0, batch.batch, h)
        counts = torch.bincount(batch.batch, minlength=num_graphs).clamp(min=1)
        pooled = pooled / counts.unsqueeze(1).to(h.dtype)
        return torch.sigmoid(self.out(pooled))


def _make_batch(num_graphs: int, seed: int) -> Batch:
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(num_graphs):
        n = 3 + (i % 4)
        x = torch.rand(n, 4, generator=g)
        edge_index = torch.randint(0, n, (2, max(1, n)), generator=g)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.rand(1, 1, generator=g)))
    return Batch.from_data_list(graphs)


class TestCompileModel(unittest.TestCase):
    """`compile_model` -- see lightning_wrapper.py's module docstring.

    torch.compile is genuinely slow to trace/compile (real JIT compilation),
    so these tests are inherently slower than the rest of the suite. Kept
    minimal and CPU-only (no CUDA-specific backend assumptions) so they still
    run in the same environment as everything else.
    """

    def test_default_leaves_the_model_unwrapped(self):
        model = _TinyGraphModel()
        wrapped = BaselineRegressionLightningModule(model, lr=1e-3)
        self.assertIs(wrapped.model, model)

    def test_compile_model_true_wraps_with_torch_compile(self):
        model = _TinyGraphModel()
        wrapped = BaselineRegressionLightningModule(
            model, lr=1e-3, compile_model=True
        )
        self.assertIsNot(wrapped.model, model)
        # OptimizedModule is torch.compile's own wrapper type; asserting on it
        # by name (rather than importing torch._dynamo directly) keeps this
        # test stable across torch versions that move the class around.
        self.assertEqual(type(wrapped.model).__name__, "OptimizedModule")

    def test_compiled_forward_matches_eager(self):
        torch.manual_seed(0)
        model = _TinyGraphModel()
        model.eval()
        batch = _make_batch(num_graphs=4, seed=1)

        with torch.no_grad():
            eager_out = model(batch)

        wrapped = BaselineRegressionLightningModule(
            model, lr=1e-3, compile_model=True
        )
        wrapped.eval()
        with torch.no_grad():
            compiled_out = wrapped(batch)

        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-4, atol=1e-5)

    def test_compiled_gradients_flow(self):
        torch.manual_seed(0)
        model = _TinyGraphModel()
        wrapped = BaselineRegressionLightningModule(
            model, lr=1e-3, compile_model=True
        )
        wrapped.train()
        batch = _make_batch(num_graphs=5, seed=2)

        wrapped(batch).sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, msg=f"no gradient for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), msg=name)

    def test_compiled_handles_varying_batch_shapes_without_crashing(self):
        """Node-budget batching (and, for HOGA/DeepGate4, per-graph feature
        adapters) means every real baseline sees a DIFFERENT shape every step
        -- never a fixed batch size. This just requires several distinct
        shapes to complete a forward+backward without error; it does NOT
        verify dynamic=True is doing anything -- see the next test for that,
        and its docstring for why "doesn't crash" alone cannot tell them apart.
        """
        torch.manual_seed(0)
        model = _TinyGraphModel()
        wrapped = BaselineRegressionLightningModule(
            model, lr=1e-3, compile_model=True
        )
        wrapped.train()

        for num_graphs in (3, 7, 2, 10):
            batch = _make_batch(num_graphs=num_graphs, seed=num_graphs)
            out = wrapped(batch)
            self.assertEqual(out.shape, (num_graphs, 1))
            out.sum().backward()
            wrapped.zero_grad()

    def test_dynamic_true_compiles_substantially_less_than_without_it(self):
        """Confirms dynamic=True is actually doing something, which "doesn't
        crash" cannot: torch's own automatic dynamic-shape detection adapts
        after ONE recompile even without it (verified: on the real
        GamoraGraphRegressor, dynamic=None recompiles once more than
        dynamic=True across 6 shapes, then both stabilize -- see
        lightning_wrapper.py's module docstring), so a plain
        crash-or-doesn't-crash check passes identically whether dynamic=True
        is present or silently dropped. This counts actual compiler
        invocations via a custom backend instead, and compares dynamic=True
        against the SAME model/shapes without it in one test, rather than
        against a hardcoded threshold that could drift with the torch version.
        """
        import torch._dynamo as dynamo

        shapes = (3, 7, 2, 10, 5, 8)

        def compiles_for(dynamic) -> int:
            dynamo.reset()
            count = 0

            def counting_backend(gm, example_inputs):
                nonlocal count
                count += 1
                return gm.forward

            model = _TinyGraphModel()
            compiled = torch.compile(model, dynamic=dynamic, backend=counting_backend)
            for num_graphs in shapes:
                compiled(_make_batch(num_graphs=num_graphs, seed=num_graphs))
            return count

        with_dynamic = compiles_for(True)
        without_dynamic = compiles_for(None)  # torch's own default, not False:
        # this is the realistic regression (someone drops the kwarg), not an
        # artificially pessimistic comparison against fully-static compiling.
        self.assertLess(
            with_dynamic,
            without_dynamic,
            f"dynamic=True ({with_dynamic} compiles) should compile fewer "
            f"times than the default ({without_dynamic}) across varying shapes",
        )


class TestCompiledCheckpointKeys(unittest.TestCase):
    """torch.compile prefixes every submodule key under the compiled attribute
    with `_orig_mod.` (verified directly: `OptimizedModule.state_dict()`
    returns `_orig_mod.lin.weight`, not `lin.weight`, though `._orig_mod is`
    the original module -- same tensors, different key names). This repo's
    own baseline-checkpoint consumers strip a fixed `"model."` prefix and load
    strict=True (e.g. diagnose_synthnet_baseline.py:108-113); left alone, a
    compiled Gamora checkpoint would silently be the only one in a different
    key format. state_dict()/load_state_dict() correct for this.
    """

    def test_state_dict_keys_have_no_compile_prefix(self):
        wrapped = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=True
        )
        wrapped(_make_batch(num_graphs=2, seed=0))  # trigger compilation
        keys = list(wrapped.state_dict().keys())
        self.assertTrue(keys, "state_dict is empty")
        self.assertTrue(all("_orig_mod" not in k for k in keys), keys)
        self.assertIn("model.lin.weight", keys)

    def test_state_dict_keys_match_the_uncompiled_module_exactly(self):
        """The whole point: a checkpoint must be interchangeable regardless of
        whether the run that produced it was compiled."""
        torch.manual_seed(0)
        compiled_wrapped = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=True
        )
        torch.manual_seed(0)
        plain_wrapped = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=False
        )
        self.assertEqual(
            set(compiled_wrapped.state_dict().keys()),
            set(plain_wrapped.state_dict().keys()),
        )

    def test_a_compiled_checkpoint_loads_into_an_uncompiled_module(self):
        """The realistic downstream case: train under compile, evaluate
        without it (e.g. a future test-split eval path)."""
        torch.manual_seed(0)
        compiled_wrapped = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=True
        )
        saved = compiled_wrapped.state_dict()

        fresh = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=False
        )
        fresh.load_state_dict(saved)  # must not raise
        torch.testing.assert_close(
            dict(fresh.named_parameters())["model.lin.weight"],
            saved["model.lin.weight"],
        )

    def test_an_uncompiled_checkpoint_loads_into_a_compiled_module(self):
        """The other direction: resume a plain run under compile."""
        torch.manual_seed(0)
        plain_wrapped = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=False
        )
        saved = plain_wrapped.state_dict()

        fresh = BaselineRegressionLightningModule(
            _TinyGraphModel(), lr=1e-3, compile_model=True
        )
        fresh.load_state_dict(saved)  # must not raise
        torch.testing.assert_close(
            dict(fresh.state_dict())["model.lin.weight"],
            saved["model.lin.weight"],
        )


if __name__ == "__main__":
    unittest.main()
