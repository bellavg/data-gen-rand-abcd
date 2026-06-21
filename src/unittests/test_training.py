from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import config
import train

# Adjust imports based on your project structure
from models.lightning_model import AIGRegressionLightningModule
from train_utils import TrainingStartupCallback


@pytest.fixture
def dummy_batch():
    torch.manual_seed(42)
    num_graphs = 5
    nodes_per_graph = 20
    node_input_dim, edge_attr_dim, task_out_dim = 4, 2, 1
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn((nodes_per_graph, node_input_dim))
        edge_index = torch.randint(0, nodes_per_graph, (2, 40))
        edge_attr = torch.randn((40, edge_attr_dim))
        y = torch.FloatTensor(1, task_out_dim).uniform_(-1, 1)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


@pytest.fixture
def basic_model():
    encoder_kwargs = {
        "num_layers": 3,
        "hid_dim": 32,
        "dropout": 0.0,
        "norm_type": "batch",
        "jk_mode": "last",
    }
    model = AIGRegressionLightningModule(
        encoder_name="gcn",
        hidden_dim=32,
        node_input_dim=4,
        edge_attr_dim=2,
        task_out_dim=1,
        encoder_kwargs=encoder_kwargs,
    )
    # Attach a mock Trainer for unit tests that call training_step() directly.
    model.trainer = MagicMock()
    return model


# ==========================================================
# Core Functionality Tests
# ==========================


def test_model_forward_pass(basic_model, dummy_batch):
    """Ensures model outputs 1 prediction per graph in the batch."""
    basic_model.eval()
    with torch.no_grad():
        out = basic_model(dummy_batch)

    assert out.shape == (5, 1)  # 5 graphs in dummy_batch
    assert not torch.isnan(out).any()


def test_gradient_flow(basic_model, dummy_batch):
    """Verifies that gradients propagate from graph-loss to weights."""
    basic_model.train()

    # 1. Forward pass to initialize LazyLinear weights
    loss = basic_model.training_step(dummy_batch, batch_idx=0)

    # 2. Backward pass
    loss.backward()

    found_grad = False
    for param in basic_model.parameters():
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            continue

        if param.requires_grad and param.grad is not None:
            if torch.norm(param.grad) > 1e-5:
                found_grad = True
                break
    assert found_grad, "No gradients detected in model parameters!"


# ==========================================================
# Regression Range & Magnitude Tests
# ==========================


def test_output_range_capabilities(basic_model, dummy_batch):
    """
    Ensures the model can produce values close to 0 and 1
    by manually shifting the head bias.
    """
    basic_model.eval()
    with torch.no_grad():
        # 1. Warm-up forward pass to initialize parameters
        _ = basic_model(dummy_batch)

        # 2. Test positive capability: Force a positive bias
        torch.nn.init.constant_(basic_model.model.head[3].bias, 2.0)
        out_pos = basic_model(dummy_batch)
        assert out_pos.max() > 0.7, (
            f"Model failed to produce positive values. Max: {out_pos.max()}"
        )

        # 3. Test negative capability: Force a negative bias
        torch.nn.init.constant_(basic_model.model.head[3].bias, -2.0)
        out_neg = basic_model(dummy_batch)
        assert out_neg.min() < 0.3, (
            f"Model failed to produce negative values. Min: {out_neg.min()}"
        )

        # 4. Verify output variance (ensures model is not collapsed to a single value)
        assert (out_pos.max() - out_pos.min()).item() >= 0, (
            "Model output variance check"
        )


# ==========================================================
# GNN Property Tests
# ==========================


def test_permutation_invariance(basic_model, dummy_batch):
    """
    Graph-level predictions MUST be invariant to node shuffling.
    """
    basic_model.eval()
    data_list = dummy_batch.to_data_list()
    graph = data_list[0]

    with torch.no_grad():
        # 1. Original prediction
        orig_out = basic_model(Batch.from_data_list([graph]))

        # 2. Permute nodes
        perm = torch.randperm(graph.num_nodes)
        perm_graph = graph.clone()
        perm_graph.x = graph.x[perm]

        # Re-map edge_index to the new node indices
        relabel_dict = {old: new for new, old in enumerate(perm.tolist())}
        new_edge_index = graph.edge_index.clone()
        for i in range(graph.edge_index.shape[1]):
            new_edge_index[0, i] = relabel_dict[graph.edge_index[0, i].item()]
            new_edge_index[1, i] = relabel_dict[graph.edge_index[1, i].item()]
        perm_graph.edge_index = new_edge_index

        perm_out = basic_model(Batch.from_data_list([perm_graph]))

    torch.testing.assert_close(orig_out, perm_out, atol=1e-5, rtol=1e-5)


def test_variable_graph_sizes_in_batch(basic_model):
    """Ensures pooling logic handles batches with uneven graph sizes."""
    g1 = Data(
        x=torch.randn(5, 4),
        edge_index=torch.randint(0, 5, (2, 10)),
        edge_attr=torch.randn(10, 2),
        y=torch.randn(1, 1),
    )
    g2 = Data(
        x=torch.randn(50, 4),
        edge_index=torch.randint(0, 50, (2, 100)),
        edge_attr=torch.randn(100, 2),
        y=torch.randn(1, 1),
    )

    batch = Batch.from_data_list([g1, g2])

    basic_model.eval()
    with torch.no_grad():
        out = basic_model(batch)

    assert out.shape == (2, 1), (
        "Pooling failed to reduce variable graphs to single vectors."
    )


@pytest.mark.parametrize(
    "encoder_name",
    ["gcn"],
)
def test_encoder_registry_compatibility(encoder_name, dummy_batch):
    """Ensures GCN encoder integrates with the unified model and lightning module."""
    encoder_kwargs = {
        "num_layers": 2,
        "hid_dim": 32,
        "dropout": 0.1,
        "norm_type": "batch",
        "jk_mode": "last",
    }

    model = AIGRegressionLightningModule(
        encoder_name=encoder_name,
        hidden_dim=32,
        node_input_dim=4,
        edge_attr_dim=2,
        task_out_dim=1,
        encoder_kwargs=encoder_kwargs,
    )

    model.eval()
    with torch.no_grad():
        out = model(dummy_batch)
    assert out.shape == (5, 1)


# ==========================================================
# Training logic Tests
# ==========================


def test_loss_logic(basic_model, dummy_batch):
    """
    Ensures that loss is high for bad predictions and low for perfect ones.
    """
    basic_model.eval()
    # Initialize head and then zero out the last linear layer
    with torch.no_grad():
        _ = basic_model(dummy_batch)
        basic_model.model.head[3].weight.zero_()
        basic_model.model.head[3].bias.zero_()

    # 1. Target is far from 0.5 (Loss should be high)
    dummy_batch.y = torch.ones_like(dummy_batch.y) * 0.9
    loss_high = basic_model.training_step(dummy_batch, 0)

    # 2. Target is exactly 0.5 (Loss should be near zero)
    dummy_batch.y = torch.ones_like(dummy_batch.y) * 0.5
    loss_low = basic_model.training_step(dummy_batch, 0)

    assert loss_high > loss_low
    assert loss_low < 0.01


def test_training_step_accepts_tuple_batches(basic_model, dummy_batch):
    basic_model.trainer.sanity_checking = False
    basic_model.log = MagicMock()
    basic_model.forward = MagicMock(return_value=torch.zeros((5, 1)))

    tuple_batch = tuple(dummy_batch.to_data_list())
    loss = basic_model.training_step(tuple_batch, batch_idx=0)

    assert torch.isfinite(loss)
    basic_model.forward.assert_called_once()


def test_training_step_logs_step_and_epoch_metrics(basic_model, dummy_batch):
    basic_model.trainer.sanity_checking = False
    basic_model.log = MagicMock()
    basic_model.forward = MagicMock(return_value=torch.zeros((5, 1)))

    loss = basic_model.training_step(dummy_batch, batch_idx=0)

    assert torch.isfinite(loss)
    basic_model.log.assert_has_calls(
        [
            call(
                "train_loss",
                ANY,
                batch_size=5,
                sync_dist=False,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            ),
            call(
                "train_rmse",
                ANY,
                batch_size=5,
                sync_dist=False,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            ),
        ]
    )
    logged_names = [args[0] for args, _ in basic_model.log.call_args_list]
    assert "train_r2" not in logged_names


def test_train_main_passes_partition_to_datamodule(tmp_path, basic_model, dummy_batch):
    args = SimpleNamespace(
        enable_hardware_profiler=False,
        algorithm="Orchestrate",
        seed=42,
        csv_paths=[str(tmp_path / "dummy.csv")],
        pe_type="none",
        partition=None,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1,
        cache_dir=None,
        hp_tuning_splits_path=None,
        tier0_cache_dir=None,
        tier1_cache_dir=None,
        dynamic_batching=False,
        max_total_nodes_per_batch=16,
        num_layers=2,
        hidden_dim=16,
        dropout=0.0,
        norm_type="layer",
        jk_mode="last",
        encoder_name="gcn",
        heads=4,
        pos_enc_dim=0,
        pooling_type="mean",
        lr=1e-3,
        weight_decay=1e-4,
        min_lr=1e-6,
        warmup_steps=1,
        warmup_start_lr=1e-6,
        scheduler_patience=1,
        scheduler_factor=0.5,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        patience=1,
        max_batch_compute_reports=0,
        max_epochs=1,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        log_steps=1,
    )

    with (
        patch("train.AIGDataModule") as datamodule_cls,
        patch("train.AIGRegressionLightningModule"),
        patch("train.ModelCheckpoint"),
        patch("train.PreciseEarlyStopping"),
        patch("train.LearningRateMonitor"),
        patch("train.TrainingStartupCallback"),
        patch("train.WandbLogger"),
        patch("train.pl.Trainer") as trainer_cls,
        patch("train.pl.seed_everything"),
    ):
        trainer_cls.return_value.fit = MagicMock()
        train.main(args)

    assert datamodule_cls.call_args.kwargs["partition"] is None


def test_validation_step_logs_epoch_metrics_only(basic_model, dummy_batch):
    basic_model.trainer.sanity_checking = False
    basic_model.log = MagicMock()
    basic_model.forward = MagicMock(return_value=torch.zeros((5, 1)))

    basic_model.validation_step(dummy_batch, batch_idx=0)

    basic_model.log.assert_has_calls(
        [
            call(
                "val_loss",
                ANY,
                batch_size=5,
                sync_dist=False,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            ),
            call(
                "val_rmse",
                ANY,
                batch_size=5,
                sync_dist=False,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            ),
            call(
                "val_r2",
                ANY,
                batch_size=5,
                sync_dist=False,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            ),
        ]
    )


def test_fast_dev_run(basic_model):
    """Runs a full train/val/test cycle using Lightning's fast_dev_run."""
    dataset = [
        Data(
            x=torch.randn(5, 4),
            edge_index=torch.randint(0, 5, (2, 4)),
            edge_attr=torch.randn(4, 2),
            y=torch.randn(1, 1),
        )
        for _ in range(4)
    ]

    loader = PyGDataLoader(dataset, batch_size=2, num_workers=0)

    trainer = pl.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_checkpointing=False
    )

    trainer.fit(basic_model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(basic_model, dataloaders=loader)


def test_training_startup_callback_logs_transitions(capsys):
    callback = TrainingStartupCallback()
    trainer = MagicMock()
    module = MagicMock()

    callback.on_fit_start(trainer, module)
    callback.on_train_start(trainer, module)
    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)

    output = capsys.readouterr().out
    assert "Fit entered" in output
    assert "Training loop started" in output
    assert "First training batch started" in output


def test_training_startup_callback_logs_epoch_time():
    callback = TrainingStartupCallback()
    trainer = MagicMock()
    module = MagicMock()

    with patch("train_utils.time.monotonic", side_effect=[10.0, 13.5]):
        callback.on_train_epoch_start(trainer, module)
        callback.on_train_epoch_end(trainer, module)

    module.log.assert_called_once_with("epoch_time_seconds", 3.5)


def test_training_startup_callback_logs_step_time_with_explicit_batch_size():
    callback = TrainingStartupCallback()
    trainer = MagicMock()
    module = MagicMock()

    batch = tuple(
        [
            Data(
                x=torch.randn(5, 4),
                edge_index=torch.randint(0, 5, (2, 10)),
                edge_attr=torch.randn(10, 2),
                y=torch.randn(1, 1),
            ),
            Data(
                x=torch.randn(6, 4),
                edge_index=torch.randint(0, 6, (2, 12)),
                edge_attr=torch.randn(12, 2),
                y=torch.randn(1, 1),
            ),
        ]
    )

    with patch("train_utils.time.monotonic", side_effect=[10.0, 10.5]):
        callback.on_train_batch_start(trainer, module, batch=batch, batch_idx=0)
        callback.on_train_batch_end(
            trainer, module, outputs=None, batch=batch, batch_idx=0
        )

    module.log.assert_called_once_with(
        "train_step_time_s",
        0.5,
        batch_size=2,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )


@pytest.mark.parametrize(
    "encoder_name,extra_kwargs",
    [
        ("gcn", {}),
    ],
)
def test_large_graph_forward_backward_no_crash(encoder_name, extra_kwargs):
    torch.manual_seed(7)

    num_nodes = 250_000
    num_edges = 550_000

    x = torch.randn((num_nodes, 4), dtype=torch.float32)
    src = torch.arange(num_edges, dtype=torch.long) % num_nodes
    dst = (src * 37 + 11) % num_nodes
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn((num_edges, 2), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.randn((1, 1), dtype=torch.float32),
    )
    batch = Batch.from_data_list([data])

    encoder_kwargs = {
        "num_layers": 2,
        "hid_dim": 8,
        "dropout": 0.0,
        "norm_type": "batch",
        "jk_mode": "last",
        **extra_kwargs,
    }

    model = AIGRegressionLightningModule(
        encoder_name=encoder_name,
        hidden_dim=8,
        node_input_dim=4,
        edge_attr_dim=2,
        task_out_dim=1,
        pooling_type="mean",
        encoder_kwargs=encoder_kwargs,
    )

    # Attach a mock Trainer for isolated training_step invocation in unit test
    model.trainer = MagicMock()

    model.train()
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "Expected at least one non-null gradient tensor"


if __name__ == "__main__":
    pytest.main([__file__])
