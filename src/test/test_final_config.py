import copy
import csv
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

import config
from data.data_utils import aig_to_pytorch_geometric
from data.datamodule import AIGDataModule
from models.lightning_model import AIGRegressionLightningModule


class GradientVerificationCallback(pl.Callback):
    """Snapshots weights right before the optimizer steps, and verifies the active ones changed."""
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Weights are initialized and gradients computed, but weights haven't updated yet.
        self.initial_weights = {
            name: p.clone().detach() 
            for name, p in pl_module.named_parameters() if p.requires_grad
        }

    def on_train_end(self, trainer, pl_module):
        updated_layers = 0
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                if torch.equal(self.initial_weights[name], p):
                    print(f"\n[Info] {name} did not update (likely bypassed by config like jk_mode='last').")
                else:
                    updated_layers += 1
                    
        assert updated_layers > 0, "Catastrophic gradient failure: NO layers were updated!"


def _mock_dataset(tmp_path: Path, algorithm: str) -> Path:
    """Clones adder.aig 10 times with injected feature noise."""
    aig_path = Path("src/test/data/adder.aig")
    assert aig_path.exists(), f"Dummy AIG missing at {aig_path}!"
    
    base_data = aig_to_pytorch_geometric(aig_path)
    pt_paths = []
    
    # Create 10 noisy copies for a healthy 8/1/1 data split
    for i in range(10):
        data = base_data.clone()
        if data.x.is_floating_point():
            data.x = data.x + torch.randn_like(data.x) * 0.05
        else:
            data.x = data.x.float() + torch.randn_like(data.x.float()) * 0.05
            
        pt = tmp_path / f"graph_{i}.pt"
        torch.save(data, pt)
        pt_paths.append(pt)
        
    csv_path = tmp_path / f"dummy_{algorithm}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["unoptimized_graph_path", "design", "algorithm", "tier_id", "optimizability"]
        )
        writer.writeheader()
        for i, pt in enumerate(pt_paths):
            writer.writerow({
                "unoptimized_graph_path": str(pt),
                "design": "adder",
                "algorithm": algorithm,
                "tier_id": "1",
                "optimizability": 0.5 + (i * 0.01),
            })
    return csv_path


@pytest.mark.parametrize("algorithm", config.VALID_ALGORITHMS)
def test_final_training_pipeline_fast_dev_run(algorithm: str, tmp_path: Path):
    """Runs a forward/backward pass and verifies gradient flow using a callback."""
    csv_path = _mock_dataset(tmp_path, algorithm)
    
    encoder_kwargs = copy.deepcopy(config.ENCODER_KWARGS_DEFAULTS)
    encoder_kwargs.update({
        "num_layers": config.NUM_LAYERS,
        "hid_dim": config.HIDDEN_DIM,
        "dropout": config.DROPOUT,
        "norm_type": config.NORM_TYPE,
        "jk_mode": config.JK_MODE,
    })
    if config.ENCODER_NAME in ["transformer_conv", "graphgps"]:
        encoder_kwargs["heads"] = config.HEADS

    model = AIGRegressionLightningModule(
        encoder_name=config.ENCODER_NAME,
        hidden_dim=config.HIDDEN_DIM,
        pe_type=config.PE_TYPE,
        pos_enc_dim=config.POS_ENC_DIM if config.PE_TYPE != "none" else 0,
        pooling_type=config.POOLING_TYPE,
        encoder_kwargs=encoder_kwargs,
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        huber_delta=config.HUBER_DELTA,
    )

    datamodule = AIGDataModule(
        csv_paths=[str(csv_path)],
        positional_encoding=config.PE_TYPE if config.PE_TYPE != "none" else None,
        batch_size=2,
        num_workers=0,
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        callbacks=[GradientVerificationCallback()]
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
