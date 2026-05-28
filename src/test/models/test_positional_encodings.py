import torch
from torch_geometric.data import Data

# Adjust imports based on your project structure
from models.layers.positional_encodings import get_pe_transform


def test_continuous_pe_log_scaling():
    """
    Test that massive continuous features like 'local_sp_sum' and 'pi_paths'
    are successfully squished using log1p and returned as floats.
    """
    # 1. Setup fake data with massive numbers
    # A mix of zeros (to ensure log(0) doesn't crash), small numbers, and huge outliers
    fake_local_sp_sum = torch.tensor([[0.0], [2.0], [50000.0], [1000000.0]])
    data = Data(x=torch.randn(4, 10), local_sp_sum=fake_local_sp_sum)

    # 2. Get the transform for local_sp_sum
    transform = get_pe_transform("local_sp_sum", attr_name="pos_enc")

    # Precompute expected values BEFORE the transform mutates the source tensor in-place
    expected_values = torch.log1p(fake_local_sp_sum.clone().float())

    # 3. Apply the transform
    data = transform(data)

    # 4. Verify the results
    assert hasattr(data, "pos_enc"), "pos_enc attribute was not attached"
    assert data.pos_enc.dtype == torch.float32, "Continuous PE should be float"
    assert not hasattr(data, "local_sp_sum"), "Source PE tensor should be removed"

    # Check the math (log1p(x) = ln(1 + x))

    # 50,000 should be squished down to ~10.8
    assert torch.allclose(data.pos_enc, expected_values), "Log scaling math failed!"
    assert data.pos_enc[2].item() < 12.0, "Massive value was not scaled down!"


def test_discrete_pe_no_scaling():
    """
    Test that discrete categorical features like 'level' (logic depth)
    are NOT log-scaled and are correctly cast to Long tensors for Embeddings.
    """
    # 1. Setup fake logic depths
    fake_levels = torch.tensor([[0], [1], [5], [12]])
    data = Data(x=torch.randn(4, 10), level=fake_levels)

    # 2. Get the transform for level
    transform = get_pe_transform("level", attr_name="pos_enc")

    # 3. Apply the transform
    data = transform(data)

    # 4. Verify the results
    assert hasattr(data, "pos_enc"), "pos_enc attribute was not attached"
    assert data.pos_enc.dtype == torch.int64, "Discrete PE should be a Long tensor"
    assert not hasattr(data, "level"), "Source PE tensor should be removed"

    # The values should remain exactly the same (no log scaling)
    assert torch.equal(data.pos_enc, fake_levels.long()), (
        "Discrete values were improperly modified!"
    )


def test_sinusoidal_pe():
    """
    Ensure the fallback sinusoidal PE works and attaches the correct shape.
    """
    data = Data(x=torch.randn(100, 10), num_nodes=100)
    transform = get_pe_transform("sinusoidal", attr_name="pos_enc", dim=16)

    data = transform(data)

    assert hasattr(data, "pos_enc")
    assert data.pos_enc.shape == (100, 16), "Sinusoidal PE attached the wrong shape"
    assert data.pos_enc.dtype == torch.float32
