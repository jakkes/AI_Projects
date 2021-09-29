import torch
import ai.rl.decision_transformer as dt


def test_attention():
    model = dt.Attention(10, 5, 3)
    out = model(torch.rand(2, 5, 10), torch.rand(2, 5, 10), torch.rand(2, 5, 10))
    assert out.shape == (2, 5, 3)


def test_multiheadattention():
    model = dt.MultiHeadAttention(8, 10, 5, 3)
    out = model(
        torch.rand(2, 5, 10), torch.rand(2, 5, 10), torch.rand(2, 5, 10)
    )
    assert out.shape == (2, 5, 10)
