import torch
import ai.rl.decision_transformer._transformer as tr


def test_attention():
    model = tr.Attention(10, 5, 3)
    out = model(torch.rand(2, 5, 10), torch.zeros(2, 5, 5, dtype=torch.bool))
    assert out.shape == (2, 5, 3)
    out = model(torch.rand(2, 5, 10), torch.ones(2, 5, 5, dtype=torch.bool))
    assert out.shape == (2, 5, 3)
    out = model(torch.rand(2, 5, 10), torch.randint(0, 2, (2, 5, 5), dtype=torch.bool))
    assert out.shape == (2, 5, 3)


def test_multiheadattention():
    model = tr.MultiHeadAttention(8, 10, 5, 3)
    out = model(torch.rand(2, 5, 10))
    assert out.shape == (2, 5, 10)


def test_transformer_encoder():
    model = tr.TransformerEncoder(5, 8, 10, 5, 3, 32)
    out = model(torch.rand(2, 5, 10))
    assert out.shape == (2, 5, 10)

    out = model(torch.rand(2, 5, 10), mask=torch.rand(2, 5, 5) > 0.5)
    assert out.shape == (2, 5, 10)
