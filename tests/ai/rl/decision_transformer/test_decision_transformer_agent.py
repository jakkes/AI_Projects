import torch

from ai.rl.decision_transformer import TransformerEncoder
from ai.rl.decision_transformer._agent import interleave, evaluate_transformer, decode


def test_interleave():
    rtgs = torch.rand(5, 5, 3)
    states = torch.rand(5, 5, 3)
    actions = torch.rand(5, 4, 3)

    out = interleave(rtgs, states, actions)

    assert out.shape == (5, 5+5+4, 3)

    assert out[1, 0, 1] == rtgs[1, 0, 1]
    assert out[0, 3, 2] == rtgs[0, 1, 2]

    assert out[0, 1, 0] == states[0, 0, 0]
    assert out[4, 4, 0] == states[4, 1, 0]

    assert out[0, 2, 2] == actions[0, 0, 2]
    assert out[2, 5, 2] == actions[2, 1, 2]


def test_evaluate_transformer():
    model = TransformerEncoder(4, 4, 8, 3, 2, 50)

    data = torch.randn(5, 32, 8)
    lengths = torch.randint(20, 32, (5, ))

    out = evaluate_transformer(model, data, lengths)
    print(out)


def test_decode():
    model = torch.nn.Identity()

    x = torch.randn(3, 50)
    l = torch.tensor([1, 2, 3])
    
    y = decode(model, x, l)

    assert y[0] == x[0, 1]
    assert y[1] == x[1, 4]
    assert y[2] == x[2, 7]
