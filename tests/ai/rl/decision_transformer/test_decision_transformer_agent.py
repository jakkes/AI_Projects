import torch
from torch import nn

from ai.utils import Factory
from ai.rl.decision_transformer import TransformerEncoder, Agent
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

    out = evaluate_transformer(model, data)
    print(out)


def test_decode():
    model = torch.nn.Identity()

    x = torch.randn(3, 50)
    l = torch.tensor([1, 2, 3])
    
    y = decode(model, x, l)

    assert y[0] == x[0, 1]
    assert y[1] == x[1, 4]
    assert y[2] == x[2, 7]


def test_agent():
    agent = Agent(
        Factory(nn.Linear, 2, 4),
        Factory(nn.Linear, 1, 4),
        Factory(nn.Linear, 1, 4),
        Factory(nn.Linear, 1, 4),
        Factory(TransformerEncoder, 2, 5, 4, 3, 2, 16),
        Factory(nn.Linear, 4, 1)
    )

    states = torch.randn(5, 32, 2)
    actions = torch.randn(5, 31, 1)
    reward_to_gos = torch.randn(5, 32)
    time_steps = torch.randn(5, 32)
    lengths = torch.tensor([5, 3, 1, 8, 2])

    agent.evaluate(states, actions, reward_to_gos, time_steps, lengths)

    torch.autograd.set_detect_anomaly(True)

    loss_fn = nn.MSELoss()
    first_loss = agent.loss(states, actions, reward_to_gos, time_steps, lengths, loss_fn)

    opt = torch.optim.Adam(agent.model.parameters())
    for _ in range(10):
        loss = agent.loss(states, actions, reward_to_gos, time_steps, lengths, loss_fn)
        assert not loss.isnan()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss < first_loss
