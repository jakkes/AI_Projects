import pytest
import numpy as np
from torch import nn

import ai.rl.a3c as a3c


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(4, 2)
        self.v = nn.Linear(4, 1)

    def forward(self, x):
        return self.p(x), self.v(x)


def test_agent():
    agent = a3c.Agent(Net())
    agent.act(np.random.random((4, )), np.ones((2, ), dtype=np.bool_))
    with pytest.raises(ValueError):
        agent.act(np.random.random((4, )), np.zeros((2, ), dtype=np.bool_))
