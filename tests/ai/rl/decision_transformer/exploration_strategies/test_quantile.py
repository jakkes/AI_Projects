import torch

import ai.rl.decision_transformer.exploration_strategies as es


def test_quantile():
    estimator = es.Quantile(0.9)
    for _ in range(10000):
        estimator.update(torch.randn(100, 1))
    assert abs(estimator.reward_to_go() - 1.282) < 0.1
