import torch
from torch import nn

import ai.utils.torch.nn as ai_nn


class CartPole(nn.Module):
    """Example implementation of a network for the CartPole environment."""

    def __init__(
        self,
        use_distributional: bool = False,
        n_atoms: int = None,
        std_init: float = 0.5,
    ):
        """
        Args:
            use_distributional (bool, optional): If True, the network outputs in format
                required by distributional DQN. Defaults to False.
            n_atoms (int, optional): Number of atoms to use in distributional DQN.
                Required if distributional DQN is used, otherwise has no effect.
            std_init (float, optional): Initial standard deviation of noise in
                NoisyNets. Defaults to 0.5. Set to 0.0 to disable NoisyNets.
        """
        super().__init__()
        self._body = nn.Sequential(
            ai_nn.NoisyLinear(4, 128, std_init),
            nn.ReLU(inplace=True),
            ai_nn.NoisyLinear(128, 2 * n_atoms if use_distributional else 2, std_init)
        )
        self._use_distributional = use_distributional
        self._n_atoms = n_atoms

    def forward(self, x):
        x = self._body(x)
        if self._use_distributional:
            x = torch.softmax(x.view(-1, 2, self._n_atoms), dim=-1)
        return x
