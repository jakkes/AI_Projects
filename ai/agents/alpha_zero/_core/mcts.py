from torch import nn
import numpy as np

import ai.simulators as simulators
from .mcts_node import MCTSNode


class MCTSConfig:
    """Configuration for the Monte Carlo Tree Search."""

    def __init__(self) -> None:
        self.c = 1.25
        """Exploration coefficient. Larger value results in more emphasis on exploration
        in the Monte Carlo Tree Search."""

        self.T = 0.1
        """Policy temperature."""

        self.alpha = 3
        """Affects the dirchlet noise added to the root used in the tree search."""

        self.epsilon = 0.25
        """Coefficient governing the impact of dirchlet noise."""

        self.simulations = 200
        """Number of MCTS steps to run per state evaluation."""

        self.zero_sum_game: bool = True
        """If True, the simulator is assumed to be a zero sum game where players
        alternate moves. Rewards are zero until reaching a terminal state."""

        self.discount_factor: float = 0.99
        """Discount factor, used only if `zero_sum_game` is False."""


def mcts(
    state: np.ndarray,
    action_mask: np.ndarray,
    simulator: simulators.Base,
    network: nn.Module,
    config: MCTSConfig,
    root_node: MCTSNode = None,
) -> MCTSNode:
    """Runs the Monte Carlo Tree Search algorithm.

    Args:
        state (np.ndarray): Start state.
        action_mask (np.ndarray): Start action mask.
        simulator (Simulator): Simulator.
        network (nn.Module): Network.
        config (MCTSConfig): Configuration.
        simulations (int, optional): Number of MCTS steps. Defaults to 50.
        root_node (MCTSNode, optional): If not None, this node is used as root. Useful
            when the tree has previously been traversed, i.e. previously computed
            children are maintained instead of erasing the already computed tree.
            Defaults to None.

    Returns:
        MCTSNode: Root node.
    """

    root = (
        MCTSNode(state, action_mask, simulator, network, config=config)
        if root_node is None
        else root_node
    )
    if not np.array_equal(state, root.state):
        raise ValueError("Given state and state of the root node differ.")
    if not np.array_equal(action_mask, root.action_mask):
        raise ValueError("Given action mask and action mask of the root node differ.")
    root.rootify()

    for _ in range(config.simulations):
        node = root
        while not node.is_leaf:
            node = node.select()

        node.expand()
        node.backup()
    return root
