from __future__ import annotations
from typing import List, Optional

import torch
from torch import nn

import numpy as np
from rl.simulators import Simulator

import rl.agents.alpha_zero._core.mcts as mcts


class MCTSNode:
    """Node in the MCTS algorithm."""

    def __init__(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        simulator: Simulator,
        network: nn.Module,
        parent: MCTSNode = None,
        config: mcts.MCTSConfig = None,
        action: int = None,
        reward: float = None,
        terminal: bool = None,
    ):
        """
        Args:
            state (np.ndarray): State of this node.
            action_mask (np.ndarray): Action mask of the state.
            simulator (Simulator): Simulator used in the roll out
            network (nn.Module): Network used in the roll out
            parent (MCTSNode, optional): Parent node. Defaults to None.
            config (MCTSConfig, optional): Configuration of the MCTS. Defaults to None.
            action (int, optional): Action that led to this node. Defaults to None.
            reward (float, optional): Reward obtained upon transitioning to this node.
            Defaults to None.
            terminal (bool, optional): Whether or not this node is in a terminal state.
            Defaults to None.
        """
        self._state = state
        self._action_mask = action_mask
        self._action = action
        self._reward = reward
        self._terminal = terminal
        self._simulator = simulator
        self._network = network
        self._config = config
        self._parent = parent
        self._P = None
        self._N = None
        self._W = None
        self._V = None
        self._children: List[MCTSNode] = None
        self._expanded: bool = False

        if not np.any(self._action_mask) and not self._terminal:
            print("Stop")

    @property
    def is_leaf(self) -> bool:
        """True if this node is a leaf node."""
        return self._children is None

    @property
    def is_root(self) -> bool:
        """True if this node is the root."""
        return self._parent is None

    @property
    def is_terminal(self) -> bool:
        """True if this node is a terminal state."""
        return self._terminal

    @property
    def parent(self) -> Optional[MCTSNode]:
        """The parent of this node."""
        return self._parent

    @property
    def children(self) -> Optional[List[Optional[MCTSNode]]]:
        """The children of this node. If this node has not been expanded, then
        `None` is returned, otherwise the list of (possible children) is returned. If
        not None, the list consists of `MCTSNode`s on indices representing legal
        actions. Illegal action indices are `None`. In other words, the child at index
        `i` corresponds to the node retrieved by action `i`.
        """
        return self._children

    @property
    def state(self) -> np.ndarray:
        """The state of this node."""
        return self._state

    @property
    def action(self) -> Optional[int]:
        """The action that led to this node. If this node is the root, then this value
        is `None`."""
        return self._action

    @property
    def reward(self) -> Optional[bool]:
        """The reward obtained on the transition into this node. If this node is the
        root, then this value is `None`."""
        return self._reward

    @property
    def action_mask(self) -> np.ndarray:
        """The action mask of this node."""
        return self._action_mask

    @property
    def action_policy(self) -> np.ndarray:
        """Action policy calculated in this node."""
        distribution = np.power(self._N, 1 / self._config.T)
        return distribution / np.sum(distribution)

    def select(self) -> MCTSNode:
        """Traverses one step in the tree from this node according to the selection
        policy.

        Raises:
            ValueError: If this node is in a terminal state.

        Returns:
            MCTSNode: The node selected.
        """
        if self.is_terminal:
            raise ValueError("Cannot select action from terminal state.")

        Q = np.zeros_like(self._N)
        mask = self._N > 0
        Q[mask] = self._W[mask] / self._N[mask]
        if np.any(self._N > 0):
            U = self._P * np.sqrt(np.sum(self._N)) / (1 + self._N)
        else:
            U = self._P
        QU = Q + self._config.c * U
        QU[~self._action_mask] = -np.inf
        return self._children[np.argmax(QU)]

    def expand(self):
        """Expands the node. If the node has already been expanded, then this is a
        no-op.
        """

        if self._expanded:
            return

        self._init_pv()

        if not self.is_terminal:
            actions = np.arange(self._action_mask.shape[0])[self._action_mask]
            states = np.expand_dims(self._state, 0)
            states = np.repeat(states, actions.shape[0], axis=0)
            next_states, next_masks, rewards, terminals, _ = self._simulator.step_bulk(
                states, actions
            )

            self._children = [None] * self._action_mask.shape[0]
            for next_state, next_mask, reward, terminal, action in zip(
                next_states, next_masks, rewards, terminals, actions
            ):
                self._children[action] = MCTSNode(
                    next_state,
                    next_mask,
                    self._simulator,
                    self._network,
                    parent=self,
                    config=self._config,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                )

            self._N = np.zeros(self._action_mask.shape[0])
            self._W = np.zeros(self._action_mask.shape[0])

        self._expanded = True

    def backup(self):
        """Runs the backpropagation from this node up to the root."""
        if self.is_root:
            return

        if self.is_terminal:
            self.parent._backpropagate(self._action, self._reward)
        else:
            self.parent._backpropagate(self._action, -self._V)

    def _backpropagate(self, action: int, value: float):
        self._N[action] += 1
        self._W[action] += value

        if self.parent is not None:
            self.parent._backpropagate(self.action, -value)

    def _init_pv(self):
        with torch.no_grad():
            p, v = self._network(
                torch.as_tensor(self._state, dtype=torch.float).unsqueeze_(0),
                torch.as_tensor(self._action_mask, dtype=torch.bool).unsqueeze_(0),
            )
        self._P = torch.softmax(p, dim=1).squeeze_(0).numpy()
        self._V = v[0, 0].numpy()

        if self.is_root:
            self.add_noise()

    def add_noise(self):
        """Adds dirchlet noise to the prior probability."""
        d = np.random.dirichlet(
            self._config.alpha * np.ones(self._action_mask.shape[0])[self._action_mask]
        )
        self._P[self._action_mask] = (1 - self._config.epsilon) * self._P[
            self._action_mask
        ] + self._config.epsilon * d

    def rootify(self):
        """Converts this node to a root node, cutting ties with all parents, while
        maintaining its children."""

        if self.is_terminal:
            raise ValueError("Cannot rootify a terminal state.")

        self._parent = None
        self._action = None
        self._reward = None
        self._terminal = False

        if self._P is not None:
            self.add_noise()
