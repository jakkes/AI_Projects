from typing import Sequence, Tuple, List, Dict

import numpy as np

import ai.simulators as simulators


class ActionSpace(simulators.action_spaces.Discrete):
    def __init__(self, dim: int, sizes: Sequence[int]):
        super().__init__()
        self._dim = dim
        self._sizes = np.array(sizes).reshape((1, -1))

    @property
    def size(self) -> int:
        return self._dim * 2

    def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
        i = np.arange(self._dim)
        re = np.ones((states.shape[0], self.size), dtype=np.bool_)
        re[:, 2*i] = states[:, 0] + 1 < self._sizes
        re[:, 2*i + 1] = states[:, 0] > 0


class Grid(simulators.Base):
    """Simple grid navigation environment. Agents can move in either direction in all
    dimensions and need to reach a goal state.

    States are given by two vectors, specifying the grid coordinates of the agent and
    goal respectively. A grid of dimension `N` therefore has a state shape of
    `(2, N)`. On reset, start state and goal state are sampled uniformly across the
    grid. Steps can be taken in one dimension only, and the action space is discrete
    with size `2N`. Action indices `2k` moves the agent in the positive direction of
    dimension `k`, and `2k+1` in the negative direction. Rewards are given as 1 whenever
    the goal is reached, otherwise 0."""

    def __init__(self, dim: int, sizes: Sequence[int]):
        super().__init__(True)

        if len(sizes) != dim:
            raise ValueError("Need to provide one size per dimension.")

        self._dim = dim
        self._sizes = sizes
        self._action_space = ActionSpace(dim)

    @property
    def action_space(self) -> simulators.action_spaces.Discrete:
        return self._action_space

    def reset_bulk(self, n: int) -> np.ndarray:
        positions = np.product(self._sizes)
        start_states = np.random.randint(0, positions, size=n)
        goal_states = np.random.randint(0, positions, size=n)

        while np.any(goal_states == start_states):
            goal_states[goal_states == start_states] = np.random.randint(
                0, positions, size=(goal_states == start_states).sum()
            )
        
        return np.stack((start_states, goal_states), axis=1)

    def step_bulk(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        nvec = np.arange(actions.shape[0])
        if np.any(self.action_space.action_mask_bulk(states)[nvec, actions]):
            raise ValueError("Cannot execute an illegal action.")
        dim = actions // 2
        increment = - ((actions % 2) * 2 - 1)
        next_states = states.copy()
        next_states[nvec, dim] += increment

        terminals = np.all(next_states[:, 0] == next_states[:, 1], axis=1)
        rewards = terminals.astype(np.float32)

        return next_states, rewards, terminals, [{} for _ in nvec]
