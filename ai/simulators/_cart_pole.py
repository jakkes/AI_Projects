import math
from typing import Tuple, List, Dict

import numpy as np

import ai.simulators as simulators


class CartPole(simulators.Base):
    """Simulator implementation of the inverted pendulum (aka CartPole).

    State is given by a vector in R^5 with elements indicating cart position, cart
    velocity, pole angular deviation from center, pole angular velocity, and steps made.

    Action space is discrete and of size two. Action 0 applies a force in the positive
    direction (positive relative to cart position), and action 1 negative.

    Rewards are given as 1 for each step. Episodes are terminated when either 200 steps
    have passed, or the cart position is greater than 2.4 from the start location, or
    the pole deviation is greater than 12 degrees.

    This simulator is based on https://github.com/openai/gym/blob/a5a6ae6bc0a5cfc0ff1ce9be723d59593c165022/gym/envs/classic_control/cartpole.py     # noqa
    """

    class ActionSpace(simulators.action_spaces.Discrete):
        @property
        def size(self) -> int:
            return 2

        def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
            return np.ones((states.shape[0], 2), dtype=np.bool_)

    def __init__(self) -> None:
        super().__init__(True)

        self._gravity = 9.8
        self._masscart = 1.0
        self._masspole = 0.1
        self._total_mass = self.masspole + self.masscart
        self._length = 0.5  # actually half the pole's length
        self._polemass_length = self.masspole * self.length
        self._force_mag = 10.0
        self._tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self._theta_threshold_radians = 12 * 2 * math.pi / 360
        self._x_threshold = 2.4

        self._action_space = CartPole.ActionSpace()

    @property
    def action_space(self) -> "simulators.action_spaces.Discrete":
        return self._action_space

    def reset_bulk(self, n: int) -> np.ndarray:
        states = np.random.uniform(low=-0.05, high=0.05, size=(n, 5))
        states[:, 4] = 0

    def step_bulk(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        if np.any(actions > 1) or np.any(actions < 0):
            raise ValueError("Invalid action given. All actions must be either 0 or 1.")

        force = self._force_mag * np.ones(actions.shape[0])
        force[actions == 0] *= -1
        costheta = np.cos(states[:, 2])
        sintheta = np.sin(states[:, 2])

        temp = (
            force + self._polemass_length + states[:, 3] ** 2 * sintheta
        ) / self._total_mass
        thetaacc = self._gravity * sintheta - costheta * temp / (
            self._length
            * (4.0 / 3.0 - self._masspole * costheta ** 2 / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * costheta / self._total_mass

        next_states = states.copy()
        next_states[:, 0] += self._tau * next_states[:, 1]
        next_states[:, 1] += self._tau * xacc
        next_states[:, 2] += self._tau * next_states[:, 3]
        next_states[:, 3] += self._tau * thetaacc
        next_states[:, 4] += 1

        terminals = (
            next_states[:, 0]
            < -self._x_threshold | next_states[:, 0]
            > self._x_threshold | next_states[:, 2]
            < -self._theta_threshold_radians | next_states[:, 2]
            > self._theta_threshold_radians
        )
        rewards = np.ones(actions.shape, dtype=np.float32)

        return next_states, rewards, terminals, [{} for _ in actions]
