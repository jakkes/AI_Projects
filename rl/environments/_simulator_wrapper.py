from typing import Tuple, Dict

import numpy as np

import rl.simulators as simulators
import rl.environments as environments


class SimulatorWrapper(environments.Base):
    """Environment that wraps a simulator, exposing it as an environment."""

    def __init__(self, simulator: simulators.Base):
        """
        Args:
            simulator (simulators.Base): Simulator class to wrap into an environment.
        """
        super().__init__()
        self._simulator = simulator
        self._ready = False
        self._state: np.ndarray = None
        self._action_mask: np.ndarray = None

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        if not self._ready:
            raise ValueError(
                "Must call 'reset()' after having received a terminal state"
            )

        self._state, self._action_mask, reward, terminal, debug = self._simulator.step(
            self._state, action
        )
        if terminal:
            self._ready = False
        return self._state, self._action_mask, reward, terminal, debug

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._state, self._action_mask = self._simulator.reset()
        self._ready = True
        return self._state, self._action_mask
