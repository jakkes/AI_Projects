from typing import Tuple, Dict

import numpy as np

import ai.simulators as simulators
import ai.environments as environments


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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if not self._ready:
            raise ValueError(
                "Must call 'reset()' after having received a terminal state"
            )

        self._state, reward, terminal, debug = self._simulator.step(
            self._state, action
        )
        if terminal:
            self._ready = False
        return self._state, self._action_mask, reward, terminal, debug

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._state, self._action_mask = self._simulator.reset()
        self._ready = True
        return self._state, self._action_mask

    def close(self):
        self._simulator.close()

    @classmethod
    def get_factory(cls, factory: simulators.Factory) -> environments.Factory:
        """Wraps a simulator factory into an environment factory."""
        return Factory(factory)


class Factory(environments.Factory):
    def __init__(self, factory: simulators.Factory):
        self._factory = factory

    def __call__(self) -> SimulatorWrapper:
        return SimulatorWrapper(self._factory())
