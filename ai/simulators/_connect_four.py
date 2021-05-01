from typing import Callable, Tuple, List, Dict

import numpy as np
from scipy.signal import convolve

import ai.utils.np as np_utils
import ai.simulators as simulators


class ConnectFour(simulators.Base):
    """Connect four (four in a row) game simulator.

    States are given by a single `np.ndarray` of shape `(43, )`. The first 42 elements
    denote the game board in row-major order (board is of shape `(6, 7)`). Each board
    element is in `{-1, 0, 1}`, where `-1` and `1` denote occupied cells and `0` empty
    cells. The last element in the state vector (i.e. element index 42) is either `+1`
    or `-1`, denoting the player who is about to play.

    Actions are discrete in `{0, 1, ..., 6}`, denoting, from the left, which column to
    place the next marker in.

    Rewards are given at the end of a game round, i.e. intermediate rewards are zero.
    Then, if a winning action is rewarded with `+1` and a losing action is rewarded with
    `-1`."""

    class ActionSpace(simulators.action_spaces.Discrete):
        """Action space of the ConnectFour simulator."""

        @property
        def size(cls) -> int:
            return 7

        def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
            return (states[:, :-1].reshape((-1, 6, 7)) != 0).sum(1) < 6

    def __init__(self) -> None:
        super().__init__(True)
        self._action_space = ConnectFour.ActionSpace()

    def reset_bulk(self, n: int) -> np.ndarray:
        states = np.zeros((n, 7 * 6 + 1))
        states[:, -1] = 1.0
        return states

    def close(self):
        pass

    @property
    def action_space(self) -> "ConnectFour.ActionSpace":
        return self._action_space

    def _compute_rewards(self, states: np.ndarray, players: np.ndarray) -> np.ndarray:

        players = 4 * players.reshape((-1, 1, 1))

        win_horisontal = convolve(states, np.ones((1, 1, 4)), mode="valid") == players
        win_horisontal = np.any(np.any(win_horisontal, axis=2), axis=1)

        win_vertical = convolve(states, np.ones((1, 4, 1)), mode="valid") == players
        win_vertical = np.any(np.any(win_vertical, axis=2), axis=1)

        win_diagonal = (
            convolve(states, np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == players
        )
        win_diagonal = np.any(np.any(win_diagonal, axis=2), axis=1)

        win_xdiagonal = (
            convolve(states, np_utils.cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == players
        )
        win_xdiagonal = np.any(np.any(win_xdiagonal, axis=2), axis=1)

        win = win_horisontal | win_vertical | win_diagonal | win_xdiagonal

        loss_horisontal = convolve(states, np.ones((1, 1, 4)), mode="valid") == -players
        loss_horisontal = np.any(np.any(loss_horisontal, axis=2), axis=1)

        loss_vertical = convolve(states, np.ones((1, 4, 1)), mode="valid") == -players
        loss_vertical = np.any(np.any(loss_vertical, axis=2), axis=1)

        loss_diagonal = (
            convolve(states, np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == -players
        )
        loss_diagonal = np.any(np.any(loss_diagonal, axis=2), axis=1)

        loss_xdiagonal = (
            convolve(states, np_utils.cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == -players
        )
        loss_xdiagonal = np.any(np.any(loss_xdiagonal, axis=2), axis=1)

        loss = loss_horisontal | loss_vertical | loss_diagonal | loss_xdiagonal

        rewards = np.zeros(states.shape[0])
        rewards[win] = 1.0
        rewards[loss] = -1.0

        return rewards

    def step_bulk(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:

        next_states = states.copy()
        batchvec = np.arange(next_states.shape[0])
        players = next_states[:, -1]

        next_states = next_states[:, :-1].reshape((-1, 6, 7))
        heights = (next_states != 0).sum(axis=1)

        if np.any(heights[batchvec, actions] >= 6):
            raise ValueError("Cannot place a piece in an already full column")

        next_states[batchvec, 5 - heights[batchvec, actions], actions] = players[
            batchvec
        ]
        heights[batchvec, actions] += 1
        rewards = self._compute_rewards(next_states, players)
        terminals = (rewards != 0) | np.all(heights == 6, axis=1)

        next_states = next_states.reshape((-1, 6 * 7))
        next_states = np.concatenate((next_states, -players.reshape((-1, 1))), axis=1)

        return (
            next_states,
            rewards,
            terminals,
            [{} for _ in range(next_states.shape[0])],
        )

    def render(self, state: np.ndarray, output_fn: Callable[[str], None] = print):
        """Renders the game board to a string and then outputs it to the given
        output function.

        Args:
            state (np.ndarray): State to render
            output_fn (Callable[[str], None], optional): Output function, accepting one
            string argument. Defaults to `print`.
        """

        def tile(value) -> str:
            if value == 0:
                return " "
            elif value == 1:
                return "X"
            elif value == -1:
                return "O"
            else:
                raise ValueError(f"Unexpected value {value}")

        def print_line(i: int):
            output_fn(" | ".join(tile(state[7 * i + j]) for j in range(7)))

        output_fn(" | ".join(str(x) for x in range(7)))
        for i in range(6):
            print_line(i)
