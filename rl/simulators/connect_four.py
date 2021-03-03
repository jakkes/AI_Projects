from typing import Callable, Tuple, List, Dict

import numpy as np
from scipy.signal import convolve

from rl.utils.np_utils import cross_diag
from .simulator import Simulator, States, State


class ConnectFour(Simulator):
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

    @classmethod
    def reset_bulk(self, n: int) -> Tuple[States, np.ndarray]:
        states = np.zeros((n, 7 * 6 + 1))
        states[:, -1] = 1.0
        return (states,), np.ones((n, 7), dtype=np.bool_)

    @classmethod
    def _compute_rewards(cls, states: States, players: np.ndarray) -> np.ndarray:

        players = 4 * players.reshape((-1, 1, 1))

        win_horisontal = (
            convolve(states[0], np.ones((1, 1, 4)), mode="valid") == players
        )
        win_horisontal = np.any(np.any(win_horisontal, axis=2), axis=1)

        win_vertical = convolve(states[0], np.ones((1, 4, 1)), mode="valid") == players
        win_vertical = np.any(np.any(win_vertical, axis=2), axis=1)

        win_diagonal = (
            convolve(states[0], np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == players
        )
        win_diagonal = np.any(np.any(win_diagonal, axis=2), axis=1)

        win_xdiagonal = (
            convolve(states[0], cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == players
        )
        win_xdiagonal = np.any(np.any(win_xdiagonal, axis=2), axis=1)

        win = win_horisontal | win_vertical | win_diagonal | win_xdiagonal

        loss_horisontal = (
            convolve(states[0], np.ones((1, 1, 4)), mode="valid") == -players
        )
        loss_horisontal = np.any(np.any(loss_horisontal, axis=2), axis=1)

        loss_vertical = (
            convolve(states[0], np.ones((1, 4, 1)), mode="valid") == -players
        )
        loss_vertical = np.any(np.any(loss_vertical, axis=2), axis=1)

        loss_diagonal = (
            convolve(states[0], np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == -players
        )
        loss_diagonal = np.any(np.any(loss_diagonal, axis=2), axis=1)

        loss_xdiagonal = (
            convolve(states[0], cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid")
            == -players
        )
        loss_xdiagonal = np.any(np.any(loss_xdiagonal, axis=2), axis=1)

        loss = loss_horisontal | loss_vertical | loss_diagonal | loss_xdiagonal

        rewards = np.zeros(states[0].shape[0])
        rewards[win] = 1.0
        rewards[loss] = -1.0

        return rewards

    @classmethod
    def step_bulk(
        cls, states: States, actions: np.ndarray
    ) -> Tuple[States, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:

        next_states = states[0].copy()
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
        rewards = cls._compute_rewards(next_states, players)
        terminals = (rewards != 0) | np.all(heights == 6, axis=1)

        next_masks = heights < 6
        next_states = next_states.reshape((-1, 6 * 7))
        next_states = np.concatenate((next_states, -players.reshape((-1, 1))), axis=1)

        return (
            (next_states,),
            next_masks,
            rewards,
            terminals,
            [{} for _ in range(next_states.shape[0])],
        )

    @classmethod
    def render(cls, state: State, output_fn: Callable[[str]] = print):
        """Renders the game board to a string and then outputs it to the given
        output function.

        Args:
            state (State): State to render
            output_fn (Callable[[str]], optional): Output function, accepting one string
            argument. Defaults to `print`.
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
            output_fn(" | ".join(tile(state[0][7 * i + j]) for j in range(7)))

        output_fn(" | ".join(str(x) for x in range(7)))
        for i in range(6):
            print_line(i)
