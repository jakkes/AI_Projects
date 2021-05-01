from typing import Callable, Tuple, List, Dict
import numpy as np

import ai.simulators as simulators


_DIAG_INDICES = np.array([0, 4, 8]).astype(np.int32)
_CROSS_DIAG_INDICES = np.array([2, 4, 6]).astype(np.int32)


class TicTacToe(simulators.Base):

    """TicTacToe (connect three, or three in a row) simulator.

    States are given by a single `np.ndarray` of shape `(10, )`. The first 9 elements
    denote the game board in row-major order (board is of shape `(3, 3)`). Each board
    element is in `{-1, 0, 1}`, where `-1` and `1` denote occupied cells and `0` empty
    cells. The last element in the state vector (i.e. element index 9) is either `+1`
    or `-1`, denoting the player who is about to play.

    Actions discrete in `{0, 1, ..., 8}`, denoting, in row-major order, which cell to
    place the next marker in.

    Rewards are given at the end of a game round, i.e. intermediate rewards are zero.
    Then, if a winning action is rewarded with `+1` and a losing action is rewarded with
    `-1`."""

    class ActionSpace(simulators.action_spaces.Discrete):
        """Action space for the TicTacToe simulator."""

        @property
        def size(cls) -> int:
            return 9

        def action_mask_bulk(self, states: np.ndarray) -> np.ndarray:
            return states[:, :-1] == 0

    def __init__(self) -> None:
        super().__init__(True)
        self._action_space = TicTacToe.ActionSpace()

    @property
    def action_space(self) -> "TicTacToe.ActionSpace":
        return self._action_space

    def close(self):
        pass

    def reset_bulk(self, n: int) -> np.ndarray:
        states = np.zeros((n, 10))
        states[:, -1] = 1.0
        return states

    def _check_win(self, states: np.ndarray) -> np.ndarray:
        batchvec = np.arange(states.shape[0])
        repeated_batchvec = np.repeat(batchvec, 3)
        tiled_diag_indices = np.tile(_DIAG_INDICES, batchvec.shape[0])
        tiled_cross_diag_indices = np.tile(_CROSS_DIAG_INDICES, batchvec.shape[0])

        player = np.reshape(states[batchvec, -1], (-1, 1))
        own_marks = states[:, :-1] == player

        row_win = np.any(np.all(np.reshape(own_marks, (-1, 3, 3)), axis=2), axis=1)
        col_win = np.any(np.all(np.reshape(own_marks, (-1, 3, 3)), axis=1), axis=1)
        diagwin = np.all(
            np.reshape(own_marks[repeated_batchvec, tiled_diag_indices], (-1, 3)),
            axis=1,
        )
        crossdiagwin = np.all(
            np.reshape(own_marks[repeated_batchvec, tiled_cross_diag_indices], (-1, 3)),
            axis=1,
        )

        return row_win | col_win | diagwin | crossdiagwin

    def _check_loss(self, states: np.ndarray) -> np.ndarray:
        states = states.copy()
        states[:, -1] = -states[:, -1]
        return self._check_win(states)

    def step_bulk(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        states = states

        if np.any(self._check_loss(states)) or np.any(self._check_win(states)):
            raise ValueError("Cannot step from a state already in a win condition.")

        next_states = states.copy()
        batchvec = np.arange(next_states.shape[0])

        if np.any(next_states[batchvec, actions] != 0.0):
            raise ValueError("Cannot place a piece at an already occupied spot")

        next_states[batchvec, actions] = next_states[batchvec, -1]

        rewards = np.zeros(next_states.shape[0])
        win = self._check_win(next_states)
        rewards[win] = 1.0
        loss = self._check_loss(next_states)
        rewards[loss] = -1.0

        terminals = win | loss | np.all(next_states != 0, axis=1)

        next_states[batchvec, -1] = -states[batchvec, -1]

        return (
            next_states,
            rewards,
            terminals,
            [{} for _ in range(next_states.shape[0])],
        )

    def render(self, state: np.ndarray, output_fn: Callable[[str], None] = print):
        """Renders the game board and action index map to a string that is then output
        through the given output function.

        Args:
            state (np.ndarray): State to render
            output_fn (Callable[[str], None], optional): Output function, called with
            the generated string. Defaults to `print`.
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

        output_fn(
            f"""
| --- | --- | --- |         | --- | --- | --- |
|  {tile(state[0])}  |  {tile(state[1])}  |  {tile(state[2])}  |     |  0  |  1  |  2  |
| --- | --- | --- |         | --- | --- | --- |
|  {tile(state[3])}  |  {tile(state[4])}  |  {tile(state[5])}  |     |  3  |  4  |  5  |
| --- | --- | --- |         | --- | --- | --- |
|  {tile(state[6])}  |  {tile(state[7])}  |  {tile(state[8])}  |     |  6  |  7  |  8  |
| --- | --- | --- |         | --- | --- | --- |
        """
        )
