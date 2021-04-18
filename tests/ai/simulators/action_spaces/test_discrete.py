import pytest
import ai.simulators as simulators
import ai.simulators.action_spaces as action_spaces


def test_casting():
    space = simulators.TicTacToe.ActionSpace()

    space = space.as_discrete
    space.cast_to(action_spaces.Discrete)

    with pytest.raises(RuntimeError):
        space.cast_to(simulators.ConnectFour.ActionSpace)
