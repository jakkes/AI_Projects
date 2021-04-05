import pytest
import ai.simulators.action_spaces as action_spaces


def test_casting():
    space = action_spaces.TicTacToe()

    space = space.as_discrete
    space.cast_to(action_spaces.Discrete)

    with pytest.raises(RuntimeError):
        space.cast_to(action_spaces.ConnectFour)
