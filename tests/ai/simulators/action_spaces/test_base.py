import pytest
import ai.simulators.action_spaces as action_spaces


def test_casting():
    space = action_spaces.Base

    with pytest.raises(RuntimeError):
        space.as_discrete()

    with pytest.raises(RuntimeError):
        space.as_type(action_spaces.TicTacToe)
