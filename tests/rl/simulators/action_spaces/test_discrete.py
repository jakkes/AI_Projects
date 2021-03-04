import pytest
import rl.simulators.action_spaces as action_spaces


def test_casting():
    space = action_spaces.Discrete

    space.as_discrete()
    space.as_type(action_spaces.Discrete)

    with pytest.raises(RuntimeError):
        space.as_type(action_spaces.ConnectFour)
