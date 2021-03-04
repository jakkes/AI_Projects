import pytest
import ai.simulators.action_spaces as action_spaces


def test_casting():
    with pytest.raises(TypeError):
        action_spaces.Base()
