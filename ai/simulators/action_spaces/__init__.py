"""Action spaces for simulators."""

from ._base import Base
from ._discrete import Discrete
from ._connect_four import ConnectFour
from ._tic_tac_toe import TicTacToe


__all__ = ["Base", "Discrete", "ConnectFour", "TicTacToe"]
