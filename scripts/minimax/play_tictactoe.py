import random

import tap
import numpy as np

import ai.simulators as simulators
import ai.agents.minimax as minimax


class ArgumentParser(tap.Tap):
    search_depth: int = 5


class Heuristic(minimax.Heuristic):
    def __call__(self, state: np.ndarray) -> float:
        return 0.0


def main(args: ArgumentParser):
    simulator = simulators.TicTacToe()
    i = random.randrange(2)

    state = simulator.reset()
    terminal = False

    while not terminal:
        simulator.render(state)

        if i % 2 == 0:
            action = minimax.minimax(state, simulator, args.search_depth, Heuristic(), True)
        else:
            action = int(input("Action: "))
        state, _, terminal, _ = simulator.step(state, action)
        i += 1


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
