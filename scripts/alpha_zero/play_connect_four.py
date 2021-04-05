import os
from argparse import ArgumentParser
from random import randrange

import numpy as np
from torch import nn, jit

import ai.simulators as simulators
import ai.agents.alpha_zero as alpha_zero


parser = ArgumentParser()
parser.add_argument("--save-path", type=str, default=None)


def play(
    simulator: simulators.Factory, network: nn.Module, config: alpha_zero.MCTSConfig
):
    step = randrange(2)
    simulator = simulator()

    state = simulator.reset()
    mask = simulator.action_space.as_discrete.action_mask(state)
    terminal = False
    root: alpha_zero.MCTSNode = None

    while not terminal:
        step += 1
        print()
        simulator.render(state)

        if step % 2 == 0:
            action = int(input("Action: "))
        else:
            root = alpha_zero.mcts(
                state, mask, simulator, network, config, root_node=root
            )
            action = np.random.choice(mask.shape[0], p=root.action_policy)

        state, _, terminal, _ = simulator.step(state, action)
        mask = simulator.action_space.as_discrete.action_mask(state)
        if root is not None:
            root = root.children[action]

    simulator.render(state)


if __name__ == "__main__":
    args = parser.parse_args()

    net = jit.load(os.path.join(args.save_path, "network.pt"))
    play(simulators.ConnectFour(), net, alpha_zero.MCTSConfig())
