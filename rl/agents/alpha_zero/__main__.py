from argparse import ArgumentParser
from time import sleep
from random import randrange

import numpy as np
from torch import nn, optim, jit
from torch.multiprocessing import Queue

from rl.simulators import TicTacToe, Simulator
from .config import Config
from .learner_worker import LearnerWorker
from .self_play_worker import SelfPlayWorker
from .loggers import LearnerLogger, SelfPlayLogger
from .mcts import mcts
from .node import Node
from .networks import TicTacToeNetwork



def play(simulator: Simulator, network: nn.Module, config: Config):
    step = randrange(2)

    state, mask = simulator.reset()
    terminal = False
    root: Node = None

    while not terminal:
        step += 1
        simulator.render(state)

        if step % 2 == 0:
            action = int(input("Action: "))
        else:
            root = mcts(
                state,
                mask,
                simulator,
                network,
                config,
                root_node=root,
                simulations=config.simulations,
            )
            action = np.random.choice(mask.shape[0], p=root.action_policy)

        state, mask, reward, terminal, _ = simulator.step(state, action)
        if root is not None:
            root = root.children[action]

    simulator.render(state)


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.play:
        net = TicTacToeNetwork()
        train(
            4,
            TicTacToe,
            net,
            optim.SGD(net.parameters(), lr=1e-4, weight_decay=1e-5),
            Config(),
            args.save_path,
        )

    else:
        net = jit.load(args.save_path)
        play(TicTacToe, net, Config())
