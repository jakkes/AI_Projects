"""Implementation of the AlphaZero algorithm, based on Monte Carlo Tree Search for
zero sum games.

The AlphaZero algorithm consists of two core components: the `LearnerWorker` and the
`SelfPlayWorker`. The `SelfPlayWorker`s generate rollouts of the policy that are then
passed onto the `LearnerWorker` where network updates are performed. These are
configurable by the `LearnerConfig` and `SelfPlayConfig`.

In addition to these, there are two logging servers used for visualization to
tensorboard: `LearnerLogger` and `SelfPlayLogger`.

For a basic implementation of the AlphaZero algorithm, see the `train` method. This
method uses pure python multiprocessing. However, replacing the python queues by some
other transportation protocol (implementing the queue interface) should allow the
algorithm to run across multiple machines as well. If running across machines, network
parameters must be communicated as well. In the `train` method, the network is simply
put into shared memory.

When evaluating a model, use the `mcts` method.
"""

from ._core.learner_worker import LearnerWorker, LearnerConfig
from ._core.self_play_worker import SelfPlayWorker, SelfPlayConfig
from ._core.loggers import LearnerLogger, SelfPlayLogger
from ._core.mcts import mcts, MCTSConfig
from ._core.mcts_node import MCTSNode
from ._train import train

__all__ = [
    "mcts",
    "LearnerWorker",
    "LearnerConfig",
    "SelfPlayWorker",
    "SelfPlayConfig",
    "LearnerLogger",
    "SelfPlayLogger",
    "MCTSConfig",
    "MCTSNode",
    "train",
]
