import numpy as np
from linq import Query

import ai.simulators as simulators
import ai.search.minimax as minimax


def _alphabeta(
    state: np.ndarray,
    simulator: simulators.Base,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    heuristic: minimax.Heuristic,
) -> float:
    if depth == 0:
        return heuristic(state)

    action_space = simulator.action_space.as_discrete
    actions = np.arange(action_space.size)[action_space.action_mask(state)]
    if maximizing_player:
        value = -np.inf
        for action in actions:
            next_state, reward, terminal, _ = simulator.step(state, action)
            if terminal:
                if reward > 0:
                    value = np.inf
                elif reward == 0:
                    value = 0
                else:
                    value = - np.inf
            else:
                value = max(value, _alphabeta(next_state, simulator, depth - 1, alpha, beta, False, heuristic))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.inf
        for action in actions:
            next_state, reward, terminal, _ = simulator.step(state, action)
            if terminal:
                if reward > 0:
                    value = - np.inf
                elif reward == 0:
                    value = 0
                else:
                    value = np.inf
            else:
                value = min(value, _alphabeta(next_state, simulator, depth - 1, alpha, beta, True, heuristic))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def minimax(
    state: np.ndarray,
    simulator: simulators.Base,
    search_depth: int,
    heuristic: minimax.Heuristic,
    maximizing_player: bool
) -> int:
    """Runs the minimax algorithm from the given state in the given simulator. The
    simulator is expected to be one a two player zero-sum game, where a terminal state
    with positive reward indicates a win and a negative reward a loss.

    Args:
        state (np.ndarray): Current state.
        simulator (simulators.Base): Simulator instance.
        search_depth (int): Maximum search depth.
        heuristic (minimax.Heuristic): Heuristic function.
        maximizing_player (bool): True, if the player to make a move aims to maximize
            the heuristic.

    Returns:
        int: Best available action.
    """

    if not simulator.deterministic:
        raise ValueError("Cannot run minimax on a stochastic environment.")

    def value_of_action(action: int) -> float:
        next_state, reward, terminal, _ = simulator.step(state, action)
        if terminal:
            if reward > 0:
                return np.inf if maximizing_player else -np.inf
            if reward == 0:
                return 0
            return -np.inf if maximizing_player else np.inf
        return _alphabeta(next_state, simulator, search_depth - 1, -np.inf, np.inf, not maximizing_player, heuristic)

    action_space = simulator.action_space.as_discrete
    actions = np.arange(action_space.size)[action_space.action_mask(state)]
    return Query(actions).argmax(
        lambda a: value_of_action(a)
    )
