from typing import Union

import numpy as np
import torch
from torch import nn

import ai


class Agent:
    """A3C agent. This agent wraps a network to be used for inference."""

    def __init__(self, network: nn.Module, state_dtype: torch.dtype = torch.float32):
        """
        Args:
            network (nn.Module): Network with two output heads, policy logits and state
                value.
            state_dtype (torch.dtype, optional): Data type of states to be fed into the
                network. Defaults to `torch.float32`.
        """
        self._network = network
        self._state_dtype = state_dtype

    def act(self, state: Union[np.ndarray, torch.Tensor], action_mask: Union[np.ndarray, torch.Tensor]) -> int:
        """Returns an action, given the state and action mask.

        Args:
            state (Union[np.ndarray, torch.Tensor]): State.
            action_mask (Union[np.ndarray, torch.Tensor]): Action mask, indicating legal
                actions.

        Returns:
            int: Action index.
        """
        state = torch.as_tensor(state, dtype=self._state_dtype)
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
        return self.act_bulk(state.unsqueeze(0), action_mask.unsqueeze(0)).item()

    def act_bulk(self, states: Union[np.ndarray, torch.Tensor], action_masks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Returns a set of actions for the given states and action masks.

        Args:
            states (Union[np.ndarray, torch.Tensor]): States.
            action_masks (Union[np.ndarray, torch.Tensor]): Action masks, indicating
                legal actions.

        Returns:
            torch.Tensor: Tensor containing action indices, data type is `torch.long`.
        """
        states = torch.as_tensor(states, dtype=self._state_dtype)
        action_masks = torch.as_tensor(action_masks, dtype=torch.bool)

        with torch.no_grad():
            p, _ = self._network(states)
        p[~action_masks] = -float("inf")
        p = torch.softmax(p, dim=1)
        return ai.utils.torch.random.choice(p)
