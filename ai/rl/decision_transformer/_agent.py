import itertools
from dataclasses import dataclass
from typing import Iterator, Callable

import torch
from torch import nn

import ai
from ai.utils import Factory


@dataclass
class Model:
    state_encoder: nn.Module
    action_encoder: nn.Module
    reward_encoder: nn.Module
    positional_encoder: nn.Module
    transformer: "ai.rl.decision_transformer.TransformerEncoder"
    action_decoder: nn.Module

    def parameters(self) -> Iterator[torch.Tensor]:
        yield from itertools.chain(
            self.state_encoder.parameters(),
            self.action_encoder.parameters(),
            self.reward_encoder.parameters(),
            self.positional_encoder.parameters(),
            self.transformer.parameters(),
            self.action_decoder.parameters(),
        )

    def to(self, device: torch.device):
        return Model(
            self.state_encoder.to(device),
            self.action_encoder.to(device),
            self.reward_encoder.to(device),
            self.positional_encoder.to(device),
            self.transformer.to(device),
            self.action_decoder.to(device),
        )


@dataclass
class ModelFactory:
    state_encoder: Factory[nn.Module]
    action_encoder: Factory[nn.Module]
    reward_encoder: Factory[nn.Module]
    positional_encoder: Factory[nn.Module]
    transformer: Factory["ai.rl.decision_transformer.TransformerEncoder"]
    action_decoder: Factory[nn.Module]

    def get_model(self) -> Model:
        return Model(
            self._state_encoder(),
            self._action_encoder(),
            self._reward_encoder(),
            self._positional_encoder(),
            self._transformer(),
            self._action_decoder(),
        )


def interleave(
    reward_to_gos: torch.Tensor, states: torch.Tensor, actions: torch.Tensor
):
    bs = states.shape[0]
    encoding_dim = states.shape[-1]
    actions = torch.cat(
        (
            actions,
            torch.zeros(
                bs, 1, encoding_dim, device=actions.device, dtype=actions.dtype
            ),
        ),
        dim=1,
    )
    stacked = torch.stack((reward_to_gos, states, actions), dim=2)
    return stacked.view(bs, -1, encoding_dim)[:, :-1, :]


def encode_and_interleave(
    self: "Agent",
    states: torch.Tensor,
    actions: torch.Tensor,
    reward_to_gos: torch.Tensor,
    time_steps: torch.Tensor,
) -> torch.Tensor:
    positions = self._model.positional_encoder(time_steps)
    states = self._model.state_encoder(states) + positions
    actions = self._model.action_encoder(actions) + positions[:, :-1]
    reward_to_gos = self._model.reward_encoder(reward_to_gos) + positions
    return interleave(reward_to_gos, states, actions)


def evaluate_transformer(
    transformer: "ai.rl.decision_transformer.TransformerEncoder",
    sequences: torch.Tensor,
) -> torch.Tensor:
    seq_shape = sequences.shape[1]
    bs = sequences.shape[0]
    seq_vec = torch.arange(seq_shape, device=sequences.device)
    mask = (seq_vec.view(1, -1) <= seq_vec.view(-1, 1)).unsqueeze_(0).expand(bs, -1, -1)
    return transformer(sequences, mask=mask)


def decode(
    decoder: nn.Module, transformer_output: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    indices = 3 * lengths - 2
    batch_vec = torch.arange(
        transformer_output.shape[0], device=transformer_output.device
    )
    return decoder(transformer_output[batch_vec, indices])


class Agent:
    """Utility class for easily evaluating and updating decision transformer models."""

    def __init__(
        self,
        state_encoder: Factory[nn.Module],
        action_encoder: Factory[nn.Module],
        reward_encoder: Factory[nn.Module],
        positional_encoder: Factory[nn.Module],
        transformer: Factory["ai.rl.decision_transformer.TransformerEncoder"],
        action_decoder: Factory[nn.Module],
    ):
        """All networks are passed wrapped in a factory object.
        Args:
            state_encoder (Factory[nn.Module]): State encoder network.
            action_encoder (Factory[nn.Module]): Action encoder network.
            reward_encoder (Factory[nn.Module]): Reward (to go) encoder network.
            positional_encoder (Factory[nn.Module]): Positional encoder network.
            transformer (Factory["ai.rl.decision_transformer.TransformerEncoder"]): Transformer network.
            action_decoder (Factory[nn.Module]): Decoding network.
        """
        self._factory = ModelFactory(
            state_encoder,
            action_encoder,
            reward_encoder,
            positional_encoder,
            transformer,
            action_decoder,
        )
        self._model = self._factory.get_model()

    @property
    def model_factory(self) -> ModelFactory:
        """Wrapper class containing all network factories used during agent
        construction."""
        return self._factory

    @property
    def model(self) -> Model:
        """Wrapper class containing all networks used by the agent."""
        return self._model

    def compute_loss(self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Computes the loss given the observed data."""
        sequences = encode_and_interleave(
            self, states, actions, reward_to_gos, time_steps
        )
        transformer_output = evaluate_transformer(self, sequences)
        max_length = lengths.max()
        indices = torch.arange(1, 3 * max_length - 1, step=3, device=lengths.device).view(1, -1)
        batch_vec = torch.arange(transformer_output.shape[0], device=lengths.device).view(-1, 1)


    def _evaluate_action_untraced(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            sequences = encode_and_interleave(
                self, states, actions, reward_to_gos, time_steps
            )
            transformer_output = evaluate_transformer(self, sequences)
            return decode(self._model.action_decoder, transformer_output, lengths)

    def evaluate_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates the model on the given sequence of data. All sequences are given in
        `(BATCH, SEQ, *FEATURES)` format. The inference is run in `inference_mode`,
        implying no backward passes possible.

        Args:
            states (torch.Tensor): Sequence of states.
            actions (torch.Tensor): Sequence of actions.
            reward_to_gos (torch.Tensor): Sequence of reward to gos.
            time_steps (torch.Tensor): Time steps of each observation.
            lengths (torch.Tensor): Sequence lengths, in number of states observed.

        Returns:
            torch.Tensor: Model output for the next action, shaped
                `(BATCH, *ACTIONFEATURES)`.
        """
        self.evaluate_action = torch.jit.trace(
            self._evaluate_action_untraced,
            (states, actions, reward_to_gos, time_steps, lengths),
        )
        return self.evaluate_action(states, actions, reward_to_gos, time_steps, lengths)
