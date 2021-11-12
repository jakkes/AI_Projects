import itertools
from dataclasses import dataclass
from typing import Iterator, Callable

import torch
from torch import nn

import ai
from ai.utils import Factory


class Model(nn.Module):
    def __init__(
        self,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        reward_encoder: nn.Module,
        positional_encoder: nn.Module,
        transformer: "ai.rl.decision_transformer.TransformerEncoder",
        action_decoder: nn.Module
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.reward_encoder = reward_encoder
        self.positional_encoder = positional_encoder
        self.transformer = transformer
        self.action_decoder = action_decoder

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        sequences = encode_and_interleave(
            self, states, actions, reward_to_gos, time_steps
        )
        transformer_output = evaluate_transformer(self.transformer, sequences)
        return decode(self.action_decoder, transformer_output, lengths)

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        sequences = encode_and_interleave_full(
            self, states, actions, reward_to_gos, time_steps
        )
        transformer_output = evaluate_transformer(self.transformer, sequences)
        max_length = lengths.max()
        action_indices = torch.arange(
            1, 3 * max_length - 1, step=3, device=lengths.device
        )
        hidden_states = transformer_output[:, action_indices]
        target_actions = actions[:, :max_length]
        action_mask = torch.arange(max_length, device=lengths.device).view(
            1, -1
        ) < lengths.view(-1, 1)

        decoded_actions = self.action_decoder(hidden_states[action_mask])
        target_actions = target_actions[action_mask]

        return loss_fn(decoded_actions, target_actions)


@dataclass
class ModelFactory:
    state_encoder: Factory[nn.Module]
    action_encoder: Factory[nn.Module]
    reward_encoder: Factory[nn.Module]
    positional_encoder: Factory[nn.Module]
    transformer: Factory["ai.rl.decision_transformer.TransformerEncoder"]
    action_decoder: Factory[nn.Module]

    def __call__(self) -> Model:
        return self.get_model()

    def get_model(self) -> Model:
        return Model(
            self.state_encoder(),
            self.action_encoder(),
            self.reward_encoder(),
            self.positional_encoder(),
            self.transformer(),
            self.action_decoder(),
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
    self: "Model",
    states: torch.Tensor,
    actions: torch.Tensor,
    reward_to_gos: torch.Tensor,
    time_steps: torch.Tensor,
) -> torch.Tensor:
    positions = self.positional_encoder(time_steps.unsqueeze(-1))
    states = self.state_encoder(states) + positions
    actions = self.action_encoder(actions) + positions[:, :-1]
    reward_to_gos = self.reward_encoder(reward_to_gos.unsqueeze(-1)) + positions
    return interleave(reward_to_gos, states, actions)


def encode_and_interleave_full(
    self: "Model",
    states: torch.Tensor,
    actions: torch.Tensor,
    reward_to_gos: torch.Tensor,
    time_steps: torch.Tensor,
) -> torch.Tensor:
    positions = self.positional_encoder(time_steps.unsqueeze(-1))
    states = self.state_encoder(states) + positions
    actions = self.action_encoder(actions) + positions
    reward_to_gos = self.reward_encoder(reward_to_gos.unsqueeze(-1)) + positions
    stacked = torch.stack((reward_to_gos, states, actions), dim=2)

    # view (batchsize, sequences, embeddingdim)
    return stacked.view(states.shape[0], -1, states.shape[-1])[:, :-1, :]


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
            transformer (Factory["ai.rl.decision_transformer.TransformerEncoder"]):
                Transformer network.
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

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Computes the loss over the given data.

        Args:
            states (torch.Tensor): Sequence of states, shaped 
                `(BATCH, SEQ, *STATEFEATURES)`.
            actions (torch.Tensor): Sequence of actions, shaped
                `(BATCH, SEQ, *ACITONFEATURES)`.
            reward_to_gos (torch.Tensor): Sequence of reward to gos, shaped
                `(BATCH, SEQ)`.
            time_steps (torch.Tensor): Time steps of each observation, shaped
                `(BATCH, SEQ)`.
            lengths (torch.Tensor): Sequence lengths, in number of states observed.
                Shaped `(BATCH, )`.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Function
                taking the output from the action decoding step and the corresponding
                actions (in batch format, shaped `(BATCH, *ACTION_FEATURES)`), producing
                a scalar tensor.

        Returns:
            torch.Tensor: Loss, output from `loss_fn`.
        """
        return self._model.loss(states, actions, reward_to_gos, time_steps, lengths, loss_fn)

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        reward_to_gos: torch.Tensor,
        time_steps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates the model on the given sequence of data. All sequences are given in
        `(BATCH, SEQ, *FEATURES)` format. The inference is run in inference_mode,
        implying no backward passes are possible on the output.

        Args:
            states (torch.Tensor): Sequence of states, shaped 
                `(BATCH, SEQ, *STATEFEATURES)`.
            actions (torch.Tensor): Sequence of actions, shaped
                `(BATCH, SEQ, *ACITONFEATURES)`.
            reward_to_gos (torch.Tensor): Sequence of reward to gos, shaped
                `(BATCH, SEQ)`.
            time_steps (torch.Tensor): Time steps of each observation, shaped
                `(BATCH, SEQ)`.
            lengths (torch.Tensor): Sequence lengths, in number of states observed.
                Shaped `(BATCH, )`.

        Returns:
            torch.Tensor: Model output for the next action, shaped
                `(BATCH, *ACTIONFEATURES)`.
        """
        with torch.inference_mode():
            return self._model(states, actions, reward_to_gos, time_steps, lengths)
