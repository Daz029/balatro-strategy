"""Compound autoregressive pointer policy for hand actions.

Monotone ascent is the important structural choice here: every legal card set
has exactly one reachable sorted pick sequence, so its sequence NLL is
comparable to a flat softmax NLL over the same discrete event.  The type token
conditions the decoder because selecting a play and selecting a discard are
different policies even when they point at the same cards.  A five-card action
terminates by construction, so its target has no stop token; shorter actions
have a stop token unless the mask leaves stop as the only legal choice (whose
masked log-probability is exactly zero).

``initial_type_mask`` and ``pick_step_mask`` are the sole legality machinery.
They are pure torch functions and are the parity contract for the BC, PPO,
evaluation, and partner-policy call sites.  Consumers must route every
per-step legality decision through these functions rather than reproducing
mask logic locally.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from gymnasium import spaces

from jackdaw.agents.hand_policy_v3 import HandPlayFeaturesExtractorV3

CARD_SLOTS = 40
STOP_INDEX = CARD_SLOTS
MAX_PICKS = 5
TYPE_COUNT = 2
POOLED_DIM = 256
CARD_LATENT_DIM = 64

_GC_HANDS_LEFT_IDX = 13
_GC_DISCARDS_LEFT_IDX = 14


def initial_type_mask(
    hands_left: int | torch.Tensor, discards_left: int | torch.Tensor
) -> torch.Tensor:
    """Return the budget legality mask for ``(PlayHand, Discard)``.

    Scalar inputs return ``(2,)``; batch-shaped inputs return ``(..., 2)``.
    A row with neither budget available is a caller error and raises.
    """

    hands, discards = torch.broadcast_tensors(
        torch.as_tensor(hands_left), torch.as_tensor(discards_left)
    )
    mask = torch.stack((hands >= 1, discards >= 1), dim=-1)
    if mask.ndim == 1:
        invalid = not bool(mask.any())
    else:
        invalid = bool((~mask.any(dim=-1)).any())
    if invalid:
        raise ValueError("no legal hand action: both hands_left and discards_left are zero")
    return mask


def pick_step_mask(
    hand_mask: torch.Tensor, last_pick: int | torch.Tensor, n_picked: int | torch.Tensor
) -> torch.Tensor:
    """Return the 40-pick-plus-stop legality mask for one decode step.

    ``last_pick`` is exclusive: only live indices strictly greater than it
    are legal.  Stop is legal after at least one pick and before the fifth
    pick.  Inputs may be scalar/batch-shaped for ``last_pick`` and
    ``n_picked``; the returned shape is ``hand_mask.shape[:-1] + (41,)``.
    """

    hand = torch.as_tensor(hand_mask, dtype=torch.bool)
    if hand.ndim < 1 or hand.shape[-1] != CARD_SLOTS:
        raise ValueError(f"hand_mask must end in ({CARD_SLOTS},), got {tuple(hand.shape)}")

    prefix = hand.shape[:-1]
    last = torch.as_tensor(last_pick, dtype=torch.long, device=hand.device)
    count = torch.as_tensor(n_picked, dtype=torch.long, device=hand.device)
    try:
        last = torch.broadcast_to(last, prefix)
        count = torch.broadcast_to(count, prefix)
    except RuntimeError as exc:
        raise ValueError(
            f"last_pick and n_picked must broadcast to hand batch shape {prefix}"
        ) from exc

    view_shape = (1,) * len(prefix) + (CARD_SLOTS,)
    indices = torch.arange(CARD_SLOTS, device=hand.device).reshape(view_shape)
    pick_legal = hand & (indices > last.unsqueeze(-1)) & (count.unsqueeze(-1) < MAX_PICKS)
    stop_legal = (count >= 1) & (count < MAX_PICKS)
    return torch.cat((pick_legal, stop_legal.unsqueeze(-1)), dim=-1)


def _masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a categorical mask and return normalized log probabilities."""

    if bool((~mask.any(dim=-1)).any()):
        raise ValueError("a decode step has no legal token")
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def _entropy(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy of a masked categorical distribution without ``0 * -inf``."""

    finite_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probabilities = log_probs.exp()
    return -(probabilities * finite_log_probs).sum(dim=-1)


class PointerActionHead(nn.Module):
    """Type-conditioned autoregressive distribution over hand actions."""

    def __init__(
        self,
        pooled_dim: int = POOLED_DIM,
        card_latent_dim: int = CARD_LATENT_DIM,
        state_dim: int = POOLED_DIM,
    ) -> None:
        super().__init__()
        self.pooled_dim = pooled_dim
        self.card_latent_dim = card_latent_dim
        self.state_dim = state_dim

        self.type_head = nn.Linear(pooled_dim, TYPE_COUNT)
        self.type_embedding = nn.Embedding(TYPE_COUNT, 32)
        self.state_init = nn.Sequential(
            nn.Linear(pooled_dim + 32, state_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(card_latent_dim, state_dim)
        self.pick_head = nn.Sequential(
            nn.Linear(state_dim + card_latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.stop_head = nn.Linear(state_dim, 1)

    @property
    def type_logits(self) -> nn.Linear:
        """Compatibility alias for callers that refer to the type logits head."""

        return self.type_head

    def _state_from_type(self, pooled: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        type_embedding = self.type_embedding(action_type)
        return self.state_init(torch.cat((pooled, type_embedding), dim=-1))

    def _pointer_logits(
        self, state: torch.Tensor, card_latents: torch.Tensor
    ) -> torch.Tensor:
        state_per_card = state.unsqueeze(1).expand(-1, CARD_SLOTS, -1)
        pick_logits = self.pick_head(torch.cat((state_per_card, card_latents), dim=-1)).squeeze(-1)
        stop_logits = self.stop_head(state).squeeze(-1).unsqueeze(-1)
        return torch.cat((pick_logits, stop_logits), dim=-1)

    @staticmethod
    def _batchify_inputs(
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        action_type: torch.Tensor | None = None,
        card_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if card_latents.ndim == 2:
            card_latents = card_latents.unsqueeze(0)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        if hand_mask.ndim == 1:
            hand_mask = hand_mask.unsqueeze(0)
        if action_type is not None and action_type.ndim == 0:
            action_type = action_type.unsqueeze(0)
        if card_indices is not None and card_indices.ndim == 1:
            card_indices = card_indices.unsqueeze(0)
        return card_latents, pooled, hand_mask, action_type, card_indices

    def _validate_labels(
        self,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        action_type: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> torch.Tensor:
        if action_type.ndim != 1 or card_indices.ndim != 2 or card_indices.shape[-1] != MAX_PICKS:
            raise ValueError("action_type must be (B,) and card_indices must be (B, 5)")
        if (
            card_indices.shape[0] != hand_mask.shape[0]
            or action_type.shape[0] != hand_mask.shape[0]
        ):
            raise ValueError("all labels and observations must have the same batch size")

        action_type = action_type.long()
        card_indices = card_indices.long()
        if bool(((action_type < 0) | (action_type >= TYPE_COUNT)).any()):
            raise ValueError("action_type must contain only 0 (PlayHand) or 1 (Discard)")
        if bool((card_indices < -1).any()):
            raise ValueError("card_indices may only use -1 as padding")

        lengths = (card_indices >= 0).sum(dim=-1)
        if bool(((lengths < 1) | (lengths > MAX_PICKS)).any()):
            raise ValueError("card_indices must contain 1-5 real ascending entries")
        positions = torch.arange(MAX_PICKS, device=card_indices.device).unsqueeze(0)
        if bool(((positions >= lengths.unsqueeze(-1)) & (card_indices != -1)).any()):
            raise ValueError("card_indices padding must be trailing -1 values")
        adjacent = positions[:, 1:] < lengths.unsqueeze(-1)
        if bool(((card_indices[:, 1:] <= card_indices[:, :-1]) & adjacent).any()):
            raise ValueError("card_indices must be strictly ascending")
        if bool((card_indices >= CARD_SLOTS).any()):
            raise ValueError(f"card_indices must be below {CARD_SLOTS}")

        safe_indices = card_indices.clamp_min(0)
        live = hand_mask.gather(1, safe_indices).bool()
        if bool((~live & (positions < lengths.unsqueeze(-1))).any()):
            raise ValueError("card_indices must point at live hand rows")

        type_mask = initial_type_mask(hands_left, discards_left).to(hand_mask.device)
        if type_mask.ndim == 1:
            type_mask = type_mask.unsqueeze(0).expand(hand_mask.shape[0], -1)
        if type_mask.shape[0] != hand_mask.shape[0]:
            raise ValueError("budget tensors must broadcast to the observation batch")
        if bool(~type_mask.gather(1, action_type.unsqueeze(-1)).squeeze(-1).all()):
            raise ValueError("action_type is illegal under the supplied budgets")
        return lengths

    def _teacher_forced_distributions(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        action_type: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked type/pointer distributions and active-step flags."""

        card_latents, pooled, hand_mask, action_type, card_indices = self._batchify_inputs(
            card_latents, pooled, hand_mask, action_type, card_indices
        )
        assert action_type is not None and card_indices is not None
        card_latents = card_latents.to(device=pooled.device)
        hand_mask = hand_mask.to(device=pooled.device, dtype=torch.bool)
        action_type = action_type.to(device=pooled.device)
        card_indices = card_indices.to(device=pooled.device)
        lengths = self._validate_labels(
            hand_mask, hands_left, discards_left, action_type, card_indices
        )
        action_type = action_type.long()
        card_indices = card_indices.long()

        type_mask = initial_type_mask(hands_left, discards_left).to(pooled.device)
        if type_mask.ndim == 1:
            type_mask = type_mask.unsqueeze(0).expand(pooled.shape[0], -1)
        type_log_probs = _masked_log_probs(self.type_head(pooled), type_mask)
        state = self._state_from_type(pooled, action_type)
        last_pick = torch.full(
            (pooled.shape[0],), -1, dtype=torch.long, device=pooled.device
        )
        n_picked = torch.zeros(pooled.shape[0], dtype=torch.long, device=pooled.device)
        active_steps = [torch.ones(pooled.shape[0], dtype=torch.bool, device=pooled.device)]
        pointer_distributions = []

        for step in range(MAX_PICKS):
            active_pick = step < lengths
            active_token = active_pick | ((step == lengths) & (lengths < MAX_PICKS))
            after_stop = step > lengths
            mask_last = torch.where(
                after_stop,
                torch.full_like(last_pick, CARD_SLOTS - 1),
                last_pick,
            )
            mask_count = torch.where(
                after_stop,
                torch.full_like(n_picked, MAX_PICKS - 1),
                n_picked,
            )
            mask = pick_step_mask(hand_mask, mask_last, mask_count)
            pointer_log_probs = _masked_log_probs(self._pointer_logits(state, card_latents), mask)
            pointer_distributions.append(pointer_log_probs)
            active_steps.append(active_token)

            target = torch.where(
                active_pick,
                card_indices[:, step],
                torch.full_like(card_indices[:, step], STOP_INDEX),
            )
            safe_target = target.clamp(0, CARD_SLOTS - 1)
            picked_latent = card_latents.gather(
                1, safe_target[:, None, None].expand(-1, 1, card_latents.shape[-1])
            ).squeeze(1)
            proposed_state = self.gru(picked_latent, state)
            state = torch.where(active_pick.unsqueeze(-1), proposed_state, state)
            last_pick = torch.where(active_pick, target, last_pick)
            n_picked = n_picked + active_pick.long()

        return (
            type_log_probs,
            torch.stack(pointer_distributions, dim=1),
            torch.stack(active_steps, dim=1),
            type_mask,
        )

    def teacher_forced_step_distributions(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        action_type: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked log-prob distributions along a labeled sequence.

        The return value is ``(type_log_probs, pointer_log_probs, active)``
        with shapes ``(B, 2)``, ``(B, 5, 41)``, and ``(B, 6)``.  The first
        active step is the type token; pointer steps are active through the
        taken picks and the stop token when a sequence ends before five picks.
        Illegal tokens are represented by ``-inf`` and active distributions
        therefore normalize to one over legal tokens.
        """

        type_log_probs, pointer_log_probs, active, _ = self._teacher_forced_distributions(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            action_type,
            card_indices,
        )
        return type_log_probs, pointer_log_probs, active

    def teacher_forced_log_probs(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        action_type: torch.Tensor,
        card_indices: torch.Tensor,
        *,
        return_uniform_log_probs: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Return ``(per_step_log_probs, sequence_log_prob, entropies)``.

        The per-step tensors have shape ``(B, 6)``: type followed by five
        pointer steps.  Pointer steps contain either a picked card or stop;
        columns after the labeled sequence terminates are zero.

        When ``return_uniform_log_probs`` is true, append a fourth tensor with
        the legal-token uniform mean log-probability at every active step.
        """

        type_log_probs, pointer_log_probs, active, type_mask = self._teacher_forced_distributions(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            action_type,
            card_indices,
        )
        card_indices = card_indices.to(device=pointer_log_probs.device, dtype=torch.long)
        if card_indices.ndim == 1:
            card_indices = card_indices.unsqueeze(0)
        action_type = action_type.to(device=type_log_probs.device, dtype=torch.long)
        if action_type.ndim == 0:
            action_type = action_type.unsqueeze(0)

        type_target = type_log_probs.gather(1, action_type.unsqueeze(-1)).squeeze(-1)
        type_uniform = (
            type_log_probs.masked_fill(~type_mask, 0.0).sum(dim=-1)
            / type_mask.sum(dim=-1).clamp_min(1)
        )
        type_entropy = _entropy(type_log_probs, type_mask)
        step_log_probs = [type_target]
        step_uniform_log_probs = [type_uniform]
        step_entropies = [type_entropy]

        for step in range(MAX_PICKS):
            target = torch.where(
                card_indices[:, step] >= 0,
                card_indices[:, step],
                torch.full_like(card_indices[:, step], STOP_INDEX),
            )
            target_log_prob = pointer_log_probs[:, step].gather(1, target.unsqueeze(-1)).squeeze(-1)
            pointer_mask = torch.isfinite(pointer_log_probs[:, step])
            uniform = (
                pointer_log_probs[:, step].masked_fill(~pointer_mask, 0.0).sum(dim=-1)
                / pointer_mask.sum(dim=-1).clamp_min(1)
            )
            entropy = _entropy(pointer_log_probs[:, step], pointer_mask)
            pointer_active = active[:, step + 1]
            step_log_probs.append(
                torch.where(pointer_active, target_log_prob, torch.zeros_like(target_log_prob))
            )
            step_uniform_log_probs.append(
                torch.where(pointer_active, uniform, torch.zeros_like(uniform))
            )
            step_entropies.append(
                torch.where(pointer_active, entropy, torch.zeros_like(entropy))
            )

        per_step_log_probs = torch.stack(step_log_probs, dim=-1)
        per_step_uniform_log_probs = torch.stack(step_uniform_log_probs, dim=-1)
        per_step_entropies = torch.stack(step_entropies, dim=-1)
        if return_uniform_log_probs:
            return (
                per_step_log_probs,
                per_step_log_probs.sum(dim=-1),
                per_step_entropies,
                per_step_uniform_log_probs,
            )
        return per_step_log_probs, per_step_log_probs.sum(dim=-1), per_step_entropies

    def _pack_decoded(
        self, action_types: torch.Tensor, picked: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        valid = picked >= 0
        flat = picked[valid]
        split_sizes = lengths.clamp(min=1, max=MAX_PICKS).detach().cpu().tolist()
        return action_types, tuple(torch.split(flat, split_sizes))

    def _decode(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        chooser: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        card_latents, pooled, hand_mask, _, _ = self._batchify_inputs(
            card_latents, pooled, hand_mask
        )
        card_latents = card_latents.to(device=pooled.device)
        hand_mask = hand_mask.to(device=pooled.device, dtype=torch.bool)
        if bool((~hand_mask.any(dim=-1)).any()):
            raise ValueError("cannot decode a hand with no live cards")

        type_mask = initial_type_mask(hands_left, discards_left).to(pooled.device)
        if type_mask.ndim == 1:
            type_mask = type_mask.unsqueeze(0).expand(pooled.shape[0], -1)
        type_log_probs = _masked_log_probs(self.type_head(pooled), type_mask)
        action_types = chooser(type_log_probs)
        state = self._state_from_type(pooled, action_types)

        batch_size = pooled.shape[0]
        last_pick = torch.full((batch_size,), -1, dtype=torch.long, device=pooled.device)
        n_picked = torch.zeros(batch_size, dtype=torch.long, device=pooled.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=pooled.device)
        picked = torch.full(
            (batch_size, MAX_PICKS), -1, dtype=torch.long, device=pooled.device
        )

        for step in range(MAX_PICKS):
            # Finished rows are represented as a stop-only state through the
            # shared mask function so the batch can keep decoding without a
            # Python loop over examples.
            mask_last = torch.where(
                active, last_pick, torch.full_like(last_pick, CARD_SLOTS - 1)
            )
            mask_count = torch.where(
                active, n_picked, torch.full_like(n_picked, MAX_PICKS - 1)
            )
            mask = pick_step_mask(hand_mask, mask_last, mask_count)
            pointer_log_probs = _masked_log_probs(self._pointer_logits(state, card_latents), mask)
            token = chooser(pointer_log_probs)
            is_pick = token < CARD_SLOTS
            pick_now = active & is_pick
            stop_now = active & ~is_pick
            picked[:, step] = torch.where(pick_now, token, picked[:, step])

            safe_token = token.clamp(0, CARD_SLOTS - 1)
            picked_latent = card_latents.gather(
                1, safe_token[:, None, None].expand(-1, 1, card_latents.shape[-1])
            ).squeeze(1)
            proposed_state = self.gru(picked_latent, state)
            state = torch.where(pick_now.unsqueeze(-1), proposed_state, state)
            last_pick = torch.where(pick_now, token, last_pick)
            n_picked = n_picked + pick_now.long()
            cap_now = pick_now & (n_picked >= MAX_PICKS)
            active = active & ~(stop_now | cap_now)

        return self._pack_decoded(action_types, picked, n_picked)

    def greedy_decode(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Decode by per-step argmax under the shared legality mask."""

        return self._decode(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            lambda log_probs: log_probs.argmax(dim=-1),
        )

    def sample(
        self,
        card_latents: torch.Tensor,
        pooled: torch.Tensor,
        hand_mask: torch.Tensor,
        hands_left: int | torch.Tensor,
        discards_left: int | torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Sample the type and every pointer/stop token multinomially."""

        def multinomial(log_probs: torch.Tensor) -> torch.Tensor:
            return torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)

        return self._decode(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            multinomial,
        )


class HandPointerBCModel(nn.Module):
    """The v3 trunk, pointer distribution, and pooled p-clear value head."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__()
        self.features_extractor = HandPlayFeaturesExtractorV3(observation_space)
        self.pointer_head = PointerActionHead()
        self.value_net = nn.Linear(POOLED_DIM, 1)

    @staticmethod
    def _budgets(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        global_context = obs["global_context"]
        hands_left = torch.round(global_context[..., _GC_HANDS_LEFT_IDX] * 10).long()
        discards_left = torch.round(global_context[..., _GC_DISCARDS_LEFT_IDX] * 10).long()
        return hands_left, discards_left

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        action_type: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return labeled sequence log-probabilities and pooled values."""

        card_latents, pooled = self.features_extractor(obs)
        hands_left, discards_left = self._budgets(obs)
        _, sequence_log_prob, _ = self.pointer_head.teacher_forced_log_probs(
            card_latents,
            pooled,
            obs["hand_mask"],
            hands_left,
            discards_left,
            action_type,
            card_indices,
        )
        return sequence_log_prob, self.value_net(pooled).squeeze(-1)

    @torch.no_grad()
    def decode(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return the deployment decoder's greedy type and sorted card picks."""

        card_latents, pooled = self.features_extractor(obs)
        hands_left, discards_left = self._budgets(obs)
        return self.pointer_head.greedy_decode(
            card_latents, pooled, obs["hand_mask"], hands_left, discards_left
        )
