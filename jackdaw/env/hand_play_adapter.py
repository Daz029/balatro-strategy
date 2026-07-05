"""HandPlayAdapter — isolated hand-play episodes for hand-agent training.

Starts episodes directly at the ``SELECTING_HAND`` phase with an injected,
domain-randomized ante / joker set / hands-discards-left / money, bypassing
blind-select and the shop entirely. Built on the same ``initialize_run`` +
``SelectBlind`` pipeline :class:`~jackdaw.env.game_interface.DirectAdapter`
uses for full runs, so blind construction, joker ``setting_blind`` triggers,
deck shuffling, and hand-drawing all go through the real engine unchanged —
only the values fed into that pipeline are overridden before the engine
runs.

See ``CLAUDE.md`` ("Ante-play (hand/discard) track" -> "Training the
hand-agent in isolation from shop") for the training design this supports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from jackdaw.engine.actions import Action, GamePhase, SelectBlind
from jackdaw.env.game_interface import GameState, snapshot


@dataclass(frozen=True)
class JokerCountBand:
    """One weighted joker-count option for count-first sampling.

    ``weight`` is relative probability mass for this count.  ``ante_range``,
    when set, restricts which antes this count can co-occur with (e.g.
    0/1-joker states only at antes 1-2, 5-joker boards only from ante 3 on);
    ``None`` falls back to the config-level ``ante_range``.

    Count-first was a deliberate choice (see CLAUDE.md): the band weights
    are honored *exactly* as specified, and the ante marginal absorbs the
    (mild) distortion the restrictions induce, rather than the other way
    around.
    """

    count: int
    weight: float
    ante_range: tuple[int, int] | None = None


@dataclass
class HandPlayConfig:
    """Domain-randomization ranges for isolated hand-play episodes.

    Sampled deterministically from the ``seed`` passed to :meth:`reset`, so
    episodes stay reproducible.  Boss blinds are intentionally excluded from
    ``blind_stages`` by default — their unique mechanics are out of scope
    for the initial curriculum stage (see CLAUDE.md curriculum: no jokers ->
    small random joker subsets -> full random coverage). Boss selection is
    ante-correct if enabled (see :meth:`reset`), so widening ``blind_stages``
    to include ``"Boss"`` later is safe.

    ``dollars_range`` is sampled flat/uniform regardless of ante — this is a
    stage-1/2 placeholder, not a claim that money is ante-independent in real
    play. It's out of scope until the shop-agent's marginal-value-of-$1 curve
    (see CLAUDE.md "Money/dollar handling") exists and gets fed in.

    Injected jokers are always base/no-edition, no-sticker (no Foil/Holo/
    Polychrome, no Eternal/Perishable/Rental) — a deliberate curriculum-stage
    simplification, not an oversight. Revisit once training targets "full
    random coverage" that should match real shop-purchase distributions.

    ``joker_count_bands``, when set, replaces ``joker_count_range`` +
    uniform-ante sampling with count-first weighted sampling (see
    :class:`JokerCountBand`).  A band whose ``count`` exceeds the pool or
    ``joker_slots`` raises at :meth:`reset` rather than silently clamping —
    clamping would quietly redistribute that band's probability mass onto
    smaller counts.

    ``randomize_joker_state`` controls accumulated-state injection for
    scaling jokers (Ride the Bus's mult, Obelisk's x_mult, ...): without it,
    every scaling joker appears freshly-bought (zero accumulation), which
    systematically teaches a BC policy that these jokers are dead slots.
    Accumulation is sampled uniform over [0, cap] where
    ``cap = trigger_opportunities(ante) x difficulty_fraction x
    per_trigger_gain`` — see ``_SCALING_SPECS``.  Also seeds the run-stat
    priors (``skips``, tarot-usage count) that formula-based jokers
    (Throwback, Fortune Teller) read from game state.  Known gap, deferred:
    hand *levels* (Planet upgrades) and per-hand-type usage counts
    (Supernova) are still always at run-start values.
    """

    ante_range: tuple[int, int] = (1, 8)
    joker_pool: tuple[str, ...] = ()
    joker_count_range: tuple[int, int] = (0, 0)
    hands_range: tuple[int, int] = (1, 4)
    discards_range: tuple[int, int] = (0, 3)
    dollars_range: tuple[int, int] = (0, 50)
    blind_stages: tuple[str, ...] = ("Small", "Big")
    joker_count_bands: tuple[JokerCountBand, ...] | None = None
    randomize_joker_state: bool = True


# ---------------------------------------------------------------------------
# Scaling-joker accumulation model
# ---------------------------------------------------------------------------
#
# cap(ante) = trigger_opportunities(ante) x difficulty_fraction x per_trigger
#
#   - trigger_opportunities(ante): events_per_ante x (ante - 1), optionally
#     clamped by max_events for streak-scoped jokers that reset on an event
#     (Ride the Bus resets on a face card, Obelisk on the most-played hand)
#     and so never accumulate run-length totals.
#   - difficulty_fraction encodes how hard the trigger condition is to hit:
#     0.70 common / 0.60 uncommon / 0.50 rare, with per-joker overrides
#     (Yorick 0.70 — discards are routine; Caino 0.50 — face destruction is
#     scarce). Decay jokers use 1.0: their "trigger" is automatic.
#   - The sample is quantized to whole trigger counts (k x per_trigger,
#     k uniform in [0, k_max]) so injected values are ones the engine could
#     actually have produced.
#
# per_trigger values below mirror the engine's centers.json config (verified
# against `create_joker(key).ability` — see tests). events_per_ante rates
# are documented estimates: ~10 hands, ~8 discard actions (~24 cards),
# ~3 rounds, ~1.5 cards bought/sold, ~1.5 consumables used per ante.


@dataclass(frozen=True)
class _ScalingSpec:
    """How one scaling joker's accumulated state is sampled at injection."""

    field: tuple[str, ...]  # path into card.ability, e.g. ("extra", "chips")
    per_trigger: float
    events_per_ante: float
    fraction: float
    kind: str = "gain"  # "gain" (starts 0) | "xgain" (starts 1) | "decay" | "loyalty"
    max_events: int | None = None  # streak/reset window clamp
    ante_independent: bool = False  # Campfire: resets each boss, flat range
    decay_floor: float = 0.0  # decay: value at/below this destroys the joker


_SCALING_SPECS: dict[str, _ScalingSpec] = {
    # --- additive mult gains ---
    "j_ride_the_bus": _ScalingSpec(("mult",), 1, 10, 0.70, max_events=30),
    "j_green_joker": _ScalingSpec(("mult",), 1, 3, 0.70),  # +1/hand -1/discard: net drift
    "j_trousers": _ScalingSpec(("mult",), 2, 3, 0.60),
    "j_red_card": _ScalingSpec(("mult",), 3, 0.5, 0.70),
    "j_flash": _ScalingSpec(("mult",), 2, 2, 0.60),
    "j_ceremonial": _ScalingSpec(("mult",), 6, 0.75, 0.60),  # 2x sell value, avg ~$3
    # --- additive chip gains (stored under ability["extra"]["chips"]) ---
    "j_square": _ScalingSpec(("extra", "chips"), 4, 2, 0.70),
    "j_runner": _ScalingSpec(("extra", "chips"), 15, 2, 0.70),
    "j_castle": _ScalingSpec(("extra", "chips"), 3, 6, 0.60),
    "j_wee": _ScalingSpec(("extra", "chips"), 8, 2, 0.50),
    # --- x_mult gains ---
    "j_lucky_cat": _ScalingSpec(("x_mult",), 0.25, 1, 0.60, kind="xgain"),  # lucky is scarce
    "j_hologram": _ScalingSpec(("x_mult",), 0.25, 1.5, 0.60, kind="xgain"),
    "j_constellation": _ScalingSpec(("x_mult",), 0.1, 1.5, 0.60, kind="xgain"),
    "j_vampire": _ScalingSpec(("x_mult",), 0.1, 1, 0.60, kind="xgain"),
    "j_glass": _ScalingSpec(("x_mult",), 0.75, 0.3, 0.60, kind="xgain"),
    "j_madness": _ScalingSpec(("x_mult",), 0.5, 1, 0.60, kind="xgain"),
    "j_obelisk": _ScalingSpec(("x_mult",), 0.2, 5, 0.50, kind="xgain", max_events=8),
    "j_campfire": _ScalingSpec(  # resets on boss defeat: flat [x1.0, x2.5]
        ("x_mult",), 0.25, 6, 1.0, kind="xgain", max_events=6, ante_independent=True
    ),
    "j_yorick": _ScalingSpec(("x_mult",), 1, 1, 0.70, kind="xgain"),  # ~24 cards/ante / 23
    "j_caino": _ScalingSpec(("caino_xmult",), 1, 0.5, 0.50, kind="xgain"),
    # --- decays (start high, tick down; must stay above destruction) ---
    "j_ice_cream": _ScalingSpec(("extra", "chips"), 5, 10, 1.0, kind="decay"),
    "j_popcorn": _ScalingSpec(("mult",), 4, 3, 1.0, kind="decay"),
    "j_ramen": _ScalingSpec(("x_mult",), 0.01, 24, 1.0, kind="decay", decay_floor=1.0),
    "j_selzer": _ScalingSpec(("extra",), 1, 10, 1.0, kind="decay"),
    # --- charge-position (Loyalty Card's "N hands until x4") ---
    "j_loyalty_card": _ScalingSpec((), 0, 0, 1.0, kind="loyalty"),
}


def _get_path(ability: dict, path: tuple[str, ...]) -> Any:
    value: Any = ability
    for part in path:
        value = value[part] if isinstance(value, dict) else None
    return value


def _set_path(ability: dict, path: tuple[str, ...], value: Any) -> None:
    target = ability
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def _apply_scaling_state(joker: Any, spec: _ScalingSpec, ante: int, sampler: random.Random) -> None:
    """Sample and write one scaling joker's accumulated state in place."""
    if spec.kind == "loyalty":
        # Uniform charge position: hands_played_at_create in [-every, 0]
        # sweeps the trigger counter through all its phases.
        extra = joker.ability.get("extra", {})
        every = extra.get("every", 5) if isinstance(extra, dict) else 5
        joker.ability["hands_played_at_create"] = -sampler.randint(0, every)
        return

    antes_elapsed = 1 if spec.ante_independent else max(ante - 1, 0)
    events = spec.events_per_ante * antes_elapsed
    if spec.max_events is not None:
        events = min(events, spec.max_events)
    k_max = int(events * spec.fraction)

    if spec.kind == "decay":
        start = _get_path(joker.ability, spec.field)
        # Largest k that keeps the joker strictly above its destruction
        # threshold (it would have been removed by the engine otherwise).
        alive_max = int((start - spec.decay_floor) / spec.per_trigger - 1e-9)
        k_max = min(k_max, max(alive_max, 0))

    if k_max <= 0:
        return
    k = sampler.randint(0, k_max)
    if k == 0:
        return

    if spec.kind == "gain":
        base = _get_path(joker.ability, spec.field) or 0
        _set_path(joker.ability, spec.field, base + k * spec.per_trigger)
    elif spec.kind == "xgain":
        _set_path(joker.ability, spec.field, round(1 + k * spec.per_trigger, 4))
    elif spec.kind == "decay":
        start = _get_path(joker.ability, spec.field)
        value = start - k * spec.per_trigger
        if isinstance(start, int) and float(spec.per_trigger).is_integer():
            value = int(value)
        else:
            value = round(value, 4)
        _set_path(joker.ability, spec.field, value)


class HandPlayAdapter:
    """GameAdapter that starts episodes mid-run, directly in hand-play.

    Satisfies the same duck-typed contract as
    :class:`~jackdaw.env.game_interface.DirectAdapter` (see
    ``jackdaw/env/game_interface.py::GameAdapter``), so it plugs into
    :class:`~jackdaw.env.balatro_env.BalatroEnvironment` unchanged via
    ``adapter_factory=lambda: HandPlayAdapter(config)``.

    ``done``/``won`` fire at ``ROUND_EVAL`` (blind cleared) rather than
    waiting for ``GAME_OVER``, since isolated hand-play episodes have no
    shop phase to reach.
    """

    def __init__(self, config: HandPlayConfig | None = None) -> None:
        self._config = config or HandPlayConfig()
        self._gs: dict[str, Any] = {}

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        from jackdaw.engine.blind import get_new_boss
        from jackdaw.engine.card_factory import create_joker
        from jackdaw.engine.game import step as engine_step
        from jackdaw.engine.run_init import initialize_run

        cfg = self._config
        sampler = random.Random(seed)

        gs = initialize_run(back_key, stake, seed, challenge=challenge)
        rr = gs["round_resets"]

        joker_count: int | None = None
        if cfg.joker_count_bands:
            # Validate every band upfront (not just the sampled one) so a
            # bad config fails on the first reset regardless of seed.
            # Raise-don't-clamp: silently clamping an oversized count would
            # quietly shift that band's probability mass onto smaller
            # counts, corrupting the intended distribution.
            for band in cfg.joker_count_bands:
                if band.count > len(cfg.joker_pool):
                    raise ValueError(
                        f"JokerCountBand(count={band.count}) exceeds joker_pool "
                        f"size {len(cfg.joker_pool)}"
                    )
                if band.count > gs["joker_slots"]:
                    raise ValueError(
                        f"JokerCountBand(count={band.count}) exceeds joker_slots "
                        f"{gs['joker_slots']}"
                    )
            # Count-first: band weights are honored exactly as configured;
            # the ante marginal absorbs the restriction-induced tilt.
            chosen = sampler.choices(
                cfg.joker_count_bands,
                weights=[band.weight for band in cfg.joker_count_bands],
            )[0]
            joker_count = chosen.count
            ante = sampler.randint(*(chosen.ante_range or cfg.ante_range))
        else:
            ante = sampler.randint(*cfg.ante_range)

        rr["ante"] = ante
        rr["blind_ante"] = ante

        rr["hands"] = sampler.randint(*cfg.hands_range)
        rr["discards"] = sampler.randint(*cfg.discards_range)
        gs["dollars"] = sampler.randint(*cfg.dollars_range)
        gs["blind_on_deck"] = sampler.choice(cfg.blind_stages)

        # initialize_run() already picked a boss key for ante 1 (baked into
        # blind_choices["Boss"] via its internal assign_ante_blinds(1, ...)
        # call). That key is wrong for any other sampled ante, so redraw it
        # directly with get_new_boss() rather than reusing it — and rather
        # than re-running assign_ante_blinds(ante, ...), which would also
        # pollute gs["bosses_used"] with a phantom ante-1 usage count and
        # skew the "favor least-used boss" selection.
        if gs["blind_on_deck"] == "Boss":
            rr["blind_choices"]["Boss"] = get_new_boss(
                ante,
                gs["bosses_used"],
                gs["rng"],
                win_ante=gs.get("win_ante", 8),
                banned_keys=gs.get("banned_keys"),
            )

        if joker_count is not None:
            count = joker_count
        else:
            # Legacy path keeps its historical clamp semantics — only the
            # band path gets raise-don't-clamp (validated above).
            lo, hi = cfg.joker_count_range
            count = (
                min(sampler.randint(lo, hi), len(cfg.joker_pool), gs["joker_slots"])
                if cfg.joker_pool and hi > 0
                else 0
            )
        if count > 0:
            keys = sampler.sample(list(cfg.joker_pool), count)
            gs["jokers"] = [create_joker(key) for key in keys]

        if cfg.randomize_joker_state:
            # Run-stat priors read by formula-based jokers. Drawn
            # unconditionally (even with no matching joker present) so the
            # sampler stream — and thus every later draw — depends only on
            # the seed and config, not on which jokers were sampled.
            antes_elapsed = max(ante - 1, 0)
            gs["skips"] = sampler.randint(0, min(2 * antes_elapsed, 4))  # Throwback
            tarots_used = sampler.randint(0, int(1.5 * antes_elapsed * 0.7))  # Fortune Teller
            gs["consumable_usage_total"] = {"tarot": tarots_used}
            # score_hand reads the flattened key; the engine's play-hand
            # step re-derives it from consumable_usage_total, but the
            # solver labels states *before* any step has run.
            gs["consumable_usage_tarot"] = tarots_used

            for joker in gs.get("jokers", []):
                spec = _SCALING_SPECS.get(joker.center_key)
                if spec is not None:
                    _apply_scaling_state(joker, spec, ante, sampler)

        self._gs = gs
        engine_step(self._gs, SelectBlind())
        return snapshot(self._gs)

    def step(self, action: Action) -> GameState:
        from jackdaw.engine.game import step as engine_step

        engine_step(self._gs, action)
        return snapshot(self._gs)

    def get_legal_actions(self) -> list[Action]:
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        return engine_legal(self._gs)

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._gs

    @property
    def done(self) -> bool:
        phase = self._gs.get("phase")
        return phase in (GamePhase.GAME_OVER, GamePhase.ROUND_EVAL)

    @property
    def won(self) -> bool:
        phase = self._gs.get("phase")
        return phase == GamePhase.ROUND_EVAL or bool(self._gs.get("won", False))
