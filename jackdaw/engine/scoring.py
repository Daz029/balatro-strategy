"""Scoring pipeline and eval_card wrapper.

Ports ``eval_card`` from ``common_events.lua:580`` and the scoring
pipeline from ``state_events.lua:571-1065``.

``score_hand_base``: Phases 1-4, 6-8, 12 without joker effects.
``score_hand``: Full pipeline including Phases 5, 7b-d, 8b-c, 9 with jokers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jackdaw.engine import lifecycle, read
from jackdaw.engine.effects import (
    AddChips,
    DestroyCard,
    DestroyPlayingCards,
    EaseDollars,
    Effect,
    EffectQueue,
)
from jackdaw.engine.read import StateView

if TYPE_CHECKING:
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.card import Card
    from jackdaw.engine.hand_levels import HandLevels
    from jackdaw.engine.rng import PseudoRandom


def eval_card(
    card: Card,
    context: dict[str, Any] | None = None,
    *,
    rng: PseudoRandom | None = None,
    probabilities_normal: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a single card's scoring contribution.

    Matches ``eval_card`` (common_events.lua:580).

    The return dict contains only fields with non-zero values (matching
    the source's ``if value > 0 then ret.field = value`` pattern).

    Context keys:
        - ``cardarea``: ``"play"`` or ``"hand"`` — determines which
          scoring methods to call.
        - ``repetition_only``: If true, only check for retrigger seals.
        - ``edition``: If true, only return edition info (for joker
          edition pass in Phase 9a).

    For played cards (``cardarea="play"``):
        Returns: chips, mult, x_mult, p_dollars, edition.

    For held cards (``cardarea="hand"``):
        Returns: h_mult, x_mult (from h_x_mult).

    For repetition check (``repetition_only=True``):
        Returns: seals (with repetitions count).

    Args:
        card: The card to evaluate.
        context: Context dict with cardarea and flags.
        rng: PseudoRandom instance for Lucky Card rolls.
        probabilities_normal: ``G.GAME.probabilities.normal`` (default 1).
    """
    ctx = context or {}
    ret: dict[str, Any] = {}

    # Repetition-only mode: just check seals
    if ctx.get("repetition_only"):
        seals = card.calculate_seal(repetition=True)
        if seals:
            ret["seals"] = seals
        return ret

    cardarea = ctx.get("cardarea")

    # Played cards (cardarea == G.play → "play")
    if cardarea == "play":
        chips = card.get_chip_bonus()
        if chips > 0:
            ret["chips"] = chips

        mult = card.get_chip_mult(
            rng=rng,
            probabilities_normal=probabilities_normal,
        )
        if mult > 0:
            ret["mult"] = mult

        x_mult = card.get_chip_x_mult()
        if x_mult > 0:
            ret["x_mult"] = x_mult

        p_dollars = card.get_p_dollars(
            rng=rng,
            probabilities_normal=probabilities_normal,
        )
        if p_dollars > 0:
            ret["p_dollars"] = p_dollars

        # Joker effects on played cards (calculate_joker stub)
        # Will be: jokers = card.calculate_joker(context)
        # if jokers: ret["jokers"] = jokers

        edition = card.get_edition()
        if edition:
            ret["edition"] = edition

    # Held-in-hand cards (cardarea == G.hand → "hand")
    elif cardarea == "hand":
        h_mult = card.get_chip_h_mult()
        if h_mult > 0:
            ret["h_mult"] = h_mult

        # Source maps h_x_mult to ret.x_mult (not ret.h_x_mult)
        h_x_mult = card.get_chip_h_x_mult()
        if h_x_mult > 0:
            ret["x_mult"] = h_x_mult

        # Joker effects on held cards (calculate_joker stub)

    # Joker area (cardarea == G.jokers → "jokers")
    elif cardarea == "jokers":
        if ctx.get("edition"):
            edition = card.get_edition()
            if edition:
                ret["jokers"] = edition
        # else: calculate_joker stub

    return ret


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Result of the full scoring pipeline."""

    hand_type: str
    """Detected hand type (e.g. ``"Full House"``) or ``"NULL"``."""

    scoring_cards: list[Card]
    """Cards that scored (including Splash/Stone augmentation)."""

    chips: float
    """Final chip value (after all per-card and edition bonuses)."""

    mult: float
    """Final mult value (after all additive and multiplicative bonuses)."""

    total: int
    """``floor(chips * mult)`` — the score added to G.GAME.chips."""

    debuffed: bool
    """True if the hand was blocked by a boss blind."""

    breakdown: list[str] = field(default_factory=list)
    """Step-by-step log for debugging."""

    effects: list[Effect] = field(default_factory=list)
    """State changes emitted while scoring; the caller decides whether to apply them."""

    effect_queue: EffectQueue | None = field(default=None, repr=False)
    """The pass-local queue whose money view produced ``effects``."""

    legacy_dollars_earned: int = 0
    """Jokerless scorer compatibility; full scoring derives money from effects."""

    cards_destroyed: list[Card] = field(default_factory=list)
    """Playing cards destroyed during Phase 11 (Glass shatter, etc.)."""

    @property
    def jokers_removed(self) -> list[Card]:
        """Cards targeted by queued ``DestroyCard`` effects."""
        return [effect.card for effect in self.effects if isinstance(effect, DestroyCard)]

    @property
    def dollars_earned(self) -> int:
        """Non-instant scoring payouts represented by this result's ledger effects."""
        payouts = [
            effect.amount
            for effect in self.effects
            if isinstance(effect, EaseDollars) and not effect.instant
        ]
        return sum(payouts) if payouts else self.legacy_dollars_earned


# ---------------------------------------------------------------------------
# Base scoring pipeline (Phases 1-4, 6-8, 12 without joker effects)
# ---------------------------------------------------------------------------


def score_hand_base(
    played_cards: list[Card],
    held_cards: list[Card],
    hand_levels: Any,  # HandLevels
    blind: Any,  # Blind
    rng: PseudoRandom,
    *,
    probabilities_normal: float = 1.0,
    joker_flags: dict[str, bool] | None = None,
) -> ScoreResult:
    """Score a hand without joker effects.

    Implements Phases 1-4, 6-8, 12 of the scoring pipeline from
    ``state_events.lua:571-1065``.

    `joker_flags` -- `get_hand_eval_flags(jokers)` from a caller that owns
    jokers but wants their SCORING effects excluded. Hand DETECTION is a
    separate axis from joker effects: a Four Fingers owner's 4-card
    straight really is a Straight even when pricing it jokerlessly, so a
    caller ranking such lines must pass this or it will price them as High
    Card and rank them below junk. None = no modifiers (the honest default
    for a genuinely jokerless board).

    Was previously accepted and DISCARDED (`_ = joker_flags`, "reserved for
    future joker flag passing"), which made the signature a lie.
    """
    from jackdaw.engine.hand_eval import evaluate_hand

    dollars = 0
    breakdown: list[str] = []

    # L: state_events.lua:571-612 — detect the hand and form its scoring set.
    eval_result = evaluate_hand(played_cards, jokers=None, flags=joker_flags)
    hand_type = eval_result.detected_hand
    scoring_cards = eval_result.scoring_cards
    poker_hands = eval_result.poker_hands

    if hand_type == "NULL":
        return ScoreResult(
            hand_type="NULL",
            scoring_cards=[],
            chips=0,
            mult=0,
            total=0,
            debuffed=False,
            breakdown=["No hand"],
        )

    # === Phase 3: Boss blind debuff check ===
    # Lua passes G.play.cards (all played cards, not just scoring subset)
    # to debuff_hand (state_events.lua:614).  Matters for The Psychic
    # which checks #cards >= h_size_ge against the full played hand.
    debuffed = blind.debuff_hand(played_cards, poker_hands, hand_type)
    if debuffed:
        return ScoreResult(
            hand_type=hand_type,
            scoring_cards=scoring_cards,
            chips=0,
            mult=0,
            total=0,
            debuffed=True,
            breakdown=[f"Hand blocked by {blind.name}"],
        )

    # === Phase 3b: The Arm — demote played hand type by 1 (min L1) ===
    if (
        getattr(blind, "name", "") == "The Arm"
        and not getattr(blind, "disabled", False)
        and hand_levels[hand_type].level > 1
    ):
        hand_levels.level_up(hand_type, amount=-1)

    # === Phase 4: Base chips/mult from hand level ===
    base_chips, base_mult = hand_levels.get(hand_type)
    hand_chips = float(base_chips)
    mult = float(base_mult)
    breakdown.append(
        f"Base: {hand_type} L{hand_levels[hand_type].level}"
        f" -> {int(hand_chips)} chips, {int(mult)} mult"
    )

    # Record play
    hand_levels.record_play(hand_type)

    # === Phase 6: Blind modify_hand (The Flint) ===
    new_mult, new_chips, modified = blind.modify_hand(mult, int(hand_chips))
    if modified:
        mult = float(new_mult)
        hand_chips = float(new_chips)
        breakdown.append(f"Blind modify: {int(hand_chips)} chips, {int(mult)} mult")

    # === Phase 7: Per scored card (with retriggers) ===
    for card in scoring_cards:
        if card.debuff:
            continue

        # Collect retriggers
        reps = [1]  # base evaluation
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card,
                {"cardarea": "play"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )

            # Apply effects in source order (state_events.lua:702-776)
            if "chips" in ev:
                hand_chips += ev["chips"]
            if "mult" in ev:
                mult += ev["mult"]
            if "p_dollars" in ev:
                dollars += ev["p_dollars"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]
            if "edition" in ev:
                ed = ev["edition"]
                hand_chips += ed.get("chip_mod", 0)
                mult += ed.get("mult_mod", 0)
                mult *= ed.get("x_mult_mod", 1)

    # === Phase 8: Per held card (with retriggers) ===
    for card in held_cards:
        if card.debuff:
            continue

        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for _rep_idx in range(len(reps)):
            ev = eval_card(
                card,
                {"cardarea": "hand"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )

            # Apply in source order (state_events.lua:845-862)
            if "h_mult" in ev:
                mult += ev["h_mult"]
            if "x_mult" in ev:
                mult *= ev["x_mult"]

    # === Phase 12: Final score ===
    total = math.floor(hand_chips * mult)
    breakdown.append(f"Final: {int(hand_chips)} x {mult:.1f} = {total}")

    return ScoreResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        chips=hand_chips,
        mult=mult,
        total=total,
        debuffed=False,
        breakdown=breakdown,
        legacy_dollars_earned=dollars,
    )


# ---------------------------------------------------------------------------
# Full scoring pipeline WITH joker effects (Phases 1-9, 12)
# ---------------------------------------------------------------------------


def _apply_individual_joker_effects(
    effects: list[dict[str, Any]],
    hand_chips: float,
    mult: float,
    dollars: int,
) -> tuple[float, float, int]:
    """Apply a list of individual-context joker results to running totals.

    Matches the effect application order from state_events.lua:704-776:
    chips → mult → p_dollars → dollars → extra → x_mult → edition.
    """
    for eff in effects:
        if "chips" in eff:
            hand_chips += eff["chips"]
        if "mult" in eff:
            mult += eff["mult"]
        if "p_dollars" in eff:
            dollars += eff["p_dollars"]
        if "dollars" in eff:
            dollars += eff["dollars"]
        if "x_mult" in eff:
            mult *= eff["x_mult"]
        if "edition" in eff:
            ed = eff["edition"]
            hand_chips += ed.get("chip_mod", 0)
            mult += ed.get("mult_mod", 0)
            mult *= ed.get("x_mult_mod", 1)
    return hand_chips, mult, dollars


def _apply_held_joker_effects(
    effects: list[dict[str, Any]],
    mult: float,
    dollars: int,
) -> tuple[float, int]:
    """Apply held-card joker effects. Order: dollars → h_mult → x_mult."""
    for eff in effects:
        if "dollars" in eff:
            dollars += eff["dollars"]
        if "h_mult" in eff:
            mult += eff["h_mult"]
        if "x_mult" in eff:
            mult *= eff["x_mult"]
    return mult, dollars


def score_hand(
    played_cards: list[Card],
    held_cards: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels,
    blind: Blind,
    rng: PseudoRandom,
    *,
    probabilities_normal: float | None = None,
    game_state: dict[str, Any] | None = None,
    back_key: str | None = None,
    blind_chips: int = 0,
) -> ScoreResult:
    """Full scoring pipeline with joker effects (Phases 1-14).

    Args:
        played_cards: Cards played from hand.
        held_cards: Cards remaining in hand.
        jokers: Joker cards in order (left to right).
        hand_levels: HandLevels instance for base chips/mult.
        blind: Current Blind.
        rng: PseudoRandom instance.
        probabilities_normal: Explicit probability numerator override. When
            omitted, read it from ``game_state``.
        game_state: Pre-computed game state dict.
        back_key: Deck back key (e.g. ``'b_plasma'`` for Plasma Deck).
        blind_chips: Blind chip target (for Mr. Bones save check).
    """
    from jackdaw.engine.hand_eval import evaluate_hand
    from jackdaw.engine.jokers import JokerContext, calculate_joker

    gs = game_state or {}
    queue = EffectQueue(gs)
    breakdown: list[str] = []

    overrides = (
        None if probabilities_normal is None else {"probabilities_normal": probabilities_normal}
    )
    snapshot = StateView(gs, jokers=jokers, overrides=overrides, queue=queue)
    probabilities_normal = snapshot.probabilities_normal

    # L: state_events.lua:571-612 — detect the hand and form its scoring set.
    # `jokers`, NOT None: evaluate_hand derives every hand-DETECTION flag
    # (four_fingers / shortcut / smeared / splash) from this list. Passing
    # None here -- a copy-paste carryover from score_hand_base, which is
    # jokerless BY DESIGN -- left Four Fingers, Shortcut and Smeared Joker
    # completely inert in the real pipeline (in-game, env, and every solver
    # label) while their hand_eval unit tests passed. Debuffed jokers are
    # excluded inside get_hand_eval_flags (find_joker semantics), so the
    # raw list is the correct argument.
    eval_result = evaluate_hand(played_cards, jokers=jokers)
    hand_type = eval_result.detected_hand
    scoring_cards = eval_result.scoring_cards
    poker_hands = eval_result.poker_hands

    if hand_type == "NULL":
        return ScoreResult(
            hand_type="NULL",
            scoring_cards=[],
            chips=0,
            mult=0,
            total=0,
            debuffed=False,
            breakdown=["No hand"],
            effects=queue.effects,
            effect_queue=queue,
        )

    # L: state_events.lua:574-578 — update every hand statistic once before
    # the blind branch, including visibility and last_hand_played.
    hand_levels.record_play(hand_type)
    hand_levels[hand_type].visible = True
    # Solver probes pass cloned areas against the live gs. Only the actual
    # play-area/hand objects authorize this synchronous Lua state write.
    if played_cards is gs.get("played_cards_area") and held_cards is gs.get("hand"):
        gs["last_hand_played"] = hand_type

    # === Phase 3: Boss blind debuff check ===
    # Lua passes G.play.cards (all played cards, not just scoring subset)
    # to debuff_hand (state_events.lua:614).  Matters for The Psychic
    # which checks #cards >= h_size_ge against the full played hand.
    debuffed = blind.debuff_hand(
        played_cards,
        poker_hands,
        hand_type,
        gs=gs,
        queue=queue,
    )
    if debuffed:
        # L: state_events.lua:997-1028 — blocked-hand Joker pass.
        for joker in jokers:
            if joker.debuff:
                continue
            ctx = JokerContext(
                debuffed_hand=True,
                blind=blind,
                jokers=jokers,
                full_hand=played_cards,
                scoring_hand=scoring_cards,
                scoring_name=hand_type,
                poker_hands=poker_hands,
                game=snapshot,
                queue=queue,
            )
            result = calculate_joker(joker, ctx)
            if result and result.dollars:
                queue.add(EaseDollars(amount=result.dollars, buffered=True, source=joker))

        # L: state_events.lua:1068-1075 — after runs outside both scoring
        # branches, so blocked hands still decay Ice Cream/Seltzer.
        for joker in jokers:
            if joker.debuff:
                continue
            after_ctx = JokerContext(
                after=True,
                blind=blind,
                jokers=jokers,
                full_hand=played_cards,
                scoring_hand=scoring_cards,
                scoring_name=hand_type,
                poker_hands=poker_hands,
                game=snapshot,
                queue=queue,
            )
            calculate_joker(joker, after_ctx)
        return ScoreResult(
            hand_type=hand_type,
            scoring_cards=scoring_cards,
            chips=0,
            mult=0,
            total=0,
            debuffed=True,
            breakdown=[f"Hand blocked by {blind.name}"],
            effects=queue.effects,
            effect_queue=queue,
        )

    # L: blind.lua:550-559 — The Arm mutates before base values are read.
    if (
        getattr(blind, "name", "") == "The Arm"
        and not getattr(blind, "disabled", False)
        and hand_levels[hand_type].level > 1
    ):
        blind.triggered = True
        hand_levels.level_up(hand_type, amount=-1)

    # L: state_events.lua:580-612 — augment the scoring hand. The engine's
    # intentional NEW-P5-1-05 divergence keeps selection order here.
    # REDUNDANT since Phase 1-2 started passing `jokers` to evaluate_hand
    # (which applies the same augmentation from the same flag), but kept:
    # both produce exactly `played_cards` in played order, so this is an
    # idempotent restatement rather than a double-apply, and it is the only
    # Splash path that survived the jokers=None bug -- i.e. the behaviour
    # every existing Splash test and the K1 kicker fixtures were written
    # against. Pinned by TestHandEvalFlagsIntegration::
    # test_splash_still_scores_all_played_cards.
    splash_active = any(
        getattr(j, "center_key", None) == "j_splash" and not getattr(j, "debuff", False)
        for j in jokers
    )
    if splash_active:
        scoring_cards = list(played_cards)

    # L: state_events.lua:615-642 — base values, before pass, then reread.
    base_chips, base_mult = hand_levels.get(hand_type)
    hand_chips = float(base_chips)
    mult = float(base_mult)
    breakdown.append(
        f"Base: {hand_type} L{hand_levels[hand_type].level}"
        f" -> {int(hand_chips)} chips, {int(mult)} mult"
    )

    # Shared context fields (lightweight — references snapshot, not copies)
    _shared = dict(
        full_hand=played_cards,
        scoring_hand=scoring_cards,
        scoring_name=hand_type,
        poker_hands=poker_hands,
        jokers=jokers,
        rng=rng,
        hand_levels=hand_levels,
        blind=blind,
        held_cards=held_cards,
        game=snapshot,
        queue=queue,
    )

    # === Phase 5: "before" joker pass ===
    # Collect dollars as well as level_up: this pass used to keep ONLY
    # level_up and drop every other field on the floor, so any before-context
    # joker that pays out (To Do List) silently earned nothing.
    for joker in jokers:
        if joker.debuff:
            continue
        ctx = JokerContext(before=True, **_shared)
        result = calculate_joker(joker, ctx)
        if result:
            if result.dollars:
                queue.add(EaseDollars(amount=result.dollars, buffered=True, source=joker))
            if result.copy_held_card is not None:
                # L: card.lua:3501-3511 — DNA emplaces synchronously in hand,
                # before state_events.lua:782 begins the held-card loop.
                live_hand = gs.get("hand")
                if held_cards is live_hand:
                    copied = lifecycle.copy_card(gs, result.copy_held_card)
                    lifecycle.add_playing_cards(gs, [copied], "hand", notify=False)
                else:
                    copied = lifecycle.copy_card({}, result.copy_held_card)
                    held_cards.append(copied)
            if result.level_up:
                hand_levels.level_up(hand_type)
                base_chips, base_mult = hand_levels.get(hand_type)
                hand_chips = float(base_chips)
                mult = float(base_mult)

    # Vampire's queued cosmetic event clears this guard after every Vampire
    # has visited the before pass (card.lua:3472-3477). No later scoring
    # observer reads it, so the synchronous port coalesces that event here.
    for card in scoring_cards:
        if getattr(card, "vampired", False):
            del card.vampired

    # L: state_events.lua:643-647 — blind modification precedes card scoring.
    new_mult, new_chips, modified = blind.modify_hand(mult, int(hand_chips))
    if modified:
        mult = float(new_mult)
        hand_chips = float(new_chips)
        breakdown.append(f"Blind modify: {int(hand_chips)} chips, {int(mult)} mult")

    # L: state_events.lua:648-778 — scored cards and their repetitions.
    for card in scoring_cards:
        if card.debuff:
            blind.triggered = True
            continue

        # 7a: Collect retriggers (seal + joker)
        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for joker in jokers:
            if joker.debuff:
                continue
            rep_ctx = JokerContext(
                repetition=True,
                cardarea="play",
                other_card=card,
                **_shared,
            )
            rep_result = calculate_joker(joker, rep_ctx)
            if rep_result and rep_result.repetitions > 0:
                for _ in range(rep_result.repetitions):
                    reps.append(rep_result)

        # 7b-d: Each repetition
        for _rep in reps:
            # Card's own effects
            ev = eval_card(
                card,
                {"cardarea": "play"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )
            effects: list[dict[str, Any]] = [ev]

            # Joker individual effects on this card
            for joker in jokers:
                if joker.debuff:
                    continue
                ind_ctx = JokerContext(
                    individual=True,
                    cardarea="play",
                    other_card=card,
                    **_shared,
                )
                ind_result = calculate_joker(joker, ind_ctx)
                if ind_result:
                    eff: dict[str, Any] = {}
                    if ind_result.chips:
                        eff["chips"] = ind_result.chips
                    if ind_result.mult:
                        eff["mult"] = ind_result.mult
                    if ind_result.x_mult:
                        eff["x_mult"] = ind_result.x_mult
                    if ind_result.dollars:
                        eff["dollars"] = ind_result.dollars
                    if eff:
                        eff["card"] = joker
                        effects.append(eff)

            # Lua clears this once per repetition after every Joker has seen
            # the scored card's individual context (state_events.lua:700).
            card.lucky_trigger = False

            for effect in effects:
                amount = effect.get("p_dollars", 0) + effect.get("dollars", 0)
                if amount:
                    queue.add(EaseDollars(amount=amount, buffered=True))

            hand_chips, mult, _ = _apply_individual_joker_effects(
                effects,
                hand_chips,
                mult,
                0,
            )

    # L: state_events.lua:782-872 — held cards include synchronous DNA copies.
    for card in held_cards:
        if card.debuff:
            continue

        # 8a: Collect retriggers
        reps = [1]
        seal_result = card.calculate_seal(repetition=True)
        if seal_result and seal_result.get("repetitions"):
            for _ in range(seal_result["repetitions"]):
                reps.append(seal_result)

        for joker in jokers:
            if joker.debuff:
                continue
            rep_ctx = JokerContext(
                repetition=True,
                cardarea="hand",
                other_card=card,
                **_shared,
            )
            rep_result = calculate_joker(joker, rep_ctx)
            if rep_result and rep_result.repetitions > 0:
                for _ in range(rep_result.repetitions):
                    reps.append(rep_result)

        # 8b-c: Each repetition
        for _rep in reps:
            ev = eval_card(
                card,
                {"cardarea": "hand"},
                rng=rng,
                probabilities_normal=probabilities_normal,
            )
            effects_h: list[dict[str, Any]] = [ev]

            # Joker individual effects on held card
            for joker in jokers:
                if joker.debuff:
                    continue
                ind_ctx = JokerContext(
                    individual=True,
                    cardarea="hand",
                    other_card=card,
                    **_shared,
                )
                ind_result = calculate_joker(joker, ind_ctx)
                if ind_result:
                    eff_h: dict[str, Any] = {}
                    if ind_result.h_mult:
                        eff_h["h_mult"] = ind_result.h_mult
                    if ind_result.x_mult:
                        eff_h["x_mult"] = ind_result.x_mult
                    if ind_result.dollars:
                        eff_h["dollars"] = ind_result.dollars
                    if eff_h:
                        effects_h.append(eff_h)

            for effect in effects_h:
                amount = effect.get("dollars", 0)
                if amount:
                    queue.add(EaseDollars(amount=amount, buffered=True))

            mult, _ = _apply_held_joker_effects(
                effects_h,
                mult,
                0,
            )

    # L: state_events.lua:873-944 — Joker main pass, left to right.
    for joker in jokers:
        if joker.debuff:
            continue

        # 9a: Edition additive (chip_mod, mult_mod) BEFORE joker effect
        edition = joker.get_edition()
        if edition:
            hand_chips += edition.get("chip_mod", 0)
            mult += edition.get("mult_mod", 0)

        # 9b: Main joker effect
        main_ctx = JokerContext(joker_main=True, **_shared)
        result = calculate_joker(joker, main_ctx)
        if result:
            if result.mult_mod:
                mult += result.mult_mod
            if result.chip_mod:
                hand_chips += result.chip_mod
            if result.Xmult_mod:
                mult *= result.Xmult_mod
            if result.dollars:
                queue.add(EaseDollars(amount=result.dollars, buffered=True, source=joker))

        # 9c: Joker-on-joker (other_joker context)
        for other in jokers:
            if other is joker or other.debuff:
                continue
            j2j_ctx = JokerContext(other_joker=joker, **_shared)
            j2j = calculate_joker(other, j2j_ctx)
            if j2j:
                if j2j.mult_mod:
                    mult += j2j.mult_mod
                if j2j.chip_mod:
                    hand_chips += j2j.chip_mod
                if j2j.Xmult_mod:
                    mult *= j2j.Xmult_mod

        # 9d: Edition multiplicative (x_mult_mod) AFTER joker effect
        if edition and "x_mult_mod" in edition:
            mult *= edition["x_mult_mod"]

    # L: state_events.lua:946-948 — deck-back final scoring.
    if back_key:
        from jackdaw.engine.back import Back as _Back

        _back_effect = _Back(back_key).trigger_effect(
            "final_scoring_step", chips=hand_chips, mult=mult
        )
        if _back_effect:
            prev_chips, prev_mult = hand_chips, mult
            hand_chips = _back_effect["chips"]
            mult = _back_effect["mult"]
            breakdown.append(
                f"Plasma: ({int(prev_chips + prev_mult)}) / 2"
                f" -> {int(hand_chips)} chips, {int(mult)} mult"
            )

    # L: state_events.lua:950-996 — destruction decisions and notifications.
    cards_destroyed: list[Card] = []
    for sc in scoring_cards:
        if sc.debuff:
            continue
        destroyed = False
        # Joker destroying_card checks (Sixth Sense, etc.)
        for joker in jokers:
            if joker.debuff:
                continue
            dest_ctx = JokerContext(
                destroying_card=sc,
                **_shared,
            )
            dest_result = calculate_joker(joker, dest_ctx)
            if dest_result and dest_result.remove:
                destroyed = True
                break
        # Glass Card self-shatter: 1 in (1/probabilities_normal * 4) chance
        if not destroyed and sc.ability.get("name") == "Glass Card":
            if rng.random("glass") < probabilities_normal / 4:
                destroyed = True
        if destroyed:
            if sc.ability.get("name") == "Glass Card":
                sc.shattered = True
            else:
                sc.destroyed = True
            cards_destroyed.append(sc)

    # Notify jokers of destruction (Caino, Glass Joker xMult growth) at the
    # same scoring-time observation point as before Phase 3.
    if cards_destroyed:
        for joker in jokers:
            if joker.debuff:
                continue
            dest_notify_ctx = JokerContext(
                cards_destroyed=cards_destroyed,
                **_shared,
            )
            calculate_joker(joker, dest_notify_ctx)

        queue.add(DestroyPlayingCards(cards=cards_destroyed, notify=False))

    # Removal is an F2 effect, never applied here: the solver calls score_hand
    # on cloned cards with live gs and must not delete real cards.

    # L: state_events.lua:1029-1066 — calculate the queued score delta.
    total = math.floor(hand_chips * mult)
    breakdown.append(f"Final: {int(hand_chips)} x {mult:.1f} = {total}")
    queue.add(AddChips(amount=total))

    # L: state_events.lua:1068-1075 — after pass always runs.
    for joker in jokers:
        if joker.debuff:
            continue
        after_ctx = JokerContext(after=True, **_shared)
        calculate_joker(joker, after_ctx)

    # L: state_events.lua:1077-1084 — queued permanent-debuff modifier.
    # Challenge mode: debuff all played cards after scoring.
    # Implemented as a flag check — no joker interaction.

    return ScoreResult(
        hand_type=hand_type,
        scoring_cards=scoring_cards,
        chips=hand_chips,
        mult=mult,
        total=total,
        debuffed=False,
        breakdown=breakdown,
        effects=queue.effects,
        effect_queue=queue,
        cards_destroyed=cards_destroyed,
    )
