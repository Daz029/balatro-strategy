# Engine Phase 4, ticket P4-3: consumables, tags, seals, decks and the missing dispatches

Branch `engine-phase1-state`, after ticket P4-2
(`docs/implementation-tickets/engine-p4-2-callsites.md`). Same reading list and
rules as P4-2. Every state change goes through `jackdaw/engine/effects.py`
(`EffectQueue` / `apply_effects`); add an `Effect` subclass if one is missing
(put it in `EFFECT_TYPES`), never an inline mutation.

## 1. Consumables: `ConsumableResult` -> effects

- Add `consumable_effects(result: ConsumableResult) -> list[Effect]` in
  `consumables.py`. It translates EVERY `ConsumableResult` field, preserving
  today's application order from `game.py::_apply_consumable_result`
  (enhance, change_suit, change_rank, copy_card, destroy, add_seal,
  destroy_jokers, create, dollars, money_set, level_up, add_to_deck,
  add_edition, hand_size_mod). `destroy_jokers` MUST precede `create` (Ankh,
  D35). Mapping:
  - enhance -> `SetEnhancement`; change_suit -> `ChangeSuit`; change_rank ->
    `ChangeRank`; add_seal -> `SetSeal`; add_edition -> `SetEdition`.
  - copy_card (Death) -> `CopyCard(card=source, into=target)`.
  - destroy -> one `DestroyPlayingCards(cards=...)`.
  - destroy_jokers -> one `DestroyCard` per joker.
  - create: `{"type": T, "count": n, "seed": s, "rarity": r, "forced_key": k}`
    -> `CreateCard(set=T, count=n, append=s, rarity=r, forced_key=k)` (room is
    reserved before any RNG: D46). `{"copy_of": c, "strip_edition": b}` (Ankh)
    -> `CopyCard(card=c, area="jokers", strip_edition=b, reset_invis=True)`
    (Lua `card.lua:1445-1448`). The Fool's `Tarot_Planet` set must keep
    working (`CreateCard.area` treats any non-Joker set as consumables).
  - dollars -> `EaseDollars`; money_set -> `SetDollars`.
  - level_up -> one `LevelUpHand` per `(hand, amount)`.
  - add_to_deck: `{"copy_of": c}` (Cryptid) -> `CopyCard(card=c, area="hand",
    notify=True)`; `{"suit", "rank", "enhancement"}` (Familiar, Grim,
    Incantation) -> `CreatePlayingCard(area="hand", suit=..., rank=...,
    enhancement=...)`. Lua notifies `playing_card_joker_effects` ONCE for the
    batch (`card.lua:1250-1290`): add a `notify` batching option or a
    dedicated batch effect so Hologram sees one notification with N cards,
    not N notifications (equal for Hologram's arithmetic today, but the
    contract is one batch; pin it in a test).
  - hand_size_mod -> `ChangeHandSize`.
  - `notify_jokers_consumeable` is a dispatch flag, not a state change: see 2.
- The translation must RAISE on any field it does not know. Pin that with a
  test that iterates `dataclasses.fields(ConsumableResult)`: for each field,
  build a result with only that field set and check the translation yields
  at least one effect (or, for `notify_jokers_consumeable`, is explicitly
  listed as a dispatch flag). A new field added later without a mapping must
  make that test fail.
- `game.py::_use_consumable_card`: `queue = EffectQueue(gs)`,
  `queue.add(consumable_effects(result))`, `queue.apply()`. Delete
  `_apply_consumable_result` and `_resolve_create_descriptors`;
  `card_factory.resolve_destroy_descriptor` is dead too — delete it if
  nothing uses it. Keep the `last_tarot_planet` bookkeeping where it is.
- Lua `G.FUNCS.use_card` (`button_callbacks.lua:2209-2220`): the used card
  leaves its area FIRST (the engine already pops it), then
  `use_consumeable`, then `calculate_joker({using_consumeable = true,
  consumeable = card})` on EVERY joker for EVERY consumable — not just
  Planets. Drop the `notify_jokers_consumeable` gate (keep the field only if
  something else needs it; otherwise delete it with its producers).
  Glass Joker listens here for The Hanged Man (`card.lua:2709-2716`: counts
  Glass Cards among `G.hand.highlighted`). Check the engine's Glass Joker
  handler against that, including how it currently counts Hanged Man's glass
  (via `cards_destroyed` too? Lua's `remove_playing_cards` path counts only
  `shattered` cards, and Hanged Man dissolves rather than shatters, so the
  ONLY Lua path is `using_consumeable`). Fix any double count or miss; pass
  the highlighted cards in the context.

## 2. Tags

- `game.py::_apply_tag_result`: translate to effects and apply through a
  queue: `dollars` -> `EaseDollars`; `create_jokers` -> `CreateCard(set="Joker",
  count=n, rarity="Common", append="top")` (Lua `tag.lua:138`: each creation
  checks `#jokers + joker_buffer < limit` before drawing, i.e. the reserve
  clamp); `level_up` -> `LevelUpHand(hand, levels)`. The other `TagResult`
  fields are consumed by their own contexts (packs, shop, boss reroll); leave
  those flows, but list in a comment which function consumes each field.

## 3. Missing dispatches (C02, C14, D23)

- **`first_hand_drawn` (Certificate, C02)**: Lua `game.lua:3226-3231` fires it
  after the hand is drawn, when `current_round.hands_played == 0 and
  discards_used == 0 and G.GAME.facing_blind`, before `Blind:drawn_to_hand`.
  Dispatch it in `_handle_select_blind` right after `_draw_hand` and the
  hand debuff loop, before the `drawn_to_hand` block, with
  `fire_jokers(gs, queue, first_hand_drawn=True)` + apply. Check
  `facing_blind` is written true when a blind is selected (Phase 1 added the
  field; if nothing writes it, write it in `_handle_select_blind` and clear it
  where Lua does — cite the Lua line). Certificate's card goes to the HAND,
  is debuffed by the blind, the hand is re-sorted (`card.lua:2463-2481`).
- **`open_booster` (Hallucination, C02)**: Lua `Card:open`
  (`card.lua:1797`) fires `calculate_joker({open_booster = true, card = self})`
  after the pack cards are generated. Dispatch it in
  `_handle_open_booster` AND `_open_tag_pack` (tag packs go through
  `Card:open` in Lua too), with `booster=` the pack card where one exists.
- **Anaglyph Deck (C14)**: Lua `evaluate_round` (`state_events.lua:1163`)
  calls `G.GAME.selected_back:trigger_effect({context = 'eval'})`; Anaglyph
  adds a Double Tag when `G.GAME.last_blind.boss` (`back.lua:111-119`).
  `back.py` already returns `{"create_tag": "tag_double"}` for
  `boss_defeated=True`; dispatch it in `_round_won` at the evaluate_round point
  (after the blind is marked defeated, before `calculate_round_earnings`) and
  apply `AddTag(key="tag_double")` through a queue.
- **Verdant Leaf on joker sale (D23)**: `Card:sell_card` (`card.lua:1616-1620`)
  disables the blind when a Joker is sold during Verdant Leaf (nested event:
  `DisableBlind(order=1)`). Add it to the sell flow's queue. Selling outside
  the shop is Phase 6 (legality); do not change which phases allow selling.
- `AddTag` today appends `{"key", "result": None, "blind": None}`. Check
  every reader of `awarded_tags` copes with `result=None` / `blind=None`
  (Double Tag check, tag context polling, obs encoders in `jackdaw/env`).

## 3b. Review findings from P4-2 (fix these too)

- **The real sell path still drops `selling_self` (D23).** P4-2 routed
  `shop.sell_card` (a CardArea-based helper only `tests/engine/test_shop.py`
  calls), but `step(SellCard)` goes through `game.py::_handle_sell_card`,
  which removes the card and fires `selling_card` with no `other_card`, and
  never dispatches `selling_self`. Make ONE sell implementation: Lua
  `G.FUNCS.sell_card` (`button_callbacks.lua:2318`) -> `Card:sell_card`
  (`card.lua:1590`): `selling_self` on the sold card (Invisible's copy and
  Luchador's disable happen here, card still in its row), then
  `selling_card` with `other_card=card` on every other joker, then the event
  half: `ease_dollars(sell_cost)`, removal, Verdant Leaf (joker sold) ->
  `DisableBlind(order=1)`. One queue for the whole sale. Either make
  `_handle_sell_card` call a gs-based `shop.sell_card` (drop its CardArea
  parameter if only tests use it, updating `test_shop.py`) or the reverse —
  there must be a single function. Keep `_require_phase(SHOP)` (sell phases
  are Phase 6).
- **`blind.triggered` is never cleared per play.** Lua
  `play_cards_from_highlighted` sets `G.GAME.blind.triggered = false` before
  anything else (`state_events.lua:455`). Now that a debuffed scoring card
  sets it (P4-2), it must be cleared at the start of `_handle_play_hand`,
  otherwise Matador pays on every later hand of the round. Test: a hand with
  a debuffed scoring card, then a clean hand under the same blind -> Matador
  pays once.
- `tests/engine/test_joker_lifecycle.py:10,119` imports
  `_resolve_create_descriptors`; migrate it to effects when you delete it.

## 4. Regression tests (each must FAIL on `576ff28`; check it)

In `tests/engine/test_effect_pipeline.py`, through `step()`:
- Certificate adds one sealed card to the HAND on the first draw of a round
  (deck size +1), not on later draws, and not twice.
- Hallucination can create a Tarot on booster open (seed or force the roll);
  no RNG drawn with a full consumable row; tag-opened packs dispatch too.
- Anaglyph: beating a boss awards a Double Tag; a small blind does not.
- Diet Cola sold -> Double Tag awarded. Luchador sold during a boss -> blind
  disabled. Invisible Joker sold with enough rounds -> a copy of another
  joker, Negative stripped. Verdant Leaf: selling a joker disables it.
- Ankh with five ordinary jokers -> two remain (survivor + copy, D35); a
  Negative chosen joker's copy is not Negative.
- High Priestess with one free slot creates exactly one Planet and draws RNG
  for one (D46).
- Glass Joker + Hanged Man on two Glass Cards: x_mult rises by exactly
  2 * extra.
- Familiar/Grim/Incantation: one playing_card_added notification per use.

## Rules

As P4-2. Run `uv run pytest tests/engine tests/env -q -p no:cacheprovider` and
`uv run ruff check jackdaw tests`. Do not commit. Report files changed, each
new test and whether you saw it fail on `576ff28`, uncertainties, and bugs
found but not fixed.
