"""Shop population and related helpers.

Ports:
- ``create_card_for_shop`` (``UI_definitions.lua:742``) — weighted-random type
  selection and Illusion voucher playing-card modifiers.
- ``get_pack`` (``common_events.lua:1944``) — booster pack selection with
  first-shop Buffoon guarantee.
- ``populate_shop`` — unified shop build (joker slots + voucher + boosters).
- ``sell_card`` (``card.lua:1590``) — sell a joker/consumable.

Source references
-----------------
- UI_definitions.lua:742   — ``create_card_for_shop``
- common_events.lua:1944   — ``get_pack``
- common_events.lua:2082   — ``create_card`` (called after type is selected)
- card.lua:1590             — ``Card:sell_card``
- game.lua:3099             — shop setup loop
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jackdaw.engine import lifecycle, read
from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS
from jackdaw.engine.effects import DestroyCard, DisableBlind, EaseDollars, EffectQueue

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom

# ---------------------------------------------------------------------------
# Card type constants
# ---------------------------------------------------------------------------

TYPE_JOKER = "Joker"
TYPE_TAROT = "Tarot"
TYPE_PLANET = "Planet"
TYPE_SPECTRAL = "Spectral"
TYPE_PLAYING_CARD = "PlayingCard"

# ---------------------------------------------------------------------------
# Card type selection — UI_definitions.lua:742
# ---------------------------------------------------------------------------


def select_shop_card_type(
    rng: PseudoRandom,
    ante: int,
    *,
    joker_rate: float = 20.0,
    tarot_rate: float = 4.0,
    planet_rate: float = 4.0,
    spectral_rate: float = 0.0,
    playing_card_rate: float = 0.0,
) -> str:
    """Select what type of card fills a shop joker slot.

    Mirrors the weighted-random branch in ``create_card_for_shop``
    (``UI_definitions.lua:742``).  The caller is responsible for passing
    the correct rates (modified by vouchers/deck/etc. before calling).

    RNG: one draw from stream ``'cdt' + str(ante)``.

    Default rates and base probabilities (total = 28):

    +---------------+------+----------+
    | Type          | Rate | Base %   |
    +===============+======+==========+
    | Joker         | 20   | ~71.4 %  |
    +---------------+------+----------+
    | Tarot         |  4   | ~14.3 %  |
    +---------------+------+----------+
    | Planet        |  4   | ~14.3 %  |
    +---------------+------+----------+
    | Spectral      |  0   | 0 %      |
    +---------------+------+----------+
    | Playing Card  |  0   | 0 %      |
    +---------------+------+----------+

    Voucher rate examples (passed in as modified parameters):

    * Tarot Merchant — ``tarot_rate = 9.6``
    * Planet Merchant — ``planet_rate = 9.6``
    * Magic Trick — ``playing_card_rate = 4``
    * Ghost Deck — sets ``spectral_rate`` at run start

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Advances
        stream ``'cdt' + str(ante)`` by one draw.
    ante:
        Current ante number.
    joker_rate, tarot_rate, planet_rate, spectral_rate, playing_card_rate:
        Weights for each card type.  Pass voucher-modified values from the
        caller.

    Returns
    -------
    str
        One of ``'Joker'``, ``'Tarot'``, ``'Planet'``, ``'Spectral'``, or
        ``'PlayingCard'``.
    """
    total = joker_rate + tarot_rate + planet_rate + spectral_rate + playing_card_rate

    poll = rng.random("cdt" + str(ante)) * total

    if poll < joker_rate:
        return TYPE_JOKER
    poll -= joker_rate

    if poll < tarot_rate:
        return TYPE_TAROT
    poll -= tarot_rate

    if poll < planet_rate:
        return TYPE_PLANET
    poll -= planet_rate

    if poll < spectral_rate:
        return TYPE_SPECTRAL

    return TYPE_PLAYING_CARD


# ---------------------------------------------------------------------------
# Illusion voucher modifiers — UI_definitions.lua:~780
# ---------------------------------------------------------------------------

# Probability thresholds
_ILLUSION_ENH_THRESHOLD = 0.6
_ILLUSION_EDI_CHANCE_THRESHOLD = 0.8  # roll > 0.8 → get edition (20% chance)

# Edition distribution for Illusion: Foil 50%, Holo 35%, Poly 15%
# Evaluated top-down: Poly > 0.85, Holo > 0.5, else Foil
_ILLUSION_POLY_THRESHOLD = 0.85
_ILLUSION_HOLO_THRESHOLD = 0.5


def roll_illusion_modifiers(
    rng: PseudoRandom,
    ante: int,
    *,
    include_edition: bool = True,
) -> dict[str, Any]:
    """Roll Illusion voucher modifiers for a playing card drawn from the shop.

    Mirrors the Illusion-specific path in ``create_card_for_shop``
    (``UI_definitions.lua``).

    Probabilities
    ~~~~~~~~~~~~~
    * **40%** the card is created as Enhanced; otherwise Base.
    * **20%** the card receives a random edition:
      Foil 50 %, Holo 35 %, Polychrome 15 %.
    All draws use the single ``illusion`` stream: type, edition gate, then
    edition kind when the gate succeeds (UI_definitions.lua:772,786-793).

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    Returns
    -------
    dict
        Dict with zero or more of the following keys:

        * ``'card_type'`` : ``'Enhanced'`` or ``'Base'``
        * ``'edition'`` : dict — e.g. ``{'foil': True}``
    """
    del ante
    result: dict[str, Any] = {
        "card_type": "Enhanced" if rng.random("illusion") > _ILLUSION_ENH_THRESHOLD else "Base"
    }
    if not include_edition:
        return result
    edi_chance = rng.random("illusion")
    if edi_chance > _ILLUSION_EDI_CHANCE_THRESHOLD:
        edi_roll = rng.random("illusion")
        if edi_roll > _ILLUSION_POLY_THRESHOLD:
            result["edition"] = {"polychrome": True}
        elif edi_roll > _ILLUSION_HOLO_THRESHOLD:
            result["edition"] = {"holo": True}
        else:
            result["edition"] = {"foil": True}

    return result


# ---------------------------------------------------------------------------
# Booster pool — module-level cache
# ---------------------------------------------------------------------------

_BOOSTER_POOL: list[str] = CENTER_POOLS.get("Booster", [])

# First-shop Buffoon guarantee: Lua uses math.random(1, 2) which is
# non-deterministic; we deterministically return _1 as the fallback.
_FIRST_SHOP_BUFFOON_PACK = "p_buffoon_normal_1"
_FIRST_SHOP_BUFFOON_KEY = "first_shop_buffoon"


# ---------------------------------------------------------------------------
# get_pack — common_events.lua:1944
# ---------------------------------------------------------------------------


def get_pack(
    rng: PseudoRandom,
    ante: int,
    key: str = "shop_pack",
    *,
    first_shop: bool = False,
    banned_keys: set[str] | None = None,
) -> str:
    """Select a booster pack type.

    Mirrors ``get_pack`` (``common_events.lua:1944``).

    First-shop guarantee
    ~~~~~~~~~~~~~~~~~~~~
    When *first_shop* is ``True`` and ``'p_buffoon_normal_1'`` is not in
    *banned_keys*, the function returns ``'p_buffoon_normal_1'`` immediately
    without consuming an RNG draw.  The Lua source picks variant 1 or 2 via
    the non-deterministic ``math.random(1, 2)``; we always return variant 1
    as the deterministic equivalent.

    The caller is responsible for tracking when the guarantee has been
    consumed (see :func:`populate_shop`).

    Normal selection
    ~~~~~~~~~~~~~~~~
    Compute cumulative weight of all non-banned packs, draw
    ``rng.random(key + str(ante)) * total_weight``, walk the pool.

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.  Advances
        stream ``key + str(ante)`` by one draw (skipped when the first-shop
        guarantee fires).
    ante:
        Current ante number.
    key:
        RNG stream base key.  ``'shop_pack'`` for standard shop usage.
    first_shop:
        When ``True``, apply the first-shop Buffoon guarantee (if not
        banned).  Caller must pass ``True`` only for the very first booster
        slot of a run.
    banned_keys:
        Set of banned center keys.  Banned packs are excluded from both the
        guarantee and the weighted draw.

    Returns
    -------
    str
        A key from ``CENTER_POOLS['Booster']``.  Falls back to the last
        eligible pack if the RNG lands exactly on the cumulative total.
    """
    banned: set[str] = banned_keys or set()

    # -- First-shop Buffoon guarantee (game.lua / common_events.lua:1945) --
    if first_shop and _FIRST_SHOP_BUFFOON_PACK not in banned:
        return _FIRST_SHOP_BUFFOON_PACK

    # -- Weighted random walk --
    cume = sum(BOOSTERS[k].weight for k in _BOOSTER_POOL if k not in banned)

    poll = rng.random(key + str(ante)) * cume

    it = 0.0
    for k in _BOOSTER_POOL:
        if k in banned:
            continue
        w = BOOSTERS[k].weight
        it += w
        if it >= poll and it - w <= poll:
            return k

    # Floating-point edge: poll == cume exactly → last eligible pack
    for k in reversed(_BOOSTER_POOL):
        if k not in banned:
            return k
    return _BOOSTER_POOL[-1]  # unreachable if pool is non-empty


# ---------------------------------------------------------------------------
# populate_shop — game.lua:3099 / UI_definitions.lua:742
# ---------------------------------------------------------------------------

# key_append used by create_card when called from the shop joker area
_SHOP_APPEND = "sho"

# key_appends for tag-forced joker creates (vanilla: Uncommon Tag 'uta',
# Rare Tag 'rta') — their own RNG streams, so a tag-forced joker is
# generated on the spot and does NOT consume the shop pool sequence.
_TAG_APPEND_BY_RARITY = {2: "uta", 3: "rta"}


def create_shop_slot_card(
    rng: PseudoRandom,
    ante: int,
    game_state: dict,
) -> Card:
    """Create one shop card-slot item, applying pending shop-joker tags.

    Mirrors ``create_card_for_shop`` (``UI_definitions.lua:742``):

    * ``store_joker_create`` (Rare/Uncommon Tag): an un-consumed tag
      REPLACES the normal type selection for this slot — the type poll does
      not happen; a Joker of the forced rarity is created directly.
      VERIFIED in real Balatro (2026-07-06): the forced joker is generated
      on the spot on its OWN RNG stream (vanilla appends ``'uta'``/``'rta'``),
      not drawn from the shop pool sequence — the next normal shop joker is
      unaffected by the tag create.
    * ``store_joker_modify`` (Foil/Holo/Polychrome/Negative Tag): applies to
      the first base-edition Joker created after the tag was acquired —
      sets the edition AND makes the card free (vanilla wording: "Next base
      edition shop Joker is free and becomes X").

    One tag per card, FIFO acquisition order, consumed on use.  Used by both
    :func:`populate_shop` and the reroll path so tag behavior can't diverge.
    """
    from jackdaw.engine.card_factory import create_card
    from jackdaw.engine.tags import fire_tag_context

    gs = game_state

    fired = fire_tag_context(gs, "store_joker_create", first_only=True, rng=rng)
    if fired:
        _entry, result = fired[0]
        card = create_card(
            "Joker",
            rng,
            ante,
            area="shop",
            append=_TAG_APPEND_BY_RARITY.get(result.force_rarity, _SHOP_APPEND),
            forced_rarity=result.force_rarity,
            game_state=gs,
        )
    else:
        card_type = select_shop_card_type(
            rng,
            ante,
            joker_rate=gs.get("joker_rate", 20.0),
            tarot_rate=gs.get("tarot_rate", 4.0),
            planet_rate=gs.get("planet_rate", 4.0),
            spectral_rate=gs.get("spectral_rate", 0.0),
            playing_card_rate=gs.get("playing_card_rate", 0.0),
        )
        illusion: dict[str, Any] | None = None
        if read.has_voucher(gs, "v_illusion"):
            # The type expression is evaluated after the cdt draw and before
            # create_card (UI_definitions.lua:766-776), even when another
            # weighted type ultimately wins.
            illusion = roll_illusion_modifiers(
                rng,
                ante,
                include_edition=card_type == TYPE_PLAYING_CARD,
            )
            if card_type == TYPE_PLAYING_CARD:
                card_type = illusion["card_type"]
        elif card_type == TYPE_PLAYING_CARD:
            card_type = "Base"
        card = create_card(
            card_type,
            rng,
            ante,
            area="shop",
            # L: UI_definitions.lua:776 — create_card_for_shop does not pass
            # soulable; only booster generation opts into hidden replacements.
            soulable=False,
            append=_SHOP_APPEND,
            game_state=gs,
        )
        if illusion and card_type in ("Base", "Enhanced") and illusion.get("edition"):
            card.set_edition(gs, illusion["edition"])

    # Tag callbacks run with the card already in the shop area in Lua. Make
    # that area visible before repricing so couponed cards resolve to $0.
    lifecycle.emplace(gs, card, "shop_cards")

    if fired:
        card.ability["couponed"] = True
        card.set_cost(gs)

    card_set = card.ability.get("set", "") if isinstance(card.ability, dict) else ""
    if card_set == "Joker" and not card.edition:
        fired = fire_tag_context(gs, "store_joker_modify", first_only=True, rng=rng)
        if fired:
            _entry, result = fired[0]
            card.set_edition(gs, {result.force_edition: True})
            card.ability["couponed"] = True
            card.set_cost(gs)

    return card


def populate_shop(
    rng: PseudoRandom,
    ante: int,
    game_state: dict,
) -> dict[str, list[Card] | Card | None]:
    """Build the full shop for the current round.

    Mirrors the shop-setup block in ``game.lua:3099`` and
    ``create_card_for_shop`` (``UI_definitions.lua:742``).

    Flow
    ~~~~
    1. **Joker slots** — for each of ``shop['joker_max']`` (default 2) slots:
       call :func:`select_shop_card_type` then
       :func:`~jackdaw.engine.card_factory.create_card` with
       ``area='shop'`` and ``append='sho'``.
    2. **Voucher** — create a voucher card from
       ``game_state['current_round']['voucher']`` (key pre-determined by
       :func:`~jackdaw.engine.vouchers.get_next_voucher_key`).  ``None`` if
       absent.
    3. **Boosters** — 2 packs via :func:`get_pack` with key
       ``'shop_pack'``.

    .. note::
       Tag hooks: ``store_joker_create`` / ``store_joker_modify`` are applied
       per card slot via :func:`create_shop_slot_card`. ``voucher_add`` and
       ``shop_final_pass`` (Coupon) fire in ``game._populate_shop`` — they
       need the assembled shop lists, which this function returns.

    Parameters
    ----------
    rng:
        Live :class:`~jackdaw.engine.rng.PseudoRandom` instance.
    ante:
        Current ante number.
    game_state:
        Game-state dict.  Relevant keys:

        * ``shop`` → ``joker_max`` (int, default 2)
        * ``joker_rate``, ``tarot_rate``, ``planet_rate``,
          ``spectral_rate``, ``playing_card_rate`` — type-selection weights
        * ``current_round`` → ``voucher`` (str | None)
        * ``banned_keys`` (dict) — passed to :func:`get_pack`
        * All keys forwarded to
          :func:`~jackdaw.engine.card_factory.create_card` via
          *game_state*

    Returns
    -------
    dict
        ``{'jokers': list[Card], 'voucher': Card | None,
        'boosters': list[Card]}``
    """
    from jackdaw.engine.card_factory import create_card, create_voucher

    gs = game_state

    shop_joker_max: int = gs.get("shop", {}).get("joker_max", 2)

    banned_keys: set[str] = set(gs.get("banned_keys") or {})

    # -- 1. Joker slots --
    jokers: list[Card] = []
    for _ in range(shop_joker_max):
        jokers.append(create_shop_slot_card(rng, ante, gs))

    # -- 2. Voucher --
    voucher: Card | None = None
    voucher_key: str | None = gs.get("current_round", {}).get("voucher")
    if voucher_key:
        voucher = create_voucher(voucher_key, game_state=gs)
        voucher.set_cost(gs)

    # -- 3. Boosters (always exactly 2 slots) --
    boosters: list[Card] = []
    for i in range(2):
        first_shop = i == 0 and not gs.get(_FIRST_SHOP_BUFFOON_KEY, False)
        pack_key = get_pack(rng, ante, "shop_pack", first_shop=first_shop, banned_keys=banned_keys)
        # Mark guarantee consumed when it fires (pack returned and not banned)
        if first_shop and _FIRST_SHOP_BUFFOON_PACK not in banned_keys:
            gs[_FIRST_SHOP_BUFFOON_KEY] = True
        pack_card = create_card(
            "Booster",
            rng,
            ante,
            area="shop",
            soulable=False,
            forced_key=pack_key,
            game_state=gs,
        )
        boosters.append(pack_card)

    return {"jokers": jokers, "voucher": voucher, "boosters": boosters}


# ---------------------------------------------------------------------------
# sell_card — card.lua:1590 + button_callbacks.lua:2318
# ---------------------------------------------------------------------------


def sell_card(
    game_state: dict,
    area: str,
    index: int,
) -> dict[str, Any]:
    """Execute a card sale.

    Mirrors ``Card:sell_card`` (``card.lua:1590``) and
    ``G.FUNCS.sell_card`` (``button_callbacks.lua:2318``).

    Flow
    ~~~~
    1. **Eligibility check** — eternal cards and cards not in the joker/
       consumable rows cannot be sold.  Returns
       ``{'ok': False, 'reason': 'eternal'}`` or ``'not_sellable'``.
    2. **Selling-self notification** — call
       ``calculate_joker(card, {selling_self=True})``.
    3. **Selling-card notification** — call
       ``calculate_joker(j, {selling_card=True, card=card})`` for every
       other joker in ``game_state['jokers']``.
    4. **Event half** — award sell value, remove the card, and disable
       Verdant Leaf for a sold Joker. One queue preserves Lua event order.

    Parameters
    ----------
    game_state:
        Mutable game-state dict.  Relevant keys:

        * ``dollars`` (int).
        * ``jokers`` (list[Card]) — active jokers for sell notifications.

    Returns
    -------
    dict
        ``{'ok': True, 'dollars_gained': int}`` on success, or
        ``{'ok': False, 'reason': str}`` on failure.
    """
    from jackdaw.engine.jokers import calculate_joker, context_for

    # -- 1. Eligibility --
    if area not in ("jokers", "consumables"):
        return {"ok": False, "reason": "not_sellable"}
    cards = game_state.get(area, [])
    if index < 0 or index >= len(cards):
        return {"ok": False, "reason": "invalid_index"}
    card = cards[index]
    if card.eternal:
        return {"ok": False, "reason": "eternal"}

    # -- 2. Selling-self notification --
    owned_jokers = game_state.get("jokers", [])
    queue = EffectQueue(game_state)
    calculate_joker(card, context_for(game_state, queue=queue, selling_self=True))

    # -- 3. Selling-card notification to other jokers --
    for joker in owned_jokers:
        if joker is not card:
            calculate_joker(
                joker,
                context_for(
                    game_state,
                    queue=queue,
                    selling_card=True,
                    other_card=card,
                ),
            )

    dollars_gained = card.sell_cost
    queue.add(EaseDollars(amount=dollars_gained))
    queue.add(DestroyCard(card=card))
    blind = game_state.get("blind")
    if (
        card.ability.get("set") == "Joker"
        and blind is not None
        and getattr(blind, "name", "") == "Verdant Leaf"
    ):
        # Nested immediate event in card.lua:1616-1620, after removal.
        queue.add(DisableBlind(order=1))

    # Invisible Joker copies while the sold card is still in the row because
    # its effect was emitted before the queued removal (card.lua:1590).
    queue.apply()

    return {"ok": True, "dollars_gained": dollars_gained}
