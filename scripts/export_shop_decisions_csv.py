"""Export rich shop evaluation traces as a frame-by-frame CSV.

The rich trace is one JSON object per decision.  This exporter keeps the
decision order and flattens the useful parts of each pre/post state so the
result can be inspected in a spreadsheet or used as the input to a later
steppable viewer.

By default, rows from the ``blind_select``, ``shop``, and ``pack_opening``
phases are exported, preserving both play-blind and skip-blind decisions.
Use ``--all-phases`` when any future phases are also wanted.

Usage::

    uv run python scripts/export_shop_decisions_csv.py
    uv run python scripts/export_shop_decisions_csv.py trace.jsonl -o trace.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "dumped_eval_rich.jsonl"
DECISION_PHASES = frozenset({"blind_select", "shop", "pack_opening"})

FIELDNAMES = (
    "seed",
    "step",
    "phase",
    "ante",
    "ante_after",
    "round",
    "round_after",
    "blind_on_deck",
    "dollars_before",
    "dollars_after",
    "money_delta",
    "joker_slots",
    "joker_count_before",
    "joker_count_after",
    "jokers_before",
    "jokers_after",
    "consumables_before",
    "consumables_after",
    "shop_cards",
    "shop_vouchers",
    "shop_boosters",
    "pack_cards",
    "pack_hand",
    "available_cards_json",
    "action",
    "action_family",
    "action_slot",
    "action_label",
    "n_legal",
    "legal_actions",
    "action_target_kind",
    "picked_card",
    "picked_joker",
    "bought_card",
    "bought_joker",
    "sold_card",
    "opened_booster",
    "used_or_redeemed_card",
    "selected_cards",
    "action_target_json",
    "terminal",
    "won",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _card_name(card: dict[str, Any]) -> str:
    ability = card.get("ability") or {}
    return str(card.get("name") or ability.get("name") or card.get("center_key") or "?")


def _card_view(card: dict[str, Any], slot: int) -> dict[str, Any]:
    """Keep identity and mutable properties useful for a CSV consumer."""
    edition = card.get("edition") or {}
    return {
        "slot": slot,
        "name": _card_name(card),
        "center_key": card.get("center_key"),
        "card_key": card.get("card_key"),
        "set": card.get("set"),
        "cost": card.get("cost"),
        "sell_cost": card.get("sell_cost"),
        "edition": edition.get("type") if isinstance(edition, dict) else edition,
        "debuff": bool(card.get("debuff", False)),
    }


def _card_text(card: dict[str, Any], slot: int) -> str:
    view = _card_view(card, slot)
    text = f"[{slot}] {view['name']}"
    if view["center_key"]:
        text += f" ({view['center_key']})"
    if view["cost"] is not None and view["set"] in {
        "Booster",
        "Joker",
        "Planet",
        "Spectral",
        "Tarot",
        "Voucher",
    }:
        text += f" ${view['cost']}"
    if view["edition"]:
        text += f" [{view['edition']}]"
    if view["debuff"]:
        text += " [debuff]"
    return text


def _cards_text(cards: Iterable[dict[str, Any]]) -> str:
    return " | ".join(_card_text(card, slot) for slot, card in enumerate(cards))


def _available_cards_json(shop: dict[str, Any], pack: dict[str, Any]) -> str:
    """Preserve each source area's local slot numbers in structured output."""
    return _json(
        {
            key: [_card_view(card, slot) for slot, card in enumerate(cards or [])]
            for key, cards in {
                "shop_cards": shop.get("cards"),
                "shop_vouchers": shop.get("vouchers"),
                "shop_boosters": shop.get("boosters"),
                "pack_cards": pack.get("cards"),
            }.items()
        }
    )


def _inventory_text(cards: Iterable[dict[str, Any]]) -> str:
    return _cards_text(cards)


def _target_card(record: dict[str, Any]) -> dict[str, Any] | None:
    target = record.get("action_target") or {}
    card = target.get("card")
    return card if isinstance(card, dict) else None


def _target_text(record: dict[str, Any]) -> str:
    card = _target_card(record)
    if card is None:
        return ""
    target = record.get("action_target") or {}
    return _card_text(card, int(target.get("slot", target.get("action_slot", 0))))


def _selected_cards_text(record: dict[str, Any]) -> str:
    target = record.get("action_target") or {}
    cards = target.get("cards")
    if not isinstance(cards, list):
        return ""
    slots = target.get("slots") or range(len(cards))
    return " | ".join(
        _card_text(card, int(slot)) for slot, card in zip(slots, cards, strict=False)
    )


def _choice_fields(record: dict[str, Any]) -> dict[str, str]:
    family = record.get("action_family", "")
    target = record.get("action_target") or {}
    target_card = _target_card(record)
    target_text = _target_text(record)
    picked_joker = target_text if target_card and target_card.get("set") == "Joker" else ""

    return {
        "action_target_kind": str(target.get("kind") or ""),
        "picked_card": target_text,
        "picked_joker": picked_joker if family == "PickPackCard" else "",
        "bought_card": target_text if family == "BuyCard" else "",
        "bought_joker": target_text
        if family == "BuyCard" and target_card and target_card.get("set") == "Joker"
        else "",
        "sold_card": target_text
        if family in {"SellJoker", "SellJokerExt", "SellConsumable"}
        else "",
        "opened_booster": target_text if family == "OpenBooster" else "",
        "used_or_redeemed_card": target_text
        if family in {"UseConsumable", "SellConsumable", "RedeemVoucher"}
        else "",
        "selected_cards": _selected_cards_text(record),
        "action_target_json": _json(target) if target else "",
    }


def _row(record: dict[str, Any]) -> dict[str, Any]:
    pre = record["pre_state"]
    post = record.get("post_state") or {}
    pre_inventory = pre.get("inventory") or {}
    post_inventory = post.get("inventory") or {}
    pre_shop = pre.get("shop") or {}
    pre_pack = pre.get("pack") or {}
    dollars_before = pre.get("dollars", record.get("dollars", ""))
    dollars_after = post.get("dollars", "")
    row = {
        "seed": record.get("seed", ""),
        "step": record.get("step", ""),
        "phase": pre.get("phase", ""),
        "ante": pre.get("ante", record.get("ante", "")),
        "ante_after": post.get("ante", ""),
        "round": pre.get("round", record.get("round", "")),
        "round_after": post.get("round", ""),
        "blind_on_deck": pre.get("blind_on_deck", ""),
        "dollars_before": dollars_before,
        "dollars_after": dollars_after,
        "money_delta": dollars_after - dollars_before
        if isinstance(dollars_before, (int, float)) and isinstance(dollars_after, (int, float))
        else "",
        "joker_slots": (pre.get("resources") or {}).get("joker_slots", ""),
        "joker_count_before": len(pre_inventory.get("jokers") or []),
        "joker_count_after": len(post_inventory.get("jokers") or []),
        "jokers_before": _inventory_text(pre_inventory.get("jokers") or []),
        "jokers_after": _inventory_text(post_inventory.get("jokers") or []),
        "consumables_before": _inventory_text(pre_inventory.get("consumables") or []),
        "consumables_after": _inventory_text(post_inventory.get("consumables") or []),
        "shop_cards": _cards_text(pre_shop.get("cards") or []),
        "shop_vouchers": _cards_text(pre_shop.get("vouchers") or []),
        "shop_boosters": _cards_text(pre_shop.get("boosters") or []),
        "pack_cards": _cards_text(pre_pack.get("cards") or []),
        "pack_hand": _cards_text(pre_pack.get("hand") or []),
        "available_cards_json": _available_cards_json(pre_shop, pre_pack),
        "action": record.get("action", ""),
        "action_family": record.get("action_family", ""),
        "action_slot": record.get("action_slot", ""),
        "action_label": record.get("action_label", ""),
        "n_legal": record.get("n_legal", ""),
        "legal_actions": _json(record.get("legal_actions", [])),
        "terminal": record.get("terminal", False),
        "won": "" if record.get("won") is None else record.get("won"),
    }
    row.update(_choice_fields(record))
    return row


def convert_trace(
    input_path: Path,
    output_path: Path,
    *,
    phases: frozenset[str] | None = DECISION_PHASES,
) -> tuple[int, int]:
    """Convert ``input_path`` and return ``(written_rows, skipped_rows)``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_path}") from exc
            if "pre_state" not in record:
                raise ValueError(f"Missing pre_state on line {line_number} of {input_path}")
            phase = (record["pre_state"] or {}).get("phase")
            if phases is not None and phase not in phases:
                skipped += 1
                continue
            writer.writerow(_row(record))
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="include any phases beyond blind_select, shop, and pack_opening",
    )
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_shop.csv")
    phases = None if args.all_phases else DECISION_PHASES
    written, skipped = convert_trace(args.input, output, phases=phases)
    print(f"wrote {written:,} rows to {output} (skipped {skipped:,} other-phase rows)")


if __name__ == "__main__":
    main()
