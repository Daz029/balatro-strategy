"""Serve a Balatro-style slideshow for dumped hand-evaluation decisions.

Run from the repository root with::

    python prototypes/hand-slideshow/server.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parents[1]
DEFAULT_TRACE = REPO_ROOT / "runs" / "hand_ppo_b" / "h2" / "dumped_hand_eval.jsonl"
MAX_VISIBLE_HAND = 12
MAX_VISIBLE_JOKERS = 8

RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 11,
    "Queen": 12,
    "King": 13,
    "Ace": 14,
}
SUIT_VALUES = {"Diamonds": 1, "Clubs": 2, "Hearts": 3, "Spades": 4}


def _edition_name(edition: Any) -> str | None:
    if isinstance(edition, str):
        return edition
    if isinstance(edition, dict):
        return next((str(key) for key, enabled in edition.items() if enabled), None)
    return None


def _playing_card(card: dict[str, Any], slot: int, pick_order: int | None) -> dict[str, Any]:
    base = card.get("base") or {}
    ability = card.get("ability") or {}
    center_key = card.get("center_key")
    enhancement = None
    if center_key and center_key != "c_base":
        enhancement = card.get("name") or ability.get("name") or center_key
    return {
        "slot": slot,
        "rank": base.get("rank") or "?",
        "rank_value": int(base.get("id") or RANK_VALUES.get(str(base.get("rank")), 0)),
        "suit": base.get("suit") or "Unknown",
        "suit_value": SUIT_VALUES.get(str(base.get("suit")), 0),
        "enhancement": enhancement,
        "edition": _edition_name(card.get("edition")),
        "seal": card.get("seal"),
        "debuff": bool(card.get("debuff", False)),
        "picked": pick_order is not None,
        "pick_order": pick_order,
    }


def _sort_card(card: dict[str, Any]) -> tuple[int, int, int]:
    return (card["rank_value"], card["suit_value"], -card["slot"])


def _visible_hand(
    cards: list[dict[str, Any]], selected_indices: list[int]
) -> tuple[list[dict[str, Any]], int]:
    """Sort the hand and cap it at 12 without hiding a model-selected card."""
    order_by_slot = {slot: order for order, slot in enumerate(selected_indices, start=1)}
    sorted_cards = sorted(
        (_playing_card(card, slot, order_by_slot.get(slot)) for slot, card in enumerate(cards)),
        key=_sort_card,
        reverse=True,
    )
    visible = sorted_cards[:MAX_VISIBLE_HAND]
    visible_slots = {card["slot"] for card in visible}
    hidden_picks = [
        card for card in sorted_cards[MAX_VISIBLE_HAND:] if card["slot"] in order_by_slot
    ]
    for picked in hidden_picks:
        replace_at = next(
            (index for index in range(len(visible) - 1, -1, -1) if not visible[index]["picked"]),
            None,
        )
        if replace_at is None:
            break
        visible_slots.discard(visible[replace_at]["slot"])
        visible[replace_at] = picked
        visible_slots.add(picked["slot"])
    visible.sort(key=_sort_card, reverse=True)
    return visible, max(0, len(sorted_cards) - len(visible))


def _joker(card: dict[str, Any]) -> dict[str, Any]:
    ability = card.get("ability") or {}
    detail = ability.get("effect") or ability.get("type")
    return {
        "name": card.get("name") or ability.get("name") or card.get("center_key") or "Unknown",
        "key": card.get("center_key"),
        "detail": detail or "Joker",
        "edition": _edition_name(card.get("edition")),
        "eternal": bool(card.get("eternal", False)),
        "perishable": bool(card.get("perishable", False)),
        "rental": bool(card.get("rental", False)),
        "debuff": bool(card.get("debuff", False)),
    }


def frame_from_record(record: dict[str, Any]) -> dict[str, Any]:
    selected_indices = [int(index) for index in record.get("selected_indices") or []]
    hand, hidden_count = _visible_hand(record.get("cards_in_hand") or [], selected_indices)
    blind = record.get("blind") or {}
    hand_score = record.get("hand_point_value")
    score = record.get("score") or {}
    if hand_score is None:
        hand_score = score.get("total")
    return {
        "seed": str(record.get("seed") or "UNKNOWN"),
        "decision": int(record.get("hand_decision_index") or 0),
        "action_type": str(record.get("action_type") or "Unknown"),
        "selected_indices": selected_indices,
        "hand": hand,
        "hand_count": len(record.get("cards_in_hand") or []),
        "hidden_hand_count": hidden_count,
        "jokers": [_joker(card) for card in (record.get("jokers") or [])[:MAX_VISIBLE_JOKERS]],
        "money": record.get("money", 0),
        "ante": record.get("ante", 0),
        "round": record.get("round", 0),
        "required_score": record.get("blind_points", blind.get("chips", 0)),
        "current_score": record.get("points", 0),
        "hand_score": hand_score,
        "hand_type": record.get("hand_type") or score.get("hand_type"),
        "hand_chips": record.get("hand_chips", score.get("chips")),
        "hand_mult": record.get("hand_mult", score.get("mult")),
        "hands_left": record.get("hands_left", 0),
        "discards_left": record.get("discards_left", 0),
        "blind": {
            "name": blind.get("name") or "Blind",
            "boss": bool(blind.get("boss", False)),
        },
    }


def load_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame = frame_from_record(record)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                message = f"Invalid hand trace record on line {line_number}: {error}"
                raise ValueError(message) from error
            runs[frame["seed"]].append(frame)
    return dict(runs)


class HandSlideshowHandler(SimpleHTTPRequestHandler):
    runs: dict[str, list[dict[str, Any]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self._send_json(
                [
                    {
                        "seed": seed,
                        "frames": len(frames),
                        "first_decision": frames[0]["decision"],
                        "last_decision": frames[-1]["decision"],
                    }
                    for seed, frames in self.runs.items()
                ]
            )
            return
        if parsed.path == "/api/frames":
            seed = parse_qs(parsed.query).get("seed", [""])[0]
            frames = self.runs.get(seed)
            if frames is None:
                self._send_json({"error": f"Unknown seed: {seed}"}, status=404)
            else:
                self._send_json({"seed": seed, "frames": frames})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if not args.trace.is_file():
        parser.error(f"Trace file not found: {args.trace}")
    print(f"Loading hand decisions from {args.trace} ...")
    HandSlideshowHandler.runs = load_trace(args.trace)
    frame_count = sum(len(frames) for frames in HandSlideshowHandler.runs.values())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), HandSlideshowHandler)
    print(
        f"Loaded {frame_count:,} decisions across {len(HandSlideshowHandler.runs):,} runs. "
        f"Open http://127.0.0.1:{args.port}/"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
