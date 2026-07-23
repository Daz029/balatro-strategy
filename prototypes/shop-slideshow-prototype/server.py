"""PROTOTYPE: serve a Balatro-like slideshow for rich shop evaluation traces.

Run from anywhere with::

    python prototypes/shop-slideshow-prototype/server.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROTOTYPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROTOTYPE_DIR.parents[1]
DEFAULT_TRACE = REPO_ROOT / "data" / "dumped_eval_rich.jsonl"
VISIBLE_PHASES = {"blind_select", "shop", "pack_opening"}


def _upcoming_blind(state: dict[str, Any]) -> dict[str, Any]:
    """Build the exact upcoming blind from the decision-visible state."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from jackdaw.engine.blind import Blind

    round_resets = state.get("round_resets") or {}
    blind_type = str(state.get("blind_on_deck") or "Small")
    blind_key = (round_resets.get("blind_choices") or {}).get(blind_type, "bl_small")
    run = state.get("run") or {}
    stake = int(run.get("stake", 1))
    scaling = 3 if stake >= 6 else 2 if stake >= 3 else 1
    ante_scaling = 2.0 if run.get("back_key") == "b_plasma" else 1.0
    blind = Blind.create(
        blind_key,
        int(state.get("ante", 1)),
        scaling=scaling,
        ante_scaling=ante_scaling,
    )
    return {
        "type": blind_type,
        "key": blind.key,
        "name": blind.name,
        "chips": blind.chips,
        "can_skip": blind_type != "Boss",
        "skip_tag": (round_resets.get("blind_tags") or {}).get(blind_type),
    }


def _card(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not card:
        return None
    base = card.get("base") or {}
    ability = card.get("ability") or {}
    edition = card.get("edition") or {}
    detail_parts = [ability.get("effect"), ability.get("type")]
    rank = base.get("rank")
    suit = base.get("suit")
    name = f"{rank} of {suit}" if rank and suit else None
    return {
        "name": name or card.get("name") or ability.get("name") or card.get("center_key") or "Unknown",
        "key": card.get("center_key") or card.get("card_key"),
        "set": card.get("set") or ability.get("set") or "Card",
        "cost": card.get("cost"),
        "sell_cost": card.get("sell_cost"),
        "edition": edition.get("type") if isinstance(edition, dict) else edition,
        "eternal": bool(card.get("eternal", False)),
        "rental": bool(card.get("rental", False)),
        "perishable": bool(card.get("perishable", False)),
        "debuff": bool(card.get("debuff", False)),
        "detail": " · ".join(str(part) for part in detail_parts if part),
    }


def _cards(cards: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [view for item in cards or [] if (view := _card(item)) is not None]


def _target(target: dict[str, Any] | None) -> dict[str, Any] | None:
    if not target:
        return None
    return {
        "kind": target.get("kind"),
        "slot": target.get("slot"),
        "slots": target.get("slots") or [],
        "card": _card(target.get("card")),
        "cards": _cards(target.get("cards")),
    }


def _result_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize the run outcome recorded on a terminal shop decision."""
    won = bool(record.get("won"))
    pre = record.get("pre_state") or {}
    post = record.get("post_state") or {}
    complete = won or bool(post.get("done"))

    # A win keeps the cleared blind in the pre-state. A loss keeps the failed
    # blind and final score in the post-state.
    state = pre if won else post if complete else None
    blind = (state or {}).get("blind") or {}
    final_score = (state or {}).get("chips")
    required_score = blind.get("chips")
    margin = None
    if final_score is not None and required_score is not None:
        margin = final_score - required_score

    return {
        "status": "won" if won else "lost" if complete else "incomplete",
        "final_score": final_score,
        "required_score": required_score,
        "margin": margin,
    }


def _frame(record: dict[str, Any]) -> dict[str, Any]:
    pre = record["pre_state"]
    post = record.get("post_state") or {}
    inventory = pre.get("inventory") or {}
    post_inventory = post.get("inventory") or {}
    shop = pre.get("shop") or {}
    pack = pre.get("pack") or {}
    resources = pre.get("resources") or {}
    current_round = pre.get("current_round") or {}
    dollars = pre.get("dollars", record.get("dollars", 0))
    dollars_after = post.get("dollars", dollars)
    frame = {
        "seed": record.get("seed"),
        "step": record.get("step"),
        "phase": pre.get("phase"),
        "ante": pre.get("ante"),
        "round": pre.get("round"),
        "blind_on_deck": pre.get("blind_on_deck"),
        "upcoming_blind": _upcoming_blind(pre),
        "dollars": dollars,
        "dollars_after": dollars_after,
        "money_delta": dollars_after - dollars,
        "reroll_cost": current_round.get("reroll_cost", 5),
        "joker_slots": min(8, int(resources.get("joker_slots", 5))),
        "consumable_slots": int(resources.get("consumable_slots", 2)),
        "jokers": _cards(inventory.get("jokers")),
        "jokers_after": _cards(post_inventory.get("jokers")),
        "consumables": _cards(inventory.get("consumables")),
        "consumables_after": _cards(post_inventory.get("consumables")),
        "shop_cards": _cards(shop.get("cards")),
        "vouchers": _cards(shop.get("vouchers")),
        "boosters": _cards(shop.get("boosters")),
        "pack_cards": _cards(pack.get("cards")),
        "pack_hand": _cards(pack.get("hand")),
        "pack_type": resources.get("pack_type"),
        "pack_choices_remaining": resources.get("pack_choices_remaining"),
        "action": record.get("action_family"),
        "action_label": record.get("action_label"),
        "action_slot": record.get("action_slot"),
        "target": _target(record.get("action_target")),
        "terminal": bool(record.get("terminal", False)),
        "won": record.get("won"),
    }
    if record.get("terminal"):
        frame["result"] = _result_from_record(record)
    return frame


def _run_result(frames: list[dict[str, Any]]) -> dict[str, Any]:
    return frames[-1].get("result") or {"status": "lost", "margin": None}


def load_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if (record.get("pre_state") or {}).get("phase") in VISIBLE_PHASES:
                runs[str(record["seed"])].append(_frame(record))
    return dict(runs)


class SlideshowHandler(SimpleHTTPRequestHandler):
    runs: dict[str, list[dict[str, Any]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PROTOTYPE_DIR), **kwargs)

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
                        "first_step": frames[0]["step"],
                        "last_step": frames[-1]["step"],
                        "result": _run_result(frames),
                    }
                    for seed, frames in self.runs.items()
                ]
            )
            return
        if parsed.path == "/api/frames":
            seed = parse_qs(parsed.query).get("seed", [""])[0]
            frames = self.runs.get(seed)
            if frames is None:
                self._send_json({"error": f"unknown seed: {seed}"}, status=404)
            else:
                self._send_json({"seed": seed, "frames": frames, "result": _run_result(frames)})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Loading shop frames from {args.trace} ...")
    SlideshowHandler.runs = load_trace(args.trace)
    frame_count = sum(map(len, SlideshowHandler.runs.values()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), SlideshowHandler)
    print(
        f"Loaded {frame_count:,} frames across {len(SlideshowHandler.runs):,} runs. "
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
