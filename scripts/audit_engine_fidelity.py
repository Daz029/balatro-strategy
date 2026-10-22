"""Reproduce the 2026-10-21 fidelity audit without changing engine/oracle code.

Run from the repository root: .venv/bin/python -m scripts.audit_engine_fidelity
Requires the local Balatro source and lupa.lua51; also uses native LuaJIT when
available. The extra harness calls the actual play callback and Card:update,
with UI alert work stubbed. It is a focused diagnostic, not a full Game:update.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from jackdaw.engine.actions import SellCard
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.observation import encode_joker
from tests.oracle import Action, BlindSpec, CardSpec, Scenario, run_engine, run_lua_oracle
from tests.oracle.transition import (
    _BOOTSTRAP,
    _lua_runtime,
    _source_path,
    _to_lua,
    build_engine_state,
)


def diagnostic_bootstrap() -> str:
    bootstrap = _BOOTSTRAP.read_text()
    start = bootstrap.index("    if name == 'play' then")
    end = bootstrap.index("    elseif name == 'discard' then", start)
    bootstrap = bootstrap[:start] + """    if name == 'play' then
        G.hand:unhighlight_all()
        for _, index in ipairs(action.indices or {}) do
            G.hand:add_to_highlighted(G.hand.cards[index + 1], true)
        end
        G.FUNCS.play_cards_from_highlighted()
    elseif name == 'disable_blind' then
        G.GAME.blind:disable()
""" + bootstrap[end:]
    # Run the real gameplay tally code, before action and after event updates.
    # Game:update processes the event manager before its moveable/Card updates.
    helper = """
function Card:update_alert() end -- UI/profile alerts only
local function audit_card_updates()
    G.STAGES = {RUN = 1}
    G.STAGE = G.STAGES.RUN
    for _, cards in ipairs({G.jokers.cards, G.consumeables.cards}) do
        for _, card in ipairs(cards) do
            card.states.focus = {is = false}
            card.sticker_run = 'NONE'
            card:update(0)
        end
    end
end

"""
    bootstrap = bootstrap.replace("local function drain()", helper + "local function drain()")
    bootstrap = bootstrap.replace(
        "        G.E_MANAGER:update(dt, true)",
        "        G.E_MANAGER:update(dt, true)\n        audit_card_updates()",
    )
    bootstrap = bootstrap.replace("    run_action(scenario)",
                                  "    audit_card_updates()\n    run_action(scenario)")
    return bootstrap


def lua_literal(value: object) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            f"[{lua_literal(k)}]={lua_literal(v)}" for k, v in value.items()
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "{" + ",".join(lua_literal(v) for v in value) + "}"
    return str(value)


def native_trace(executable: str, bootstrap: str, scenario: Scenario) -> dict:
    program = (
        f"BALATRO_SOURCE = {lua_literal(str(_source_path()))}\n"
        + bootstrap
        + f"\nprint(run_transition_oracle({lua_literal(scenario.to_dict())}))\n"
    )
    with tempfile.TemporaryDirectory(prefix="balatro-fidelity-") as directory:
        script = Path(directory) / "probe.lua"
        script.write_text(program)
        result = subprocess.run(
            [executable, str(script)], capture_output=True, text=True, check=True, timeout=30
        )
    return json.loads(result.stdout)["observations"][-1]


def main() -> None:
    source = _source_path()
    cases = {
        "vampire_steel": Scenario(
            areas={"hand": [CardSpec("p", front="H_2", center="m_steel")]},
            jokers=[CardSpec("v", key="j_vampire"), CardSpec("s", key="j_steel_joker")],
            action=Action.play(0),
        ),
        "dusk_last_hand": Scenario(
            hands_left=1,
            areas={"hand": [CardSpec("p", front="H_2")]},
            jokers=[CardSpec("j", key="j_dusk")],
            action=Action.play(0),
        ),
        "tooth_bull": Scenario(
            dollars=4,
            blind=BlindSpec("bl_tooth"),
            areas={"hand": [CardSpec("p", front="H_2")]},
            jokers=[CardSpec("j", key="j_bull")],
            action=Action.play(0),
        ),
        "disable_wall": Scenario(
            chips=700,
            blind=BlindSpec("bl_wall", {"chips": 1200}),
            jokers=[CardSpec("l", key="j_luchador")],
            action=Action("disable_blind"),  # diagnostic-only action
        ),
    }
    report = {
        "source_version": (source / "version.jkr").read_text().splitlines(),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("card.lua", "blind.lua", "game.lua", "engine/event.lua",
                         "functions/state_events.lua", "functions/button_callbacks.lua")
        },
        "cases": {},
    }
    gs = initialize_run("b_red", 1, "AUDIT")
    joker = create_joker("j_joker")
    face_up = encode_joker(joker, 0, gs).tolist()
    joker.facing = "back"
    face_down = encode_joker(joker, 0, gs).tolist()
    report["joker_visibility"] = {
        "face_up_equals_face_down": face_up == face_down,
        "face_down_identity_feature": face_down[0],
    }
    # Capture stock-oracle results before loading the diagnostic harness.
    for name, scenario in cases.items():
        entry = report["cases"][name] = {}
        if name == "disable_wall":
            gs = build_engine_state(scenario)
            step(gs, SellCard("jokers", 0))
            entry["engine"] = {"phase": str(gs["phase"]), "chips": gs["chips"],
                               "target": gs["blind"].chips}
        else:
            entry["engine_chips"] = run_engine(scenario).final["chips"]
            try:
                entry["stock_oracle_chips"] = run_lua_oracle(scenario).final["chips"]
            except Exception as error:
                # The missing steel_tally is itself an oracle finding. Preserve
                # errors as evidence, rather than silently skipping a scenario.
                entry["stock_oracle_error"] = str(error).splitlines()[0]

    bootstrap = diagnostic_bootstrap()
    runtime = _lua_runtime(str(source))
    runtime.execute(bootstrap)
    native = shutil.which("luajit")
    if native is None and Path("/opt/homebrew/bin/luajit").is_file():
        native = "/opt/homebrew/bin/luajit"
    try:
        for name, scenario in cases.items():
            entry = report["cases"][name]
            trace = json.loads(runtime.globals().run_transition_oracle(_to_lua(runtime, scenario)))
            final = trace["observations"][-1]
            entry["diagnostic_lua51"] = {"chips": final["chips"], "phase": final["phase"]}
            if native:
                final = native_trace(native, bootstrap, scenario)
                entry["diagnostic_luajit"] = {"chips": final["chips"], "phase": final["phase"]}
            else:
                entry["diagnostic_luajit"] = "unavailable"
    finally:
        runtime.execute(_BOOTSTRAP.read_text())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
