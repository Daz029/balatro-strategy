"""Scenario-driven differential oracle for whole Balatro transitions.

The Lua half is in :mod:`scripts.lua_transition_oracle` and executes the
unmodified 1.0.1o gameplay sources in-process with lupa.  This package owns
the shared JSON schema and the single Lua/Python snapshot mapping.
"""

from tests.oracle.transition import (
    Action,
    BlindSpec,
    CardSpec,
    Diff,
    Observation,
    OracleTrace,
    Scenario,
    build_engine_state,
    compare,
    run_engine,
    run_lua_oracle,
)

__all__ = [
    "Action",
    "BlindSpec",
    "CardSpec",
    "Diff",
    "Observation",
    "OracleTrace",
    "Scenario",
    "build_engine_state",
    "compare",
    "run_engine",
    "run_lua_oracle",
]
