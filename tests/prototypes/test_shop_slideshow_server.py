import importlib.util
from pathlib import Path


SERVER_PATH = Path(__file__).parents[2] / "prototypes" / "shop-slideshow-prototype" / "server.py"
SPEC = importlib.util.spec_from_file_location("shop_slideshow_server", SERVER_PATH)
assert SPEC and SPEC.loader
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def test_playing_cards_use_their_rank_and_suit_instead_of_default_base() -> None:
    card = {
        "name": "Default Base",
        "center_key": "c_base",
        "set": "Default",
        "base": {"rank": "King", "suit": "Hearts"},
    }

    assert SERVER._card(card)["name"] == "King of Hearts"


def test_non_playing_cards_keep_their_serialized_name() -> None:
    card = {
        "name": "The Fool",
        "center_key": "c_fool",
        "set": "Tarot",
        "base": None,
    }

    assert SERVER._card(card)["name"] == "The Fool"
