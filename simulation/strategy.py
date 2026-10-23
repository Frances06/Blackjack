"""Betting and playing rules.

choose_bet    -- Hi-Lo bet ramp (unchanged).
choose_action -- now plays from the player-EV dataset (player-ev.json):
                 for each hand it picks the highest-EV legal action.
choose_action_baseline -- the original simple hit/stand rule, kept for comparison.

player-ev.json is looked for in, in order: $BLACKJACK_EV_PATH, simulation/,
machine_learning/data/. Call configure() to point at a different path or change
the deck count / dealer rule (defaults: 6 decks, H17). The dealer rule here must
match the one the game loop plays (run_simulation.Rules.hit_soft_17).
"""
import json
import os
from dataclasses import dataclass

from simulation.blackjack import hand_value, is_soft

# ---------------------------------------------------------------------------
# Betting (unchanged)
# ---------------------------------------------------------------------------
def choose_bet(true_count, bankroll, minimum=10, maximum=100):
    if bankroll <= 0: return 0
    if true_count >= 4: proposed = 100
    elif true_count >= 3: proposed = 60
    elif true_count >= 2: proposed = 40
    elif true_count >= 1: proposed = 20
    else: proposed = minimum
    return min(maximum, bankroll, max(minimum, proposed))


# ---------------------------------------------------------------------------
# Original playing rule, kept as a baseline
# ---------------------------------------------------------------------------
def choose_action_baseline(player_cards, dealer_upcard, can_double=False, can_split=False):
    """Simplified hit/stand baseline; not full basic strategy."""
    total = hand_value(player_cards)
    if total <= 11: return "HIT"
    if total >= 17: return "STAND"
    if total >= 13 and dealer_upcard.value <= 6: return "STAND"
    return "HIT"


# ---------------------------------------------------------------------------
# Dataset-driven playing rule
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = (
    os.environ.get("BLACKJACK_EV_PATH"),
    os.path.join(_HERE, "player-ev.json"),
    os.path.join(_HERE, os.pardir, "machine_learning", "data", "player-ev.json"),
)


def find_dataset():
    """First existing player-ev.json among the candidate locations, else None."""
    for p in _CANDIDATES:
        if p and os.path.isfile(p):
            return os.path.normpath(p)
    return None


DATASET_PATH = find_dataset()

# dataset column per action; return values match the existing "HIT"/"STAND" style
_COLUMNS = {"STAND": "stand_ev", "HIT": "hit_ev", "DOUBLE": "double_ev", "SPLIT": "split_ev"}
_LETTERS = {"S": "STAND", "H": "HIT", "D": "DOUBLE", "P": "SPLIT"}


@dataclass(frozen=True)
class Entry:
    evs: dict            # action -> EV in cents per dollar of the original bet
    best_action: str     # the dataset's own pick (used to break exact ties)


class StrategyTable:
    """player-ev.json indexed by (decks, dealer_rule, category, hand, dealer_upcard)."""

    def __init__(self, rows):
        self._rows = {}
        for r in rows:
            evs = {a: float(r[c]) for a, c in _COLUMNS.items() if r.get(c) not in ("", None)}
            key = (int(r["decks"]), r["dealer_rule"], r["category"], r["hand"], r["dealer_upcard"])
            self._rows[key] = Entry(evs, _LETTERS[r["best_action"]])
        self.deck_counts = sorted({k[0] for k in self._rows})
        self.rules = sorted({k[1] for k in self._rows})

    @classmethod
    def load(cls, path=None):
        path = path or find_dataset()
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                "player-ev.json not found. Put it in machine_learning/data/ (or simulation/), "
                "set BLACKJACK_EV_PATH, or run with --policy baseline.")
        with open(path) as f:
            return cls(json.load(f)["rows"])

    def get(self, *key):
        return self._rows.get(key)


_settings = {"path": DATASET_PATH, "decks": 6, "dealer_rule": "H17"}
_table = None


def configure(decks=None, dealer_rule=None, path=None):
    """Set the game the dataset lookups should assume. Call once at startup."""
    global _table
    if path is not None and path != _settings["path"]:
        _settings["path"] = path
        _table = None
    if path is None and _table is None:
        _settings["path"] = find_dataset()
    if decks is not None:
        _settings["decks"] = decks
    if dealer_rule is not None:
        _settings["dealer_rule"] = dealer_rule
    _get_table()  # load now so a bad path or rule fails at startup, not mid-game


def _get_table():
    global _table
    if _table is None:
        _table = StrategyTable.load(_settings["path"])
    if _settings["dealer_rule"] not in _table.rules:
        raise ValueError(f"dealer_rule must be one of {_table.rules}")
    return _table


def _classify(cards, can_split):
    """Map a hand onto the dataset's (category, hand) labels.
    J/Q/K count as 10 (K,Q is the pair "10,10"); multi-card soft hands use
    their total, so A,2,4 is looked up as "A,6"."""
    total = hand_value(cards)
    soft = is_soft(cards)
    if can_split and len(cards) == 2 and cards[0].value == cards[1].value:
        v = cards[0].value
        return "pair", ("A,A" if v == 11 else f"{v},{v}"), total, soft
    if soft:
        return "soft", f"A,{total - 11}", total, True
    return "hard", str(total), total, False


def action_evs(player_cards, dealer_upcard, can_double=False, can_split=False):
    """Dataset EV (cents per dollar) of each legal action; {} if the hand isn't in the table."""
    table = _get_table()
    decks = min(table.deck_counts, key=lambda d: (abs(d - _settings["decks"]), d))
    cat, hand, _, _ = _classify(player_cards, can_split)
    up = "A" if dealer_upcard.value == 11 else str(dealer_upcard.value)
    entry = table.get(decks, _settings["dealer_rule"], cat, hand, up)
    if entry is None:
        return {}, None
    legal = {"STAND", "HIT"}
    if can_double and len(player_cards) == 2:
        legal.add("DOUBLE")
    if cat == "pair":
        legal.add("SPLIT")
    return {a: ev for a, ev in entry.evs.items() if a in legal}, entry


def choose_action(player_cards, dealer_upcard, can_double=False, can_split=False):
    """Best legal action from the dataset: "HIT", "STAND", "DOUBLE" or "SPLIT".

    can_double / can_split default to False, so existing callers that only
    handle HIT and STAND keep working; pass True once your game loop supports them.
    """
    cat, _, total, soft = _classify(player_cards, can_split)

    # Hands the dataset doesn't list (splittable pairs are always listed).
    if cat != "pair":
        if total >= 21 or (not soft and total >= 20):
            return "STAND"
        if total <= 4 or (soft and total <= 12):
            return "HIT"                    # 2,2 or A,A that can't be split

    evs, entry = action_evs(player_cards, dealer_upcard, can_double, can_split)
    if not evs:
        return choose_action_baseline(player_cards, dealer_upcard)
    best = max(evs.values())
    if evs.get(entry.best_action) == best:
        return entry.best_action
    return max(evs, key=evs.get)