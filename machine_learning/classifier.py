"""Supervised learning agent: a classifier trained on player-ev.json.

Each dataset row is one decision (player hand vs dealer upcard, for a given
deck count and dealer rule) labelled with the best action. We turn it into
numeric features and train a classifier to predict that label.

Features (no EV columns, to avoid target leakage):
    player_total, is_soft, pair_value (0 = not a pair), dealer_upcard (2-11),
    decks, h17 (1 = dealer hits soft 17)

Because the features are totals rather than exact cards, the trained model
also gives an answer for hands the table doesn't list (e.g. 3-card hands).
A second, smaller tree learns STAND-vs-HIT (label = whichever of stand_ev and
hit_ev is higher); it's used when the main tree says DOUBLE but doubling isn't
allowed (e.g. after a hit).

Run `python3 -m machine_learning.train_classifier` to compare models.
"""
import json

from simulation.blackjack import hand_value, is_soft
from simulation.strategy import find_dataset, choose_action_baseline

ACTIONS = {"S": "STAND", "H": "HIT", "D": "DOUBLE", "P": "SPLIT"}
EV_COLUMNS = {"S": "stand_ev", "H": "hit_ev", "D": "double_ev", "P": "split_ev"}
FEATURES = ["player_total", "is_soft", "pair_value", "dealer_upcard", "decks", "h17"]

# Chosen with train_classifier.py (cross-validated depth); see results/.
TREE_PARAMS = {"max_depth": 12, "min_samples_leaf": 1, "random_state": 0}


def _upcard_value(label):
    return 11 if label == "A" else int(label)


def row_features(r):
    """Feature vector for one dataset row."""
    hand, cat = r["hand"], r["category"]
    if cat == "pair":
        v = 11 if hand.startswith("A") else int(hand.split(",")[0])
        total, soft, pair = (12, 1, 11) if v == 11 else (2 * v, 0, v)
    elif cat == "soft":
        total, soft, pair = 11 + int(hand.split(",")[1]), 1, 0
    else:
        total, soft, pair = int(hand), 0, 0
    return [total, soft, pair, _upcard_value(r["dealer_upcard"]),
            int(r["decks"]), int(r["dealer_rule"] == "H17")]


def hand_features(cards, upcard, can_split, decks, h17):
    """Feature vector for a live hand in the simulator."""
    pair = cards[0].value if (can_split and len(cards) == 2
                              and cards[0].value == cards[1].value) else 0
    return [hand_value(cards), int(is_soft(cards)), pair, upcard.value, decks, int(h17)]


def load_rows(path=None):
    path = path or find_dataset()
    if path is None:
        raise FileNotFoundError("player-ev.json not found; put it in machine_learning/data/")
    with open(path) as f:
        return json.load(f)["rows"]


def dataset_xy(rows):
    X = [row_features(r) for r in rows]
    y = [r["best_action"] for r in rows]
    return X, y


def stand_hit_xy(rows):
    """Non-pair rows labelled with the better of STAND / HIT."""
    rows = [r for r in rows if r["category"] != "pair"]
    X = [row_features(r) for r in rows]
    y = ["S" if float(r["stand_ev"]) >= float(r["hit_ev"]) else "H" for r in rows]
    return X, y


def ev_regret(rows, predicted):
    """Mean EV lost (cents per dollar) by playing `predicted` instead of the best action."""
    total = 0.0
    for r, p in zip(rows, predicted):
        evs = {a: float(r[c]) for a, c in EV_COLUMNS.items() if r.get(c) not in ("", None)}
        total += max(evs.values()) - evs.get(p, min(evs.values()))
    return total / len(rows)


class TreeAgent:
    """Decision tree trained on the whole dataset, used as a playing policy."""

    def __init__(self, path=None, params=None):
        from sklearn.tree import DecisionTreeClassifier
        rows = load_rows(path)
        params = params or TREE_PARAMS
        self.model = DecisionTreeClassifier(**params).fit(*dataset_xy(rows))
        self.stand_hit = DecisionTreeClassifier(**params).fit(*stand_hit_xy(rows))
        self.decks, self.h17 = 6, True

    def attach(self, shoe, rules=None):
        self.decks = shoe.decks
        if rules is not None:
            self.h17 = rules.hit_soft_17

    def __call__(self, cards, upcard, can_double=False, can_split=False):
        total = hand_value(cards)
        if total >= 21:
            return "STAND"
        x = [hand_features(cards, upcard, can_split, self.decks, self.h17)]
        action = ACTIONS[self.model.predict(x)[0]]
        if action == "DOUBLE" and not (can_double and len(cards) == 2):
            x[0][2] = 0
            return ACTIONS[self.stand_hit.predict(x)[0]]
        if action == "SPLIT" and not can_split:      # shouldn't happen: pair_value is 0
            return choose_action_baseline(cards, upcard)
        return action
