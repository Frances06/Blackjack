"""Finite-shoe cards and Hi-Lo card counting."""
from dataclasses import dataclass
import random

RANKS = ("2","3","4","5","6","7","8","9","10","J","Q","K","A")
HI_LO = {**{r: 1 for r in ("2","3","4","5","6")},
         **{r: 0 for r in ("7","8","9")},
         **{r: -1 for r in ("10","J","Q","K","A")}}

@dataclass(frozen=True)
class Card:
    rank: str
    @property
    def value(self):
        if self.rank in ("J","Q","K"): return 10
        if self.rank == "A": return 11
        return int(self.rank)

class Shoe:
    def __init__(self, decks=6, seed=None):
        if decks < 1: raise ValueError("decks must be at least 1")
        self.decks = decks
        self.rng = random.Random(seed)
        self.reshuffle()
    def reshuffle(self):
        self.cards = [Card(r) for _ in range(self.decks) for r in RANKS for _ in range(4)]
        self.rng.shuffle(self.cards)
        self.hidden = []            # dealt face-down, not yet seen by the player
        self.running_count = 0
    @property
    def cards_remaining(self): return len(self.cards)
    @property
    def unseen_cards(self):
        """Cards the player hasn't seen: the undealt shoe plus any face-down cards.
        Agents must use this, not `cards`, or they'd effectively know the hole card."""
        return self.cards + self.hidden
    @property
    def decks_remaining(self): return max(len(self.cards)/52, 0.25)
    @property
    def true_count(self): return self.running_count / self.decks_remaining
    def draw(self, exposed=True):
        """Draw a card; hidden cards are not counted until revealed."""
        if not self.cards: raise RuntimeError("Shoe empty; reshuffle before drawing")
        card = self.cards.pop()
        if exposed:
            self.running_count += HI_LO[card.rank]
        else:
            self.hidden.append(card)
        return card

    def reveal(self, card):
        """Add a previously hidden card to the count when it becomes visible."""
        self.running_count += HI_LO[card.rank]
        if card in self.hidden:
            self.hidden.remove(card)

def hand_value(cards):
    total = sum(c.value for c in cards)
    aces = sum(c.rank == "A" for c in cards)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_soft(cards):
    """True if an ace is currently being counted as 11."""
    hard = sum(1 if c.rank == "A" else c.value for c in cards)
    return any(c.rank == "A" for c in cards) and hard + 10 <= 21


def is_blackjack(cards):
    """A natural: exactly two cards totalling 21."""
    return len(cards) == 2 and hand_value(cards) == 21


def dealer_should_hit(cards, hit_soft_17=True):
    """Dealer draws below 17, and on soft 17 under H17 rules."""
    total = hand_value(cards)
    return total < 17 or (total == 17 and hit_soft_17 and is_soft(cards))
