"""Search-based playing agent: expectimax over the cards still in the shoe.

At each decision the agent:
  1. counts the cards it hasn't seen (undealt shoe + dealer's hole card),
  2. runs the exact dealer DFS (search.dealer_outcomes_from) on that
     composition to get the dealer's final-total distribution,
  3. searches the player's game tree - chance nodes for each card drawn,
     max nodes for each choice of STAND / HIT / DOUBLE / SPLIT - and plays
     the action with the highest expected value.

Unlike a fixed strategy table, the decision depends on the actual remaining
cards, so the agent adapts to the count automatically.

Approximation (standard in blackjack analysis, and it keeps a decision to a
few milliseconds): while evaluating one decision, the draw probabilities and
the dealer distribution are held fixed at the current composition rather than
updated after every hypothetical player card. Splits are evaluated as two
independent hands with no re-splitting, matching Rules.max_hands = 2.
"""
from search.search import _VALUES, composition, dealer_outcomes_from
from simulation.blackjack import hand_value, is_soft

ACTIONS = ("STAND", "HIT", "DOUBLE", "SPLIT")


class ExpectimaxAgent:
    def __init__(self, hit_soft_17=True, double_after_split=True):
        self.shoe = None
        self.hit_soft_17 = hit_soft_17
        self.das = double_after_split
        self._cache_key = None
        self._cache = None

    def attach(self, shoe, rules=None):
        """Called by run_session so the agent can see which cards are left."""
        self.shoe = shoe
        if rules is not None:
            self.hit_soft_17 = rules.hit_soft_17
            self.das = rules.double_after_split

    # ------------------------------------------------------------------ API
    def __call__(self, cards, upcard, can_double=False, can_split=False):
        evs = self.action_evs(cards, upcard, can_double, can_split)
        return max(evs, key=evs.get)

    def action_evs(self, cards, upcard, can_double=False, can_split=False, comp=None):
        """Expected profit (in units of this hand's bet) of each legal action.

        comp -- optional composition tuple to search over; defaults to the
                cards the player hasn't seen in the attached shoe.
        """
        if comp is None:
            if self.shoe is None:
                raise RuntimeError("attach(shoe) the agent before playing")
            comp = composition(self.shoe.unseen_cards)
        model = self._model(comp, upcard.value)
        hard = sum(1 if c.rank == "A" else c.value for c in cards)
        ace = any(c.rank == "A" for c in cards)
        evs = {"STAND": model.stand(hand_value(cards)),
               "HIT": model.hit(hard, ace)}
        if can_double and len(cards) == 2:
            evs["DOUBLE"] = model.double(hard, ace)
        if can_split and len(cards) == 2 and cards[0].value == cards[1].value:
            evs["SPLIT"] = model.split(cards[0].value, self.das)
        return evs

    # ------------------------------------------------------------ internals
    def _model(self, comp, up):
        key = (comp, up, self.hit_soft_17)
        if key != self._cache_key:
            self._cache_key, self._cache = key, _HandModel(comp, up, self.hit_soft_17)
        return self._cache


class _HandModel:
    """Player EVs for one (composition, upcard), memoised over player states."""

    def __init__(self, comp, up, hit_soft_17):
        total = sum(comp)
        self.p = [(1 if v == 11 else v, v == 11, n / total)
                  for v, n in zip(_VALUES, comp) if n]
        peeked = up in (10, 11)       # acting at all means no dealer blackjack
        self.dealer = dealer_outcomes_from(comp, up, hit_soft_17, peeked)
        self._best = {}

    @staticmethod
    def _total(hard, ace):
        return hard + 10 if ace and hard + 10 <= 21 else hard

    def stand(self, total):
        if total > 21:
            return -1.0
        d = self.dealer
        ev = d["bust"] - d["blackjack"]
        for k in (17, 18, 19, 20, 21):
            ev += d[k] * ((total > k) - (total < k))
        return ev

    def best(self, hard, ace):
        """Value of a hand when the player may only stand or hit from here."""
        key = (hard, ace)
        if key not in self._best:
            t = self._total(hard, ace)
            self._best[key] = self.stand(t) if t >= 21 else max(self.stand(t), self.hit(hard, ace))
        return self._best[key]

    def hit(self, hard, ace):
        ev = 0.0
        for v, is_ace, p in self.p:
            h = hard + v
            ev += p * (-1.0 if h > 21 else self.best(h, ace or is_ace))
        return ev

    def double(self, hard, ace):
        ev = 0.0
        for v, is_ace, p in self.p:
            h = hard + v
            ev += p * (-1.0 if h > 21 else self.stand(self._total(h, ace or is_ace)))
        return 2 * ev

    def split(self, pair_value, das):
        """Two hands, each starting from one pair card plus a drawn card."""
        aces = pair_value == 11
        start = 1 if aces else pair_value
        ev = 0.0
        for v, is_ace, p in self.p:
            hard, ace = start + v, aces or is_ace
            if aces:                    # split aces: one card, then stand
                val = self.stand(self._total(hard, ace))
            else:
                val = self.best(hard, ace)
                if das:
                    val = max(val, self.double(hard, ace))
            ev += p * val
        return 2 * ev
