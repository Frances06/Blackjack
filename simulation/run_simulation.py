"""Play blackjack hands and run repeatable sessions.

Rules modelled (see Rules): N-deck shoe, dealer hits or stands on soft 17,
blackjack pays 3:2, dealer peeks for blackjack with a 10 or ace up, double on
any two cards, one split (split aces get one card each), double after split.
No insurance or surrender.
"""
import argparse
from dataclasses import dataclass
from statistics import mean, stdev

from simulation import strategy
from simulation.blackjack import (Shoe, hand_value, is_blackjack,
                                  dealer_should_hit)
from simulation.strategy import choose_bet, choose_action, choose_action_baseline

def _tree_agent():
    from machine_learning.classifier import TreeAgent
    return TreeAgent()


def _mc_agent():
    from machine_learning.monte_carlo import MonteCarloAgent
    return MonteCarloAgent.load()


def _search_agent():
    from search.agent import ExpectimaxAgent
    return ExpectimaxAgent()


# name -> zero-argument factory (lazy, so e.g. scikit-learn only loads if needed)
POLICIES = {
    "baseline": lambda: choose_action_baseline,
    "dataset": lambda: choose_action,
    "tree": _tree_agent,
    "mc": _mc_agent,
    "search": _search_agent,
}


def get_policy(name):
    return POLICIES[name]()


@dataclass(frozen=True)
class Rules:
    decks: int = 6
    hit_soft_17: bool = True        # H17; strategy is configured to match
    blackjack_payout: float = 1.5
    double_after_split: bool = True
    max_hands: int = 2              # 2 = one split allowed
    reshuffle_below: int = 78       # cards left when the shoe is reshuffled

    @property
    def dealer_rule(self):
        return "H17" if self.hit_soft_17 else "S17"


DEFAULT_RULES = Rules()


def _place_bet(shoe, bankroll, betting, minimum, maximum):
    if betting == "fixed":
        return min(minimum, bankroll)
    if betting == "count":
        return choose_bet(shoe.true_count, bankroll, minimum, maximum)
    raise ValueError("betting must be 'fixed' or 'count'")


def play_hand(shoe, bankroll, betting="count", policy=choose_action,
              rules=DEFAULT_RULES, minimum=10, maximum=100, hand_log=None):
    """Play one round; return (updated bankroll, profit/loss, initial bet).

    hand_log -- optional list; receives (cards_list, profit) for each final
                player hand, so a learning agent can credit its decisions.
    """
    bet = _place_bet(shoe, bankroll, betting, minimum, maximum)
    if bet <= 0:
        return bankroll, 0, 0

    player = [shoe.draw(), shoe.draw()]
    upcard = shoe.draw()
    hole_card = shoe.draw(exposed=False)
    dealer = [upcard, hole_card]

    # Naturals. The dealer peeks with a 10 or ace up, so a dealer blackjack
    # ends the round before the player acts (the player loses only the bet).
    if is_blackjack(dealer) or is_blackjack(player):
        shoe.reveal(hole_card)
        if is_blackjack(dealer):
            result = 0 if is_blackjack(player) else -bet
        else:
            result = bet * rules.blackjack_payout
        if hand_log is not None:
            hand_log.append((player, result))
        return bankroll + result, result, bet

    # Player's turn: hands may grow through splits.
    hands = [{"cards": player, "bet": bet, "split_aces": False}]
    committed = bet
    i = 0
    while i < len(hands):
        h = hands[i]
        cards = h["cards"]
        while not h["split_aces"] and hand_value(cards) < 21:
            two = len(cards) == 2
            afford = bankroll - committed >= h["bet"]
            can_double = two and afford and (len(hands) == 1 or rules.double_after_split)
            can_split = (two and cards[0].value == cards[1].value
                         and len(hands) < rules.max_hands and afford)
            action = policy(cards, upcard, can_double=can_double, can_split=can_split)
            if action == "STAND":
                break
            if action == "HIT":
                cards.append(shoe.draw())
            elif action == "DOUBLE" and can_double:
                committed += h["bet"]
                h["bet"] *= 2
                cards.append(shoe.draw())
                break
            elif action == "SPLIT" and can_split:
                committed += h["bet"]
                aces = cards[0].rank == "A"
                new = {"cards": [cards.pop(), shoe.draw()], "bet": h["bet"], "split_aces": aces}
                cards.append(shoe.draw())
                h["split_aces"] = aces
                hands.insert(i + 1, new)
            else:
                raise ValueError(f"policy returned illegal action {action!r}")
        i += 1

    # Dealer's turn (skipped if every player hand busted).
    shoe.reveal(hole_card)
    if any(hand_value(h["cards"]) <= 21 for h in hands):
        while dealer_should_hit(dealer, rules.hit_soft_17):
            dealer.append(shoe.draw())
    dealer_total = hand_value(dealer)

    result = 0
    for h in hands:
        total = hand_value(h["cards"])
        if total > 21:
            r = -h["bet"]
        elif dealer_total > 21 or total > dealer_total:
            r = h["bet"]
        elif total < dealer_total:
            r = -h["bet"]
        else:
            r = 0
        result += r
        if hand_log is not None:
            hand_log.append((h["cards"], r))
    return bankroll + result, result, bet


def run_session(seed, hands=100, starting_bankroll=1000, betting="count",
                policy=choose_action, rules=DEFAULT_RULES, minimum=10, maximum=100):
    if policy is choose_action:
        strategy.configure(decks=rules.decks, dealer_rule=rules.dealer_rule)
    shoe = Shoe(rules.decks, seed)
    if hasattr(policy, "attach"):       # agents that look at the remaining cards
        policy.attach(shoe, rules)
    bankroll = starting_bankroll
    played = wagered = 0
    sum_sq = 0.0                         # sum of squared hand results, for the CI
    for _ in range(hands):
        if bankroll < minimum:
            break
        if shoe.cards_remaining < rules.reshuffle_below:
            shoe.reshuffle()
        bankroll, result, bet = play_hand(shoe, bankroll, betting, policy, rules, minimum, maximum)
        wagered += bet
        sum_sq += result * result
        played += 1
    return {
        "final_bankroll": bankroll,
        "profit": bankroll - starting_bankroll,
        "hands_played": played,
        "initial_wagered": wagered,
        "sum_sq": sum_sq,
        "depleted": bankroll < minimum,
    }


def summarize(sessions):
    profits = [s["profit"] for s in sessions]
    wagered = sum(s["initial_wagered"] for s in sessions)
    n = len(profits)
    # per-hand view: edge = profit per chip of initial bet, with a standard error
    hands = sum(s["hands_played"] for s in sessions)
    mean_hand = sum(profits) / hands if hands else 0.0
    var_hand = (sum(s["sum_sq"] for s in sessions) / hands - mean_hand ** 2) if hands else 0.0
    mean_bet = wagered / hands if hands else 1.0
    se_edge = (max(var_hand, 0.0) / hands) ** 0.5 / mean_bet if hands else 0.0
    return {
        "hands": hands,
        "se_edge": se_edge,
        "mean_profit": mean(profits),
        "se_profit": stdev(profits) / n ** 0.5 if n > 1 else 0.0,
        "edge": sum(profits) / wagered if wagered else 0.0,   # profit per chip bet
        "ahead": sum(p > 0 for p in profits),
        "depleted": sum(s["depleted"] for s in sessions),
        "mean_hands": mean(s["hands_played"] for s in sessions),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--policy", choices=list(POLICIES), default="dataset")
    ap.add_argument("--betting", choices=("count", "fixed"), default="count")
    ap.add_argument("--s17", action="store_true", help="dealer stands on soft 17")
    args = ap.parse_args(argv)

    rules = Rules(hit_soft_17=not args.s17)
    policy = get_policy(args.policy)
    sessions = [run_session(i, args.hands, betting=args.betting,
                            policy=policy, rules=rules)
                for i in range(args.sessions)]
    r = summarize(sessions)
    n = args.sessions
    print(f"=== {args.betting.upper()} BETTING, {args.policy.upper()} POLICY ===")
    print(f"{n} sessions; up to {args.hands} hands; bankroll 1,000; "
          f"{rules.decks} decks {rules.dealer_rule}; bets 10-100")
    print(f"Mean profit/loss: {r['mean_profit']:.2f} ± {1.96 * r['se_profit']:.2f} chips (95% CI)")
    print(f"Return per chip wagered: {100 * r['edge']:+.2f}% ± {196 * r['se_edge']:.2f}%")
    print(f"Sessions ending ahead: {r['ahead']}/{n}   depleted: {r['depleted']}/{n}")


if __name__ == "__main__":
    main()
