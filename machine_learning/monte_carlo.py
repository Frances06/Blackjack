"""Reinforcement learning agent: on-policy Monte Carlo control.

Unlike the classifier, this agent never sees player-ev.json. It learns
action values Q(state, action) purely by playing hands in the simulator and
averaging the money each decision ended up winning or losing (Sutton & Barto,
ch. 5: on-policy first-visit MC control with epsilon-greedy exploration).

State   : (player_total, is_soft, pair_value or 0, dealer_upcard, can_double)
Actions : STAND, HIT, DOUBLE, SPLIT (only the legal ones in that state)
Reward  : profit of the round in units of the initial bet. Decisions taken
          after a split are credited with their own hand's result; the split
          decision itself gets the combined result of both hands.
Update  : Q += alpha * (G - Q) with alpha = max(1/visits, alpha_min). Pure
          averaging (alpha_min = 0) never forgets the returns collected while
          the policy was still mostly random, which slows learning badly; a
          small floor lets old experience fade as the policy improves.

Train (from the repo root):
    python3 -m machine_learning.monte_carlo --episodes 2000000
which writes models/mc_q.json and results/mc_learning_curve.csv.

Tuning (10M hands each, greedy edge on 200k evaluation hands; see
results/mc_settings_comparison.csv):
    pure averaging, epsilon 1.0 -> 0.05          -1.81%
    alpha_min 0.001, epsilon 1.0 -> 0.02         -1.23%
    alpha_min 0.0003, epsilon 0.3 -> 0.02        -0.89%   <- default
Perfect basic strategy under these rules is about -0.6%.
"""
import argparse
import csv
import json
import os
import random
import time

from simulation.blackjack import Shoe, hand_value, is_soft
from simulation.run_simulation import Rules, play_hand, run_session, summarize
from simulation.strategy import choose_action_baseline

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "mc_q.json")
RESULTS = os.path.join(HERE, "results")


def state_of(cards, upcard, can_double, can_split):
    pair = cards[0].value if can_split else 0
    return (hand_value(cards), int(is_soft(cards)), pair, upcard.value,
            int(can_double and len(cards) == 2))


def legal_actions(state):
    total, _, pair, _, can_double = state
    acts = ["STAND", "HIT"]
    if can_double:
        acts.append("DOUBLE")
    if pair:
        acts.append("SPLIT")
    return acts


class MonteCarloAgent:
    """Greedy policy from a learned Q table (also used, with epsilon, while training)."""

    def __init__(self, q=None, counts=None, epsilon=0.0, seed=0, alpha_min=0.0):
        self.q = q if q is not None else {}          # state -> {action: value}
        self.n = counts if counts is not None else {}  # state -> {action: visits}
        self.epsilon = epsilon
        self.alpha_min = alpha_min
        self.rng = random.Random(seed)
        self._visits = []
        self._split = False

    # ------------------------------------------------------------- playing
    def __call__(self, cards, upcard, can_double=False, can_split=False):
        s = state_of(cards, upcard, can_double, can_split)
        legal = legal_actions(s)
        if self.epsilon and self.rng.random() < self.epsilon:
            a = self.rng.choice(legal)
        else:
            a = self._greedy(s, legal, cards, upcard)
        if self.epsilon:                              # training: remember the visit
            self._visits.append((cards, s, a, self._split))
            if a == "SPLIT":
                self._split = True
        return a

    def _greedy(self, s, legal, cards, upcard):
        values = self.q.get(s)
        if not values:
            return choose_action_baseline(cards, upcard)   # never visited
        known = [a for a in legal if a in values]
        return max(known, key=values.get) if known else choose_action_baseline(cards, upcard)

    # ------------------------------------------------------------ learning
    def learn(self, hand_log):
        """Update Q from one finished round (hand_log from play_hand)."""
        total = sum(r for _, r in hand_log)
        own = {id(c): r for c, r in hand_log}
        for cards, s, a, after_split in self._visits:
            g = own.get(id(cards), 0.0) if after_split else total
            n = self.n.setdefault(s, {}).get(a, 0) + 1
            self.n[s][a] = n
            qs = self.q.setdefault(s, {})
            alpha = max(1.0 / n, self.alpha_min)
            qs[a] = qs.get(a, 0.0) + alpha * (g - qs.get(a, 0.0))
        self._visits.clear()
        self._split = False

    # --------------------------------------------------------- persistence
    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [{"state": list(s), "q": self.q[s], "n": self.n.get(s, {})} for s in sorted(self.q)]
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found; train it first with `python3 -m machine_learning.monte_carlo`")
        with open(path) as f:
            data = json.load(f)
        q = {tuple(d["state"]): d["q"] for d in data}
        n = {tuple(d["state"]): d["n"] for d in data}
        return cls(q, n)


def evaluate(agent, hands=50_000, seeds=50):
    """Edge (profit per chip, flat bets) of the greedy policy on fixed seeds."""
    greedy = MonteCarloAgent(agent.q, agent.n)
    per = hands // seeds
    sessions = [run_session(10_000 + s, per, starting_bankroll=10**9, betting="fixed",
                            policy=greedy) for s in range(seeds)]
    r = summarize(sessions)
    return r["edge"], r["se_edge"]


def train(episodes=10_000_000, seed=0, eps_start=0.3, eps_end=0.02, explore_frac=0.6,
          alpha_min=0.0003, checkpoints=10, eval_hands=200_000, verbose=True):
    rules = Rules()
    shoe = Shoe(rules.decks, seed)
    agent = MonteCarloAgent(epsilon=eps_start, seed=seed, alpha_min=alpha_min)
    curve = []
    every = max(1, episodes // checkpoints)
    t0 = time.time()
    for ep in range(1, episodes + 1):
        frac = min(1.0, ep / (explore_frac * episodes))
        agent.epsilon = eps_start + (eps_end - eps_start) * frac
        if shoe.cards_remaining < rules.reshuffle_below:
            shoe.reshuffle()
        log = []
        play_hand(shoe, 10**9, "fixed", agent, rules, minimum=1, maximum=1, hand_log=log)
        agent.learn(log)
        if ep % every == 0:
            edge, se = evaluate(agent, eval_hands)
            curve.append((ep, round(agent.epsilon, 4), round(edge, 5), round(se, 5)))
            if verbose:
                print(f"{ep:>10,} episodes  eps={agent.epsilon:.2f}  greedy edge "
                      f"{100 * edge:+.2f}% ± {196 * se:.2f}%  ({time.time() - t0:.0f}s)")
    agent.epsilon = 0.0
    return agent, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the Monte Carlo control agent.")
    ap.add_argument("--episodes", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", type=int, default=10)
    args = ap.parse_args(argv)

    agent, curve = train(args.episodes, args.seed, checkpoints=args.checkpoints)
    agent.save()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "mc_learning_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodes", "epsilon", "greedy_edge", "edge_se"])
        w.writerows(curve)
    print(f"Saved Q table ({len(agent.q)} states) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
