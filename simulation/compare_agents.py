"""Compare all playing agents in the simulator, on the same seeds.

Run from the repo root:
    python3 -m simulation.compare_agents                 # default 1000 x 200 hands
    python3 -m simulation.compare_agents --sessions 3000 --agents dataset search

Two experiments per agent:
  1. Decision quality vs the dataset (6 decks, H17): how often the agent's
     first decision matches the table's best action, and the EV it gives up
     (cents per dollar) when it doesn't.
  2. Play in the simulator with flat 10-chip bets (isolates playing skill)
     and with Hi-Lo count betting. Edge = profit per chip of initial bet.

Writes simulation/results/agent_comparison.csv.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

from search.search import composition
from simulation.blackjack import Card, Shoe
from simulation.run_simulation import Rules, get_policy, run_session, summarize
from simulation.strategy import find_dataset

AGENTS = {
    "baseline": "Baseline rule (hand-written)",
    "dataset": "Lookup table (dataset)",
    "tree": "ML: decision tree",
    "mc": "ML: Monte Carlo control (RL)",
    "search": "Search: expectimax",
}
EV_COLUMNS = {"STAND": "stand_ev", "HIT": "hit_ev", "DOUBLE": "double_ev", "SPLIT": "split_ev"}
LETTERS = {"S": "STAND", "H": "HIT", "D": "DOUBLE", "P": "SPLIT"}
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class _FixedShoe:
    """Stands in for a fresh shoe minus the dealt cards, for table scoring."""
    def __init__(self, decks, removed):
        cards = Shoe(decks, 0).cards
        for c in removed:
            cards.remove(c)
        self.cards, self.hidden, self.decks = cards, [], decks

    @property
    def unseen_cards(self):
        return self.cards


def _cards_for(row):
    cat, hand = row["category"], row["hand"]
    if cat in ("pair", "soft"):
        a, b = hand.split(",")
        return [Card(a), Card(b)]
    t = int(hand)
    hi = min(10, t - 2)
    lo = t - hi
    if hi == lo:
        hi, lo = hi - 1, lo + 1
    return [Card(str(hi)), Card(str(lo))]


def table_score(policy, rows, rules):
    """Agreement with the dataset's best action, and mean EV regret (c/$)."""
    match = regret = n = 0
    for r in rows:
        cards = _cards_for(r)
        up = Card(r["dealer_upcard"])
        if hasattr(policy, "attach"):
            policy.attach(_FixedShoe(rules.decks, cards + [up]), rules)
        action = policy(cards, up, can_double=True, can_split=r["category"] == "pair")
        evs = {a: float(r[c]) for a, c in EV_COLUMNS.items() if r.get(c) not in ("", None)}
        best = LETTERS[r["best_action"]]
        match += action == best
        regret += evs[best] - evs.get(action, min(evs.values()))
        n += 1
    return match / n, regret / n


_worker_policies = {}


def _run_chunk(job):
    """Run a range of seeds in a worker process (each worker builds its own agent)."""
    key, betting, seeds, hands = job
    if key not in _worker_policies:
        _worker_policies[key] = get_policy(key)
    policy = _worker_policies[key]
    return [run_session(s, hands, betting=betting, policy=policy, rules=Rules()) for s in seeds]


def run_parallel(pool, key, betting, sessions, hands, workers):
    chunks = [range(i, sessions, workers) for i in range(workers)]
    jobs = [(key, betting, list(c), hands) for c in chunks if len(c)]
    return [s for part in pool.map(_run_chunk, jobs) for s in part]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare playing agents.")
    ap.add_argument("--agents", nargs="+", choices=list(AGENTS), default=list(AGENTS))
    ap.add_argument("--sessions", type=int, default=1000)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes to run in parallel (default: all CPU cores)")
    args = ap.parse_args(argv)

    rules = Rules()
    rows = []
    if find_dataset():
        from machine_learning.classifier import load_rows
        rows = [r for r in load_rows() if int(r["decks"]) == rules.decks
                and r["dealer_rule"] == rules.dealer_rule]

    if not rows:
        print("player-ev.json not found: table scores skipped; dataset and tree agents can't run.\n")
    out = []
    pool = ProcessPoolExecutor(args.workers)
    print(f"{args.sessions} sessions x {args.hands} hands per agent and betting mode "
          f"(same seeds); {rules.decks} decks {rules.dealer_rule}; {args.workers} workers\n")
    head = (f"{'Agent':<30}{'Table match':>12}{'Regret c/$':>11}"
            f"{'Flat edge':>11}{'±95%':>7}{'Count edge':>12}{'±95%':>7}{'Time':>8}")
    print(head)
    print("-" * len(head))
    for key in args.agents:
        try:
            policy = get_policy(key)
        except FileNotFoundError as e:
            print(f"{AGENTS[key]:<30}skipped: {e}")
            continue
        match, regret = table_score(policy, rows, rules) if rows else (float("nan"),) * 2
        res = {}
        t0 = time.time()
        for betting in ("fixed", "count"):
            res[betting] = summarize(run_parallel(pool, key, betting, args.sessions,
                                                  args.hands, args.workers))
        secs = time.time() - t0
        f, c = res["fixed"], res["count"]
        tm = f"{match:>11.1%}{regret:>11.3f}" if rows else f"{'n/a':>12}{'n/a':>11}"
        print(f"{AGENTS[key]:<30}{tm}"
              f"{100 * f['edge']:>10.2f}%{196 * f['se_edge']:>6.2f}%"
              f"{100 * c['edge']:>11.2f}%{196 * c['se_edge']:>6.2f}%{secs:>7.0f}s")
        out.append((AGENTS[key], round(match, 4), round(regret, 4),
                    round(f["edge"], 5), round(f["se_edge"], 5),
                    round(c["edge"], 5), round(c["se_edge"], 5), c["depleted"], round(secs, 1)))

    pool.shutdown()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "agent_comparison.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "table_match", "table_regret_cents", "flat_edge", "flat_edge_se",
                    "count_edge", "count_edge_se", "count_sessions_depleted", "seconds"])
        w.writerows(out)
    print(f"\nEdges are profit per chip of initial bet; ±95% is 1.96 standard errors."
          f"\nWrote {path}")


if __name__ == "__main__":
    main()
