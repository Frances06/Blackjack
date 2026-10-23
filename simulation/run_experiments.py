"""Compare fixed and count-based betting on matching random seeds.

Playing decisions don't depend on the bet, so with the same seed both
conditions see the same cards (until one bankroll runs out) - a paired design.
"""
import argparse

from statistics import mean, stdev

from simulation.run_simulation import POLICIES, Rules, get_policy, run_session, summarize


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=int, default=500)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--policy", choices=list(POLICIES), default="dataset")
    args = ap.parse_args(argv)

    rules = Rules()
    policy = get_policy(args.policy)
    seeds = range(args.sessions)
    runs = {
        "Fixed 10-chip": [run_session(s, args.hands, betting="fixed", policy=policy, rules=rules) for s in seeds],
        "Count-based": [run_session(s, args.hands, betting="count", policy=policy, rules=rules) for s in seeds],
    }
    n = args.sessions
    print("=== BETTING POLICY COMPARISON ===")
    print(f"{n} matched seeds; up to {args.hands} hands/session; bankroll 1,000; "
          f"{rules.decks} decks {rules.dealer_rule}; playing policy: {args.policy}")
    print(f"{'Policy':<15}{'Mean P/L':>10}{'±95% CI':>10}{'Edge':>9}{'Ahead':>9}{'Depleted':>10}{'Hands':>8}")
    for name, sessions in runs.items():
        r = summarize(sessions)
        print(f"{name:<15}{r['mean_profit']:>10.2f}{1.96 * r['se_profit']:>10.2f}"
              f"{100 * r['edge']:>8.2f}%{r['ahead']:>9}{r['depleted']:>10}{r['mean_hands']:>8.1f}")
    # Paired test: same seed -> same cards, so compare seed by seed.
    diffs = [c["profit"] - f["profit"] for f, c in zip(runs["Fixed 10-chip"], runs["Count-based"])]
    half = 1.96 * stdev(diffs) / len(diffs) ** 0.5
    print(f"\nPaired difference (count - fixed): {mean(diffs):.2f} ± {half:.2f} chips per session (95% CI)")
    print("Significant at 5% if that interval excludes zero.")
    print("\nIf the CI is wider than the mean, the difference isn't distinguishable from luck;"
          "\nincrease --sessions/--hands before drawing conclusions.")


if __name__ == "__main__":
    main()
