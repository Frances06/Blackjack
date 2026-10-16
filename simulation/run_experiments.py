"""Compare fixed and count-based betting on matching random seeds."""
from statistics import mean
from simulation.run_simulation import run_session


def summarize(sessions):
    profits = [session["profit"] for session in sessions]
    return {
        "mean_profit": mean(profits),
        "ahead": sum(profit > 0 for profit in profits),
        "depleted": sum(session["depleted"] for session in sessions),
        "mean_hands": mean(session["hands_played"] for session in sessions),
    }


def main():
    seeds = range(500)
    fixed = [run_session(seed, betting="fixed") for seed in seeds]
    counted = [run_session(seed, betting="count") for seed in seeds]
    print("=== BETTING POLICY COMPARISON (EXPLORATORY) ===")
    print("500 matched seeds; up to 100 hands/session; initial bankroll 1,000 chips")
    print("Same simplified hit/stand playing policy in both conditions.")
    print("Policy             Mean P/L    Sessions ahead    Depleted    Mean hands")
    for name, sessions in (("Fixed 10-chip", fixed), ("Count-based", counted)):
        result = summarize(sessions)
        print(f"{name:<18} {result['mean_profit']:>9.2f} "
              f"{result['ahead']:>12}/500 {result['depleted']:>10}/500 "
              f"{result['mean_hands']:>12.1f}")
    print("\nCaution: this isolates bet sizing only; it does not prove card counting improves expected value.")
    print("The game rules and playing policy are simplified.")


if __name__ == "__main__":
    main()
