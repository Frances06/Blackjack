# Machine learning agents

Put the supplied `player-ev.json` in `data/`.

## 1. Supervised: classifier (`classifier.py`, `train_classifier.py`)
Learns to predict `best_action` from the dataset.

- Features: `player_total`, `is_soft`, `pair_value`, `dealer_upcard`, `decks`,
  `h17`. All EV columns are excluded from the inputs to avoid target leakage.
- `train_classifier.py` compares a majority-class baseline, decision tree,
  random forest, k-NN, logistic regression and an MLP with 5-fold
  cross-validation, reporting accuracy and **EV regret** (cents per dollar
  lost through wrong predictions). It also tunes tree depth, tests
  generalisation to an unseen deck count, and writes the readable tree to
  `results/tree_rules.txt`.
- `TreeAgent` plays with the decision tree. Because it works from totals, it
  also answers for hands the table doesn't list (3+ cards). A second tree,
  trained on stand-vs-hit, is used when the main tree says DOUBLE but doubling
  isn't allowed.

## 2. Reinforcement learning: Monte Carlo control (`monte_carlo.py`)
Never sees the dataset; learns Q(state, action) by playing in the simulator
and averaging the returns of each decision, with epsilon-greedy exploration.
Training writes `models/mc_q.json` (used by the `mc` agent) and
`results/mc_learning_curve.csv` (greedy edge vs hands played).

```
python3 -m machine_learning.monte_carlo --episodes 10000000
```

The dataset has no card history, count, bankroll, or bet-size target, so
count-dependent play is handled by the search agent, not these models.
