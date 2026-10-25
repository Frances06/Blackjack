# ♣ Machine Learning

<p align="center">
  <strong>TWO APPROACHES TO BLACKJACK INTELLIGENCE</strong>
  <br />
  Learn from expected-value data, or learn through experience.
</p>

## <span style=color:red>♦</span> 01 — Supervised Learning

**Files:** `classifier.py` · `train_classifier.py`

The classifier learns to predict the best action from the supplied expected-value dataset.

**Features**

- Uses player total, soft-hand status, pair value, dealer upcard, deck count and dealer rules.
- Excludes expected-value columns from model inputs to prevent target leakage.
- Compares a majority-class baseline, decision tree, random forest, k-nearest neighbours, logistic regression and an MLP.
- Uses five-fold cross-validation and reports accuracy alongside **EV regret**: the expected-value cost of incorrect decisions.
- Tunes decision-tree depth and tests generalisation to an unseen deck count.

The resulting `TreeAgent` can also make decisions for hands containing three or more cards, even when those hands aren't explicitly listed in the dataset.

When doubling is unavailable but the tree recommends it, a secondary stand-versus-hit tree handles the decision.

## ♠ 02 — Reinforcement Learning

**File:** `monte_carlo.py`

Unlike the classifier, the Monte Carlo agent **never sees the expected-value dataset**.

Instead, it learns by playing. It estimates the value of state-action pairs, \(Q(s,a)\), by averaging observed returns and uses epsilon-greedy exploration to balance trying actions with choosing those currently estimated to be best.

Training produces:

| Output | Purpose |
|:--|:--|
| `models/mc_q.json` | Trained Q-table used by the `mc` agent |
| `results/mc_learning_curve.csv` | Learning curve tracking greedy edge against hands played |

Train the agent with:

```bash
python3 -m machine_learning.monte_carlo --episodes 10000000
```

## <span style=color:red>♥</span> One Important Distinction

The dataset describes hand-level expected values; it does not contain card history, running count, bankroll or bet-size targets.

Thus, these models do not learn count-dependent betting or playing from that dataset. The search agent handles card-composition-dependent play separately.
