# Blackjack agent

A blackjack simulator with Hi-Lo card counting, and several AI agents that
play it, compared on the same games:

| Agent | Type | How it decides |
|---|---|---|
| `baseline` | hand-written rule | hit below 12, simple stand rules |
| `dataset` | lookup | best action from `player-ev.json` |
| `tree` | ML (supervised) | decision tree trained on `player-ev.json` |
| `mc` | ML (reinforcement learning) | Monte Carlo control, learns only by playing |
| `search` | search | expectimax over the cards left in the shoe, using an exact dealer DFS |

All agents bet with the same Hi-Lo count ramp (or flat bets), so differences
come from how they play.

## Setup
Python 3.9+, scikit-learn (for the `tree` agent and classifier comparison) and matplotlib (for charts):
```
python3 -m pip install -r requirements.txt
```
Put `player-ev.json` in `machine_learning/data/` (or set `BLACKJACK_EV_PATH`).

## Run (from the repo root)
```
python3 -m unittest simulation.test_simulation       # tests

# compare every agent (uses all CPU cores; a few minutes)
python3 -m simulation.compare_agents

# machine learning
python3 -m machine_learning.train_classifier         # compare classifiers, tune the tree
python3 -m machine_learning.monte_carlo              # (re)train the RL agent

# one agent / betting experiments
python3 -m simulation.run_simulation --policy search
python3 -m simulation.run_experiments --policy dataset --sessions 2000 --hands 200
```
Each script writes CSVs to a `results/` folder next to it. To turn them into
charts for the report (PNG files in `analysis/figures/`):
```
python3 -m analysis.make_charts
```

## Rules modelled
6 decks, dealer hits soft 17 (`--s17` to change), blackjack pays 3:2, dealer
peeks with 10/A up, double on any two cards, one split (split aces get one
card), double after split, reshuffle with 78 cards left. No insurance/surrender.

## Layout
- `simulation/` – shoe and counting (`blackjack.py`), betting and lookup
  policies (`strategy.py`), game loop (`run_simulation.py`), betting
  experiment (`run_experiments.py`), agent comparison (`compare_agents.py`), tests.
- `search/` – generic DFS and A* (`search.py`), exact dealer-outcome DFS,
  and the expectimax playing agent (`agent.py`).
- `analysis/` – `make_charts.py` turns the results CSVs into report figures.
- `machine_learning/` – classifier agent and model comparison
  (`classifier.py`, `train_classifier.py`), Monte Carlo control agent
  (`monte_carlo.py`), trained Q table in `models/`, dataset in `data/`.

## Data
`player-ev.json` – "Blackjack Basic-Strategy Expected Value by Action" v1.0
(updated 2026-08-09), CC BY 4.0, methodology at
https://blackjackoddstrainer.com/methodology/. 3,300 rows: 1/2/4/6/8 decks ×
S17/H17 × hard/soft/pair hands × dealer upcard, with stand/hit/double/split
EVs in cents per dollar. The EVs are modelled (fixed-proportion draws,
approximate split values), not simulated. Cite the source and access date
in the report.
