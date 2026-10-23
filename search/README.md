# Search

- `search.py`
  - `dfs`, `astar`: generic graph search over any hashable states.
  - `dealer_outcomes` / `dealer_outcomes_from`: exact distribution of the
    dealer's final hand, by DFS over every draw sequence from the remaining
    cards (memoised on composition). Matches published figures, e.g. 42.3%
    bust with a 6 up on a fresh 6-deck shoe (S17).
- `agent.py` – `ExpectimaxAgent`: at each decision, builds the game tree of
  the player's options (max nodes: stand/hit/double/split) and the cards
  that could be drawn (chance nodes), scores leaves with the dealer
  distribution above, and plays the highest-EV action. It uses the cards the
  player hasn't seen, so unlike a fixed table it adapts to the count.
  Approximation: within one decision, card probabilities are held at the
  current composition; splits are scored as two hands without re-splitting.

Ordinary DFS/A* can't solve stochastic blackjack on their own (there is no
single "goal path"); expectimax is the variant of game-tree search that
handles chance nodes, which is why the agent uses it.
