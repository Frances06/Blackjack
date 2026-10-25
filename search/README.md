# ♠ Search & Decision-Making

<p align="center">
  <strong>THINK THROUGH THE POSSIBILITIES.</strong>
  <br />
  Search algorithms to explore game states, model dealer outcomes and choose high-value actions.
</p>

## <span style="color:red">♦</span> The Search Toolkit

### `search.py` — Exploring the State Space

Provides general-purpose graph-search algorithms that work with hashable states:

- **DFS** — depth-first search.
- **A\*** — heuristic-guided graph search.

These are general tools. Blackjack's uncertain card draws require an additional approach.

### `agent.py` — Expectimax

The `ExpectimaxAgent` evaluates possible player decisions against the possible cards that could be drawn.

Its game tree alternates between:

- **Decision nodes:** the player chooses an available action — stand, hit, double or split.
- **Chance nodes:** a card is drawn according to the remaining shoe's composition.

At the leaves, the agent evaluates outcomes using the dealer's final-hand distribution and selects the action with the highest expected value.

Unlike a fixed lookup table, the agent can account for which cards have already been seen.

## <span style="color:red">♥</span> Modelling the Dealer

The `dealer_outcomes` and `dealer_outcomes_from` functions calculate the distribution of the dealer's final hand using depth-first enumeration of possible draw sequences.

Results are memoised by card composition to avoid repeatedly solving equivalent states.

For example, the implementation matches reference figures such as a dealer bust probability of approximately 42.3% with a six showing in a fresh six-deck shoe under stand-on-soft-17 rules.

## ♣ Assumptions & Limitations

- Card probabilities are held at the current shoe composition throughout an individual decision's search.
- Split hands are evaluated as two hands without further re-splitting.
- The agent's conclusions depend on the accuracy of the game rules and outcome model.

These approximations keep the search manageable while retaining information about the remaining cards.

## <span style="color:red">♦</span> Why Expectimax?

Ordinary DFS and A* explore paths through a state space, but Blackjack also involves uncertain outcomes: the next card is not a choice the player controls.

Expectimax handles this by combining decision-making with chance nodes, making it a natural fit for evaluating Blackjack actions under uncertainty.
