"""Generic graph search (DFS, A*) plus a DFS-based exact dealer-outcome calculator.

The generic searches take callables, so they work on any graph: a grid, a game
tree, a state space you define. Nodes only need to be hashable.
"""
from collections import Counter
from functools import lru_cache
from heapq import heappush, heappop
from itertools import count



# ---------------------------------------------------------------------------
# Depth-first search
# ---------------------------------------------------------------------------
def dfs(start, neighbors, goal=None, max_depth=None):
    """Iterative depth-first search.

    start      -- starting node (hashable)
    neighbors  -- callable: node -> iterable of adjacent nodes
    goal       -- optional callable: node -> bool. If given, returns the first
                  path found from start to a goal node (list of nodes), or None.
                  If omitted, returns every reachable node in DFS visit order.
    max_depth  -- optional limit on path length (edges) from start.

    Iterative rather than recursive, so deep graphs don't hit Python's
    recursion limit. Neighbours are explored in the order `neighbors` yields them.
    """
    visited = set()
    order = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if goal is not None and goal(node):
            return path
        if max_depth is not None and len(path) - 1 >= max_depth:
            continue
        # reversed so the first neighbour yielded is explored first
        for nxt in reversed(list(neighbors(node))):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return None if goal is not None else order


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
def astar(start, goal, neighbors, heuristic=lambda n: 0):
    """A* shortest path.

    start      -- starting node (hashable)
    goal       -- a target node, or a callable node -> bool
    neighbors  -- callable: node -> iterable of (neighbor, step_cost), cost >= 0
    heuristic  -- callable: node -> estimated remaining cost. Must never
                  overestimate (admissible) for the result to be optimal.
                  Default 0 makes this Dijkstra's algorithm.

    Returns (path, total_cost), or (None, inf) if the goal is unreachable.
    """
    is_goal = goal if callable(goal) else (lambda n: n == goal)
    tie = count()                      # tiebreaker so nodes never get compared
    open_heap = [(heuristic(start), next(tie), start)]
    g = {start: 0}
    came_from = {}
    closed = set()

    while open_heap:
        _, _, node = heappop(open_heap)
        if node in closed:
            continue                   # stale heap entry
        if is_goal(node):
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            return path[::-1], g[path[0]]
        closed.add(node)
        for nxt, cost in neighbors(node):
            if cost < 0:
                raise ValueError("A* requires non-negative edge costs")
            new_g = g[node] + cost
            if nxt not in closed and new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came_from[nxt] = node
                heappush(open_heap, (new_g + heuristic(nxt), next(tie), nxt))
    return None, float("inf")


# ---------------------------------------------------------------------------
# Blackjack application: exact dealer outcomes via DFS over the shoe
# ---------------------------------------------------------------------------
_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)   # 11 = ace; J/Q/K folded into 10
_KEYS = (17, 18, 19, 20, 21, "bust", "blackjack")


def composition(cards):
    """Count of each value 2..11 (ace = 11) in a list of Cards, as a tuple."""
    c = Counter(card.value for card in cards)
    return tuple(c.get(v, 0) for v in _VALUES)


def dealer_outcomes(shoe, upcard, hit_soft_17=False, dealer_peeked=False):
    """Exact probability distribution of the dealer's final hand.

    Does a DFS over every possible draw sequence from the cards actually left
    in `shoe` (finite-shoe, so it reflects the current count), weighting each
    branch by its draw probability. Memoised on (composition, total, soft).

    upcard        -- a Card or a value 2..11
    hit_soft_17   -- dealer rule; match the game loop (simulation default is H17)
    dealer_peeked -- condition on the dealer NOT having blackjack, which is what
                     the player knows when acting after a peek (10 or ace up).
    Note: `shoe` should be the cards unseen by the player; in the simulator the
    hole card has already been popped from shoe.cards, so add it back if you want
    the exact pre-reveal distribution.
    Returns dict with keys 17..21, 'bust', 'blackjack' summing to 1.
    """
    up = upcard if isinstance(upcard, int) else upcard.value
    return dealer_outcomes_from(composition(shoe.cards), up, hit_soft_17, dealer_peeked)


def dealer_outcomes_from(comp, up, hit_soft_17=False, dealer_peeked=False):
    """Same as dealer_outcomes, from a composition tuple (see `composition`)
    and an upcard value 2..11. Used by the search agent."""

    # Outcomes are kept as a 7-tuple of probabilities, in the order of KEYS,
    # which is several times faster than building dicts at every node.
    zero = (0.0,) * 7
    leaf = {k: tuple(1.0 if i == j else 0.0 for j in range(7)) for i, k in enumerate(_KEYS)}

    @lru_cache(maxsize=None)
    def walk(comp, hard, has_ace, n_cards):
        soft_total = hard + 10 if has_ace and hard + 10 <= 21 else hard
        is_soft = soft_total != hard
        if n_cards == 2 and soft_total == 21:
            return leaf["blackjack"]
        if hard > 21:
            return leaf["bust"]
        if soft_total > 17 or (soft_total == 17 and not (is_soft and hit_soft_17)):
            return leaf[soft_total]

        remaining = sum(comp)
        if remaining == 0:
            raise RuntimeError("Shoe ran out during dealer DFS")
        out = list(zero)
        for i, n in enumerate(comp):
            if not n:
                continue
            v = _VALUES[i]
            nxt = comp[:i] + (n - 1,) + comp[i + 1:]
            sub = walk(nxt, hard + (1 if v == 11 else v), has_ace or v == 11, n_cards + 1)
            p = n / remaining
            for j in range(7):
                out[j] += p * sub[j]
        return tuple(out)

    result = dict(zip(_KEYS, walk(comp, 1 if up == 11 else up, up == 11, 1)))
    out = result
    if dealer_peeked and out["blackjack"] < 1:
        p_bj = out["blackjack"]
        out = {k: (0.0 if k == "blackjack" else v / (1 - p_bj)) for k, v in out.items()}
    return out