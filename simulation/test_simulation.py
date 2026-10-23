import json
import os
import tempfile
import unittest

from simulation import strategy
from simulation.blackjack import (Card, Shoe, HI_LO, hand_value, is_soft,
                                  is_blackjack, dealer_should_hit)
from simulation.run_simulation import Rules, play_hand
from simulation.strategy import choose_bet, choose_action_baseline


def stacked_shoe(*ranks):
    """Shoe whose next draws are `ranks` in order (deal: P, P, up, hole, ...)."""
    shoe = Shoe(1, seed=0)
    shoe.cards = [Card(r) for r in reversed(ranks)]
    return shoe


def scripted(*actions):
    """Policy that returns the given actions in order."""
    it = iter(actions)
    return lambda cards, up, can_double=False, can_split=False: next(it)


class TestFoundation(unittest.TestCase):
    def test_hi_lo(self):
        self.assertEqual(HI_LO["2"], 1)
        self.assertEqual(HI_LO["8"], 0)
        self.assertEqual(HI_LO["A"], -1)

    def test_ace_value(self):
        self.assertEqual(hand_value([Card("A"), Card("9")]), 20)
        self.assertEqual(hand_value([Card("A"), Card("9"), Card("5")]), 15)

    def test_soft_and_blackjack(self):
        self.assertTrue(is_soft([Card("A"), Card("6")]))
        self.assertFalse(is_soft([Card("A"), Card("6"), Card("K")]))
        self.assertTrue(is_blackjack([Card("A"), Card("K")]))
        self.assertFalse(is_blackjack([Card("7"), Card("7"), Card("7")]))

    def test_dealer_soft_17_rule(self):
        soft17 = [Card("A"), Card("6")]
        self.assertTrue(dealer_should_hit(soft17, hit_soft_17=True))
        self.assertFalse(dealer_should_hit(soft17, hit_soft_17=False))

    def test_bet_schedule(self):
        self.assertEqual(choose_bet(0, 1000), 10)
        self.assertEqual(choose_bet(2, 1000), 40)
        self.assertEqual(choose_bet(4, 1000), 100)
        self.assertEqual(choose_bet(4, 35), 35)

    def test_six_deck_shoe(self):
        self.assertEqual(Shoe(6, seed=1).cards_remaining, 312)

    def test_hidden_card_not_counted_until_revealed(self):
        shoe = Shoe(1, seed=7)
        card = shoe.draw(exposed=False)
        self.assertEqual(shoe.running_count, 0)
        shoe.reveal(card)
        self.assertEqual(shoe.running_count, HI_LO[card.rank])


class TestPlayHand(unittest.TestCase):
    def play(self, shoe, policy, rules=Rules()):
        return play_hand(shoe, 1000, betting="fixed", policy=policy, rules=rules)

    def test_player_blackjack_pays_3_to_2(self):
        _, result, _ = self.play(stacked_shoe("A", "K", "9", "7"), scripted())
        self.assertEqual(result, 15)

    def test_dealer_blackjack_takes_only_the_bet(self):
        _, result, _ = self.play(stacked_shoe("K", "Q", "A", "K"), scripted())
        self.assertEqual(result, -10)

    def test_both_blackjack_push(self):
        _, result, _ = self.play(stacked_shoe("A", "K", "A", "Q"), scripted())
        self.assertEqual(result, 0)

    def test_h17_dealer_hits_soft_17(self):
        # player 18 stands; dealer A,6 -> H17 draws a 2 -> 19 beats 18
        shoe = stacked_shoe("K", "8", "A", "6", "2")
        _, result, _ = self.play(shoe, scripted("STAND"))
        self.assertEqual(result, -10)

    def test_s17_dealer_stands_on_soft_17(self):
        shoe = stacked_shoe("K", "8", "A", "6", "2")
        _, result, _ = self.play(shoe, scripted("STAND"), Rules(hit_soft_17=False))
        self.assertEqual(result, 10)

    def test_double_doubles_the_stake(self):
        # 6,5 = 11 vs dealer 10,7; double draws K -> 21
        _, result, _ = self.play(stacked_shoe("6", "5", "10", "7", "K"), scripted("DOUBLE"))
        self.assertEqual(result, 20)

    def test_split_plays_two_hands(self):
        # 8,8 vs 10,7; split -> 8,K (18) and 8,Q (18); both stand, both win
        shoe = stacked_shoe("8", "8", "10", "7", "Q", "K")
        _, result, _ = self.play(shoe, scripted("SPLIT", "STAND", "STAND"))
        self.assertEqual(result, 20)

    def test_split_aces_get_one_card_and_no_blackjack(self):
        # A,A vs 10,7 split -> A,K and A,Q: 21 each, paid 1:1 not 3:2
        shoe = stacked_shoe("A", "A", "10", "7", "Q", "K")
        _, result, _ = self.play(shoe, scripted("SPLIT"))
        self.assertEqual(result, 20)

    def test_baseline_accepts_new_flags(self):
        self.assertEqual(choose_action_baseline([Card("5"), Card("6")], Card("9"),
                                                can_double=True, can_split=False), "HIT")


class TestDatasetStrategy(unittest.TestCase):
    """Uses a tiny fixture in the player-ev.json format, so it runs without the real file."""

    def setUp(self):
        rows = [
            {"decks": 6, "dealer_rule": "H17", "category": "hard", "hand": "11",
             "dealer_upcard": "6", "stand_ev": -15, "hit_ev": 30, "double_ev": 60,
             "split_ev": "", "best_action": "D"},
            {"decks": 6, "dealer_rule": "H17", "category": "pair", "hand": "8,8",
             "dealer_upcard": "10", "stand_ev": -54, "hit_ev": -53, "double_ev": -100,
             "split_ev": -48, "best_action": "P"},
        ]
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"rows": rows}, f)
        self._saved = (dict(strategy._settings), strategy._table)
        strategy._table = None
        strategy.configure(decks=6, dealer_rule="H17", path=self.path)

    def tearDown(self):
        strategy._settings.clear()
        strategy._settings.update(self._saved[0])
        strategy._table = self._saved[1]
        os.remove(self.path)

    def test_picks_double_only_when_legal(self):
        hand, up = [Card("5"), Card("6")], Card("6")
        self.assertEqual(strategy.choose_action(hand, up, can_double=True), "DOUBLE")
        self.assertEqual(strategy.choose_action(hand, up, can_double=False), "HIT")

    def test_face_cards_count_as_tens(self):
        self.assertEqual(strategy.choose_action([Card("8"), Card("8")], Card("K"),
                                                can_split=True), "SPLIT")

    def test_missing_dataset_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            strategy.StrategyTable.load("/nonexistent/player-ev.json")
        self.assertIn("machine_learning/data", str(ctx.exception))


class TestSearch(unittest.TestCase):
    def test_dealer_outcomes_fresh_shoe(self):
        from search.search import dealer_outcomes
        out = dealer_outcomes(Shoe(6, seed=1), 6, hit_soft_17=False)
        self.assertAlmostEqual(sum(out.values()), 1.0)
        self.assertAlmostEqual(out["bust"], 0.423, places=2)   # published figure

    def test_peek_removes_blackjack(self):
        from search.search import dealer_outcomes
        out = dealer_outcomes(Shoe(6, seed=1), 11, dealer_peeked=True)
        self.assertEqual(out["blackjack"], 0.0)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_astar_and_dfs_on_small_graph(self):
        from search.search import astar, dfs
        g = {"A": [("B", 1), ("C", 4)], "B": [("C", 1), ("D", 5)], "C": [("D", 1)], "D": []}
        path, cost = astar("A", "D", lambda n: g[n])
        self.assertEqual((path, cost), (["A", "B", "C", "D"], 3))
        self.assertEqual(dfs("A", lambda n: [m for m, _ in g[n]])[0], "A")


class TestShoeVisibility(unittest.TestCase):
    def test_unseen_cards_include_hole_card(self):
        shoe = Shoe(1, seed=3)
        hole = shoe.draw(exposed=False)
        self.assertEqual(len(shoe.unseen_cards), 52)
        shoe.reveal(hole)
        self.assertEqual(len(shoe.unseen_cards), 51)

    def test_hand_log_records_each_split_hand(self):
        shoe = stacked_shoe("8", "8", "10", "7", "Q", "K")
        log = []
        play_hand(shoe, 1000, "fixed", scripted("SPLIT", "STAND", "STAND"), hand_log=log)
        self.assertEqual([r for _, r in log], [10, 10])


class TestSearchAgent(unittest.TestCase):
    """The expectimax agent should reproduce well-known basic-strategy plays."""

    def setUp(self):
        from search.agent import ExpectimaxAgent
        from search.search import composition
        self.agent = ExpectimaxAgent(hit_soft_17=True)
        self.comp = composition(Shoe(6, seed=1).cards)

    def best(self, ranks, up, double=True, split=False):
        evs = self.agent.action_evs([Card(r) for r in ranks], Card(up), double, split, comp=self.comp)
        return max(evs, key=evs.get)

    def test_basic_strategy_plays(self):
        self.assertEqual(self.best(("10", "2"), "4"), "STAND")
        self.assertEqual(self.best(("10", "2"), "2"), "HIT")
        self.assertEqual(self.best(("6", "5"), "6"), "DOUBLE")
        self.assertEqual(self.best(("8", "8"), "10", split=True), "SPLIT")
        self.assertEqual(self.best(("10", "10"), "6", split=True), "STAND")
        self.assertEqual(self.best(("A", "7"), "9"), "HIT")

    def test_no_double_after_hit(self):
        self.assertNotIn("DOUBLE", self.agent.action_evs(
            [Card("2"), Card("3"), Card("6")], Card("6"), True, False, comp=self.comp))

    def test_plays_in_simulator(self):
        from search.agent import ExpectimaxAgent
        from simulation.run_simulation import run_session
        r = run_session(1, 20, betting="fixed", policy=ExpectimaxAgent())
        self.assertEqual(r["hands_played"], 20)


class TestMonteCarloAgent(unittest.TestCase):
    def test_learning_and_save_load(self):
        from machine_learning.monte_carlo import MonteCarloAgent, train
        agent, curve = train(2000, checkpoints=1, eval_hands=200, verbose=False)
        self.assertTrue(agent.q)
        self.assertEqual(len(curve), 1)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            agent.save(path)
            loaded = MonteCarloAgent.load(path)
            self.assertEqual(loaded.q.keys(), agent.q.keys())
        finally:
            os.remove(path)

    def test_split_hands_credited_separately(self):
        from machine_learning.monte_carlo import MonteCarloAgent
        agent = MonteCarloAgent(epsilon=1e-9)          # effectively greedy, but records visits
        agent.q = {}
        a, b = [Card("8"), Card("8")], [Card("8"), Card("K")]
        agent._visits = [(a, ("s0",), "SPLIT", False), (a, ("s1",), "STAND", True)]
        agent.learn([(a, -10), (b, 10)])
        self.assertEqual(agent.q[("s0",)]["SPLIT"], 0)      # both hands together
        self.assertEqual(agent.q[("s1",)]["STAND"], -10)    # its own hand only


@unittest.skipUnless(__import__("importlib").util.find_spec("sklearn"), "scikit-learn not installed")
class TestTreeAgent(unittest.TestCase):
    def setUp(self):
        base = {"decks": 6, "dealer_rule": "H17", "split_ev": ""}
        rows = [
            dict(base, category="hard", hand="11", dealer_upcard="6", stand_ev=-15, hit_ev=30, double_ev=60, best_action="D"),
            dict(base, category="hard", hand="16", dealer_upcard="10", stand_ev=-54, hit_ev=-53, double_ev=-106, best_action="H"),
            dict(base, category="hard", hand="13", dealer_upcard="4", stand_ev=-21, hit_ev=-25, double_ev=-50, best_action="S"),
        ]
        fd, self.path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"rows": rows}, f)

    def tearDown(self):
        os.remove(self.path)

    def test_predicts_and_respects_legality(self):
        from machine_learning.classifier import TreeAgent
        agent = TreeAgent(self.path)
        eleven = [Card("6"), Card("5")]
        self.assertEqual(agent(eleven, Card("6"), can_double=True), "DOUBLE")
        self.assertEqual(agent(eleven, Card("6"), can_double=False), "HIT")
        self.assertEqual(agent([Card("9"), Card("4")], Card("4")), "STAND")


if __name__ == "__main__":
    unittest.main()
