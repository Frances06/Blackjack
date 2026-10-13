import unittest
from simulation.blackjack import Card, Shoe, HI_LO, hand_value
from simulation.strategy import choose_bet

class TestFoundation(unittest.TestCase):
    def test_hi_lo(self):
        self.assertEqual(HI_LO["2"], 1)
        self.assertEqual(HI_LO["8"], 0)
        self.assertEqual(HI_LO["A"], -1)
    def test_ace_value(self):
        self.assertEqual(hand_value([Card("A"), Card("9")]), 20)
        self.assertEqual(hand_value([Card("A"), Card("9"), Card("5")]), 15)
    def test_bet_schedule(self):
        self.assertEqual(choose_bet(0, 1000), 10)
        self.assertEqual(choose_bet(2, 1000), 40)
        self.assertEqual(choose_bet(4, 1000), 100)
        self.assertEqual(choose_bet(4, 35), 35)
    def test_six_deck_shoe(self):
        self.assertEqual(Shoe(6, seed=1).cards_remaining, 312)
if __name__ == "__main__": unittest.main()
