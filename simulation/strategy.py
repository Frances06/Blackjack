"""Transparent baseline betting and playing rules."""
from simulation.blackjack import hand_value

def choose_bet(true_count, bankroll, minimum=10, maximum=100):
    if bankroll <= 0: return 0
    if true_count >= 4: proposed = 100
    elif true_count >= 3: proposed = 60
    elif true_count >= 2: proposed = 40
    elif true_count >= 1: proposed = 20
    else: proposed = minimum
    return min(maximum, bankroll, max(minimum, proposed))

def choose_action(player_cards, dealer_upcard):
    """Simplified hit/stand baseline; not full basic strategy."""
    total = hand_value(player_cards)
    if total <= 11: return "HIT"
    if total >= 17: return "STAND"
    if total >= 13 and dealer_upcard.value <= 6: return "STAND"
    return "HIT"
