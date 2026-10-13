"""Run repeatable preliminary sessions for the simplified baseline."""
from statistics import mean
from simulation.blackjack import Shoe, hand_value
from simulation.strategy import choose_bet, choose_action

def play_hand(shoe, bankroll, minimum=10, maximum=100):
    bet = choose_bet(shoe.true_count, bankroll, minimum, maximum)
    if bet <= 0: return bankroll, 0
    player = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]
    upcard = dealer[0]
    while choose_action(player, upcard) == "HIT":
        player.append(shoe.draw())
        if hand_value(player) > 21:
            return bankroll - bet, -bet
    while hand_value(dealer) < 17:
        dealer.append(shoe.draw())
    p, d = hand_value(player), hand_value(dealer)
    if d > 21 or p > d: result = bet
    elif p < d: result = -bet
    else: result = 0
    return bankroll + result, result

def run_session(seed, hands=100, starting_bankroll=1000):
    shoe = Shoe(6, seed)
    bankroll = starting_bankroll
    played = 0
    for _ in range(hands):
        if bankroll < 10: break
        if shoe.cards_remaining < 78: shoe.reshuffle()
        bankroll, _ = play_hand(shoe, bankroll)
        played += 1
    return {"final_bankroll": bankroll, "profit": bankroll-starting_bankroll,
            "hands_played": played, "depleted": bankroll < 10}

def main():
    sessions = [run_session(i) for i in range(100)]
    profits = [s["profit"] for s in sessions]
    print("=== PRELIMINARY SIMULATION ===")
    print("100 sessions; up to 100 hands/session; starting bankroll 1,000 chips")
    print("6 decks; bets 10–100 chips; simplified hit/stand policy")
    print(f"Mean final bankroll: {mean(s['final_bankroll'] for s in sessions):.2f}")
    print(f"Mean profit/loss: {mean(profits):.2f} chips")
    print(f"Sessions ending ahead: {sum(p > 0 for p in profits)}/100")
    print(f"Sessions depleted: {sum(s['depleted'] for s in sessions)}/100")
    print("Preliminary results only: rules and policy are simplified.")
if __name__ == "__main__": main()
