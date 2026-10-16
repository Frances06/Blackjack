"""Run repeatable sessions for the simplified Blackjack baseline."""
from statistics import mean
from simulation.blackjack import Shoe, hand_value
from simulation.strategy import choose_bet, choose_action


def play_hand(shoe, bankroll, betting="count", minimum=10, maximum=100):
    """Play one hand; return updated bankroll and hand profit/loss."""
    if betting == "fixed":
        bet = min(minimum, bankroll)
    elif betting == "count":
        bet = choose_bet(shoe.true_count, bankroll, minimum, maximum)
    else:
        raise ValueError("betting must be 'fixed' or 'count'")
    if bet <= 0:
        return bankroll, 0

    player = [shoe.draw(), shoe.draw()]
    upcard = shoe.draw()
    hole_card = shoe.draw(exposed=False)
    dealer = [upcard, hole_card]

    while choose_action(player, upcard) == "HIT":
        player.append(shoe.draw())
        if hand_value(player) > 21:
            # The dealer's hole card is revealed when the hand resolves.
            shoe.reveal(hole_card)
            return bankroll - bet, -bet

    shoe.reveal(hole_card)
    while hand_value(dealer) < 17:
        dealer.append(shoe.draw())

    player_total, dealer_total = hand_value(player), hand_value(dealer)
    if dealer_total > 21 or player_total > dealer_total:
        result = bet
    elif player_total < dealer_total:
        result = -bet
    else:
        result = 0
    return bankroll + result, result


def run_session(seed, hands=100, starting_bankroll=1000, betting="count"):
    shoe = Shoe(6, seed)
    bankroll = starting_bankroll
    played = 0
    for _ in range(hands):
        if bankroll < 10:
            break
        if shoe.cards_remaining < 78:
            shoe.reshuffle()
        bankroll, _ = play_hand(shoe, bankroll, betting=betting)
        played += 1
    return {
        "final_bankroll": bankroll,
        "profit": bankroll - starting_bankroll,
        "hands_played": played,
        "depleted": bankroll < 10,
    }


def main():
    sessions = [run_session(seed=i, betting="count") for i in range(100)]
    profits = [session["profit"] for session in sessions]
    print("=== PRELIMINARY COUNT-BETTING SIMULATION ===")
    print("100 sessions; up to 100 hands/session; starting bankroll 1,000 chips")
    print("6 decks; bets 10–100 chips; simplified hit/stand policy")
    print(f"Mean final bankroll: {mean(s['final_bankroll'] for s in sessions):.2f}")
    print(f"Mean profit/loss: {mean(profits):.2f} chips")
    print(f"Sessions ending ahead: {sum(p > 0 for p in profits)}/100")
    print(f"Sessions depleted: {sum(s['depleted'] for s in sessions)}/100")
    print("Exploratory results only; game rules and policy are simplified.")


if __name__ == "__main__":
    main()
