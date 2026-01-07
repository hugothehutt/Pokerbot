"""
Kuhn Poker CFR - A simple 3-card poker variant for testing CFR.

Kuhn Poker is the simplest interesting poker game:
- 3 cards: Jack (0), Queen (1), King (2)
- 2 players, each gets 1 card
- 1 chip ante from each player
- Only actions: check (p), bet 1 (b)
- After bet: fold (p) or call (b)
- Higher card wins at showdown

Known Nash Equilibrium (for reference):
- Player 1 with Jack: bet with probability α ∈ [0, 1/3]
- Player 1 with King: bet with probability 3α
- Player 2 with Queen facing bet: call with probability α + 1/3
- Player 2 with Jack: always fold to bet
- Player 2 with King: always call bet
- The game value is -1/18 ≈ -0.056 for player 1

This is used to verify CFR works before scaling to full poker.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import random


PASS = 0
BET = 1
NUM_ACTIONS = 2


@dataclass
class KuhnInfoSet:
    """Information set data for Kuhn Poker."""
    regret_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def get_strategy(self, realization_weight: float) -> List[float]:
        """Get current strategy via regret matching."""
        strategy = [0.0, 0.0]
        normalizing_sum = 0.0

        for a in range(NUM_ACTIONS):
            strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += strategy[a]

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / NUM_ACTIONS
            self.strategy_sum[a] += realization_weight * strategy[a]

        return strategy

    def get_average_strategy(self) -> List[float]:
        """Get average strategy (Nash equilibrium approximation)."""
        avg_strategy = [0.0, 0.0]
        normalizing_sum = sum(self.strategy_sum)

        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / NUM_ACTIONS

        return avg_strategy


class KuhnCFR:
    """
    CFR solver for Kuhn Poker.

    This is a complete, working CFR implementation that converges
    to Nash equilibrium in ~1000 iterations.
    """

    CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}

    def __init__(self):
        self.info_sets: Dict[str, KuhnInfoSet] = {}
        self._rng = random.Random()

    def get_info_set(self, key: str) -> KuhnInfoSet:
        """Get or create info set."""
        if key not in self.info_sets:
            self.info_sets[key] = KuhnInfoSet()
        return self.info_sets[key]

    def train(self, iterations: int = 10000) -> float:
        """Train via CFR iterations."""
        cards = [0, 1, 2]
        util = 0.0

        for i in range(iterations):
            # Shuffle cards for this iteration
            self._rng.shuffle(cards)

            # Run CFR
            util += self._cfr(cards, "", 1.0, 1.0)

            if (i + 1) % 1000 == 0:
                print(f"Iteration {i+1}: avg game value = {util / (i+1):.4f}")

        return util / iterations

    def _cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        """
        CFR recursive traversal.

        Args:
            cards: [player0_card, player1_card, unused_card]
            history: Action history string (e.g., "", "p", "pb")
            p0: Player 0 reach probability
            p1: Player 1 reach probability

        Returns:
            Expected value for player 0
        """
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # Check for terminal states
        if plays > 1:
            terminal_pass = history[-1] == 'p'
            double_bet = history[-2:] == "bb"
            is_player_card_higher = cards[player] > cards[opponent]

            if terminal_pass:
                if history == "pp":
                    # Showdown after check-check
                    return 1 if is_player_card_higher else -1
                else:
                    # Fold - Loss of 1 for folder
                    return 1

            elif double_bet:
                # Showdown after bet-call
                return 2 if is_player_card_higher else -2

        # Non-terminal node
        info_key = str(cards[player]) + history
        info_set = self.get_info_set(info_key)

        # Get current strategy with realization weight
        realization_weight = p0 if player == 0 else p1
        strategy = info_set.get_strategy(realization_weight)

        # Calculate utility for each action
        util = [0.0, 0.0]
        node_util = 0.0

        for a in range(NUM_ACTIONS):
            next_history = history + ('p' if a == PASS else 'b')

            if player == 0:
                util[a] = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self._cfr(cards, next_history, p0, p1 * strategy[a])

            node_util += strategy[a] * util[a]

        # Compute regrets and add to cumulative regrets
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            if player == 0:
                info_set.regret_sum[a] += p1 * regret
            else:
                info_set.regret_sum[a] += p0 * regret

        return node_util

    def get_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get final Nash equilibrium strategy."""
        result = {}
        for key, info_set in self.info_sets.items():
            avg = info_set.get_average_strategy()
            result[key] = {'p': avg[PASS], 'b': avg[BET]}
        return result

    def print_strategy(self):
        """Print the learned strategy."""
        print("\n=== Kuhn Poker Nash Equilibrium ===")
        print("(p=pass/fold, b=bet/call)\n")

        strategy = self.get_strategy()

        # Map internal keys to readable format
        for card_idx, card_name in self.CARD_NAMES.items():
            print(f"Card: {card_name}")
            for key, probs in sorted(strategy.items()):
                if key.startswith(str(card_idx)):
                    history = key[1:] if len(key) > 1 else "(initial)"
                    # Describe the situation
                    if history == "(initial)":
                        situation = "First to act"
                    elif history == "p":
                        situation = "Checked to"
                    elif history == "b":
                        situation = "Facing bet"
                    elif history == "pb":
                        situation = "Facing bet after check"
                    else:
                        situation = f"History: {history}"
                    print(f"  {situation}: pass/fold={probs['p']:.1%}, bet/call={probs['b']:.1%}")
            print()


def train_kuhn_poker(iterations: int = 10000) -> KuhnCFR:
    """Train Kuhn Poker and return the solver."""
    print("Training Kuhn Poker CFR...")
    print("Expected Nash equilibrium game value: -0.056 (for player 1)\n")

    solver = KuhnCFR()
    avg_value = solver.train(iterations)
    solver.print_strategy()

    print(f"Final average game value: {avg_value:.4f}")
    print("(Should be close to -0.056)")

    return solver


if __name__ == "__main__":
    train_kuhn_poker(10000)
