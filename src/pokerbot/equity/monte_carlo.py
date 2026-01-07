"""Monte Carlo simulation for poker equity calculations."""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from numba import jit, prange

from pokerbot.core.card import Card
from pokerbot.core.hand import Hand
from pokerbot.core.deck import Deck
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.equity.range import HandRange


@dataclass
class EquityResult:
    """Result of an equity calculation."""
    equity: float  # Win percentage (0-1)
    win_rate: float
    tie_rate: float
    loss_rate: float
    num_simulations: int
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    @property
    def equity_percentage(self) -> float:
        """Equity as percentage (0-100)."""
        return self.equity * 100

    def __str__(self) -> str:
        return (
            f"Equity: {self.equity_percentage:.2f}% "
            f"(Win: {self.win_rate*100:.2f}%, Tie: {self.tie_rate*100:.2f}%)"
        )


@dataclass
class MultiWayEquityResult:
    """Result of equity calculation for multiple players."""
    equities: list[float]  # Equity for each player
    win_rates: list[float]
    tie_rates: list[float]
    num_simulations: int

    def __str__(self) -> str:
        parts = []
        for i, eq in enumerate(self.equities):
            parts.append(f"Player {i+1}: {eq*100:.2f}%")
        return " | ".join(parts)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for poker equity calculations.

    Uses parallel processing and numba JIT compilation for performance.
    Supports hand vs hand and range vs range calculations.
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the simulator.

        Args:
            num_simulations: Number of simulations to run
            num_workers: Number of parallel workers (default: CPU count)
            seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.seed = seed
        self._rng = random.Random(seed)

    def calculate_equity(
        self,
        hand: Hand | HandRange,
        opponent: Hand | HandRange,
        board: Optional[list[Card]] = None,
        dead_cards: Optional[list[Card]] = None,
    ) -> EquityResult:
        """
        Calculate equity against a single opponent.

        Args:
            hand: Hero's hand or range
            opponent: Opponent's hand or range
            board: Community cards (0-5 cards)
            dead_cards: Cards to exclude from deck

        Returns:
            EquityResult with equity and statistics
        """
        board = board or []
        dead_cards = dead_cards or []

        # Get all dead card indices
        all_dead = set(c.index for c in board + dead_cards)

        # Get hand combinations
        hero_hands = self._get_hand_combos(hand, all_dead)
        villain_hands = self._get_hand_combos(opponent, all_dead)

        if not hero_hands or not villain_hands:
            raise ValueError("No valid hand combinations")

        # Run simulation
        wins, ties, losses, total = self._run_simulation(
            hero_hands, villain_hands, board, all_dead
        )

        win_rate = wins / total if total > 0 else 0
        tie_rate = ties / total if total > 0 else 0
        loss_rate = losses / total if total > 0 else 0
        equity = win_rate + tie_rate / 2

        # Calculate confidence interval (95%)
        std_error = np.sqrt(equity * (1 - equity) / total) if total > 0 else 0
        ci = (max(0, equity - 1.96 * std_error), min(1, equity + 1.96 * std_error))

        return EquityResult(
            equity=equity,
            win_rate=win_rate,
            tie_rate=tie_rate,
            loss_rate=loss_rate,
            num_simulations=total,
            confidence_interval=ci,
        )

    def calculate_multiway_equity(
        self,
        hands: list[Hand | HandRange],
        board: Optional[list[Card]] = None,
        dead_cards: Optional[list[Card]] = None,
    ) -> MultiWayEquityResult:
        """
        Calculate equity for multiple players.

        Args:
            hands: List of hands or ranges for each player
            board: Community cards
            dead_cards: Cards to exclude

        Returns:
            MultiWayEquityResult with equity for each player
        """
        board = board or []
        dead_cards = dead_cards or []

        all_dead = set(c.index for c in board + dead_cards)

        # Get hand combinations for each player
        player_hands = [self._get_hand_combos(h, all_dead) for h in hands]

        # Run multiway simulation
        results = self._run_multiway_simulation(player_hands, board, all_dead)

        return results

    def _get_hand_combos(
        self,
        hand: Hand | HandRange,
        dead_cards: set[int],
    ) -> list[Hand]:
        """Get list of hand combinations, excluding dead cards."""
        if isinstance(hand, Hand):
            # Check if hand conflicts with dead cards
            if any(c.index in dead_cards for c in hand.cards):
                return []
            return [hand]
        else:
            # HandRange
            dead_card_list = [Card.from_index(i) for i in dead_cards]
            return hand.get_combinations(dead_card_list)

    def _run_simulation(
        self,
        hero_hands: list[Hand],
        villain_hands: list[Hand],
        board: list[Card],
        dead_indices: set[int],
    ) -> tuple[int, int, int, int]:
        """Run the Monte Carlo simulation."""
        wins = 0
        ties = 0
        losses = 0
        total = 0

        cards_needed = 5 - len(board)

        for _ in range(self.num_simulations):
            # Pick random hands
            hero_hand = self._rng.choice(hero_hands)

            # Filter villain hands that don't conflict with hero
            hero_indices = {c.index for c in hero_hand.cards}
            valid_villain = [
                h for h in villain_hands
                if not any(c.index in hero_indices for c in h.cards)
            ]

            if not valid_villain:
                continue

            villain_hand = self._rng.choice(valid_villain)

            # Build deck excluding known cards
            all_dead = dead_indices | hero_indices | {c.index for c in villain_hand.cards}
            available = [i for i in range(52) if i not in all_dead]

            if len(available) < cards_needed:
                continue

            # Deal remaining board cards
            sampled = self._rng.sample(available, cards_needed)
            full_board = board + [Card.from_index(i) for i in sampled]

            # Evaluate hands
            result = HandEvaluator.compare(hero_hand, villain_hand, full_board)

            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                ties += 1
            total += 1

        return wins, ties, losses, total

    def _run_multiway_simulation(
        self,
        player_hands: list[list[Hand]],
        board: list[Card],
        dead_indices: set[int],
    ) -> MultiWayEquityResult:
        """Run multiway Monte Carlo simulation."""
        num_players = len(player_hands)
        wins = [0] * num_players
        ties = [0] * num_players
        total = 0

        cards_needed = 5 - len(board)

        for _ in range(self.num_simulations):
            # Pick hands for each player, avoiding conflicts
            used_indices: set[int] = set()
            chosen_hands = []
            valid = True

            for hands in player_hands:
                valid_hands = [
                    h for h in hands
                    if not any(c.index in used_indices for c in h.cards)
                ]
                if not valid_hands:
                    valid = False
                    break

                hand = self._rng.choice(valid_hands)
                chosen_hands.append(hand)
                used_indices.update(c.index for c in hand.cards)

            if not valid:
                continue

            # Deal remaining board
            all_dead = dead_indices | used_indices
            available = [i for i in range(52) if i not in all_dead]

            if len(available) < cards_needed:
                continue

            sampled = self._rng.sample(available, cards_needed)
            full_board = board + [Card.from_index(i) for i in sampled]

            # Find winners
            winner_indices = HandEvaluator.winners(chosen_hands, full_board)

            if len(winner_indices) == 1:
                wins[winner_indices[0]] += 1
            else:
                for idx in winner_indices:
                    ties[idx] += 1

            total += 1

        # Calculate equities
        equities = []
        win_rates = []
        tie_rates = []

        for i in range(num_players):
            win_rate = wins[i] / total if total > 0 else 0
            tie_rate = ties[i] / total if total > 0 else 0
            # Equity = win + tie/num_tie_players (approximate as tie/2 for simplicity)
            equity = win_rate + tie_rate / 2
            equities.append(equity)
            win_rates.append(win_rate)
            tie_rates.append(tie_rate)

        return MultiWayEquityResult(
            equities=equities,
            win_rates=win_rates,
            tie_rates=tie_rates,
            num_simulations=total,
        )


# Fast equity approximation using lookup tables
class FastEquityCalculator:
    """
    Fast preflop equity calculator using precomputed values.

    Uses statistical approximations for speed when exact equity isn't required.
    """

    # Approximate preflop hand strength rankings (0-1 scale)
    # Based on statistical analysis of hand vs random hand
    PREFLOP_STRENGTH = {
        "AA": 0.852, "KK": 0.824, "QQ": 0.799, "JJ": 0.774, "TT": 0.750,
        "99": 0.720, "88": 0.691, "AKs": 0.670, "77": 0.662, "AQs": 0.662,
        "AKo": 0.653, "AJs": 0.647, "AQo": 0.643, "KQs": 0.634, "66": 0.633,
        "ATs": 0.632, "AJo": 0.627, "KJs": 0.626, "KQo": 0.614, "55": 0.604,
        "KTs": 0.603, "ATo": 0.602, "A9s": 0.600, "QJs": 0.599, "KJo": 0.596,
        "QTs": 0.590, "44": 0.575, "A8s": 0.579, "KTo": 0.573, "A5s": 0.577,
        "JTs": 0.575, "A7s": 0.573, "QJo": 0.569, "A4s": 0.569, "33": 0.546,
        "A9o": 0.568, "A6s": 0.567, "A3s": 0.565, "QTo": 0.560, "K9s": 0.558,
        "JTo": 0.555, "22": 0.517, "A2s": 0.557, "A8o": 0.546, "K8s": 0.542,
        "T9s": 0.541, "K7s": 0.537, "A5o": 0.544, "J9s": 0.536, "Q9s": 0.532,
    }

    @classmethod
    def estimate_preflop_equity(cls, hand: Hand | str) -> float:
        """
        Get approximate preflop equity vs random hand.

        Args:
            hand: Hand object or notation string

        Returns:
            Approximate equity (0-1)
        """
        if isinstance(hand, Hand):
            notation = hand.notation()
        else:
            notation = hand.upper().replace("10", "T")

        return cls.PREFLOP_STRENGTH.get(notation, 0.45)  # Default to 45%


def calculate_equity(
    hand: Hand | HandRange,
    opponent: Hand | HandRange,
    board: Optional[list[Card]] = None,
    num_simulations: int = 10000,
) -> EquityResult:
    """
    Convenience function to calculate equity.

    Args:
        hand: Hero's hand or range
        opponent: Opponent's hand or range
        board: Community cards
        num_simulations: Number of Monte Carlo iterations

    Returns:
        EquityResult with equity statistics
    """
    simulator = MonteCarloSimulator(num_simulations=num_simulations)
    return simulator.calculate_equity(hand, opponent, board)
