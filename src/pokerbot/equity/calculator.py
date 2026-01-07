"""High-level equity calculator interface."""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from pokerbot.core.card import Card
from pokerbot.core.hand import Hand
from pokerbot.equity.range import HandRange
from pokerbot.equity.monte_carlo import (
    MonteCarloSimulator,
    EquityResult,
    MultiWayEquityResult,
    FastEquityCalculator,
)


@dataclass
class DetailedEquityResult:
    """Detailed equity result with additional analysis."""
    equity: EquityResult
    hand_breakdown: dict[str, float]  # Equity by hand category
    board_texture_impact: float  # How much board affects equity

    def __str__(self) -> str:
        return str(self.equity)


class EquityCalculator:
    """
    High-level interface for equity calculations.

    Provides convenient methods for common equity scenarios
    and automatic selection of calculation method based on accuracy needs.
    """

    def __init__(
        self,
        default_simulations: int = 10000,
        high_accuracy_simulations: int = 100000,
    ):
        """
        Initialize calculator.

        Args:
            default_simulations: Default number of simulations
            high_accuracy_simulations: Simulations for high accuracy mode
        """
        self.default_simulations = default_simulations
        self.high_accuracy_simulations = high_accuracy_simulations
        self._fast_calc = FastEquityCalculator()
        self._simulator = MonteCarloSimulator(num_simulations=default_simulations)

    def hand_vs_hand(
        self,
        hand1: Hand | str,
        hand2: Hand | str,
        board: Optional[list[Card] | str] = None,
        high_accuracy: bool = False,
    ) -> EquityResult:
        """
        Calculate equity of hand1 vs hand2.

        Args:
            hand1: First hand (e.g., "AsKs" or Hand object)
            hand2: Second hand
            board: Community cards (e.g., "Ks Qh 2d")
            high_accuracy: Use more simulations

        Returns:
            EquityResult with equity and statistics
        """
        # Parse hands
        if isinstance(hand1, str):
            hand1 = Hand(hand1)
        if isinstance(hand2, str):
            hand2 = Hand(hand2)

        # Parse board
        if isinstance(board, str):
            board = self._parse_board(board)

        # Run simulation
        num_sims = self.high_accuracy_simulations if high_accuracy else self.default_simulations
        simulator = MonteCarloSimulator(num_simulations=num_sims)

        return simulator.calculate_equity(hand1, hand2, board)

    def hand_vs_range(
        self,
        hand: Hand | str,
        range_str: str | HandRange,
        board: Optional[list[Card] | str] = None,
        high_accuracy: bool = False,
    ) -> EquityResult:
        """
        Calculate equity of a hand vs a range.

        Args:
            hand: Hero's hand
            range_str: Opponent's range (e.g., "TT+,AQs+,AKo")
            board: Community cards
            high_accuracy: Use more simulations

        Returns:
            EquityResult with equity
        """
        if isinstance(hand, str):
            hand = Hand(hand)
        if isinstance(range_str, str):
            opponent_range = HandRange(range_str)
        else:
            opponent_range = range_str

        if isinstance(board, str):
            board = self._parse_board(board)

        num_sims = self.high_accuracy_simulations if high_accuracy else self.default_simulations
        simulator = MonteCarloSimulator(num_simulations=num_sims)

        return simulator.calculate_equity(hand, opponent_range, board)

    def range_vs_range(
        self,
        range1: str | HandRange,
        range2: str | HandRange,
        board: Optional[list[Card] | str] = None,
        high_accuracy: bool = False,
    ) -> EquityResult:
        """
        Calculate equity of range1 vs range2.

        Args:
            range1: First player's range
            range2: Second player's range
            board: Community cards
            high_accuracy: Use more simulations

        Returns:
            EquityResult with equity
        """
        if isinstance(range1, str):
            range1 = HandRange(range1)
        if isinstance(range2, str):
            range2 = HandRange(range2)

        if isinstance(board, str):
            board = self._parse_board(board)

        num_sims = self.high_accuracy_simulations if high_accuracy else self.default_simulations
        simulator = MonteCarloSimulator(num_simulations=num_sims)

        return simulator.calculate_equity(range1, range2, board)

    def multiway(
        self,
        hands: list[Hand | str | HandRange],
        board: Optional[list[Card] | str] = None,
    ) -> MultiWayEquityResult:
        """
        Calculate equity for multiple players.

        Args:
            hands: List of hands or ranges for each player
            board: Community cards

        Returns:
            MultiWayEquityResult with equity for each player
        """
        parsed_hands = []
        for h in hands:
            if isinstance(h, str):
                if "," in h or "+" in h or "-" in h:
                    parsed_hands.append(HandRange(h))
                else:
                    parsed_hands.append(Hand(h))
            else:
                parsed_hands.append(h)

        if isinstance(board, str):
            board = self._parse_board(board)

        return self._simulator.calculate_multiway_equity(parsed_hands, board)

    def preflop_equity(self, hand: Hand | str) -> float:
        """
        Get fast approximate preflop equity vs random hand.

        Uses lookup tables for instant results.

        Args:
            hand: Hand to evaluate

        Returns:
            Approximate equity (0-1)
        """
        if isinstance(hand, str):
            hand = Hand(hand)
        return FastEquityCalculator.estimate_preflop_equity(hand)

    def required_equity_to_call(
        self,
        pot: float,
        to_call: float,
    ) -> float:
        """
        Calculate required equity to profitably call.

        Args:
            pot: Current pot size
            to_call: Amount to call

        Returns:
            Required equity (0-1)
        """
        if to_call <= 0:
            return 0.0
        return to_call / (pot + to_call)

    def pot_odds_percentage(
        self,
        pot: float,
        to_call: float,
    ) -> float:
        """
        Calculate pot odds as a percentage.

        Args:
            pot: Current pot size
            to_call: Amount to call

        Returns:
            Pot odds as percentage (0-100)
        """
        return self.required_equity_to_call(pot, to_call) * 100

    def ev_of_call(
        self,
        pot: float,
        to_call: float,
        equity: float,
    ) -> float:
        """
        Calculate expected value of calling.

        Args:
            pot: Current pot size
            to_call: Amount to call
            equity: Estimated equity (0-1)

        Returns:
            Expected value of calling
        """
        return equity * (pot + to_call) - to_call

    def ev_of_raise(
        self,
        pot: float,
        raise_to: float,
        fold_equity: float,
        call_equity: float,
    ) -> float:
        """
        Calculate expected value of raising.

        Args:
            pot: Current pot size
            raise_to: Total raise amount
            fold_equity: Probability opponent folds
            call_equity: Our equity when called

        Returns:
            Expected value of raising
        """
        # EV = fold_equity * pot + (1 - fold_equity) * (call_equity * new_pot - raise_to)
        ev_when_fold = fold_equity * pot
        new_pot = pot + raise_to + raise_to  # Assuming opponent calls
        ev_when_called = (1 - fold_equity) * (call_equity * new_pot - raise_to)
        return ev_when_fold + ev_when_called

    def _parse_board(self, board_str: str) -> list[Card]:
        """Parse board string like 'Ks Qh 2d'."""
        cards = []
        for card_str in board_str.split():
            if card_str:
                cards.append(Card(card_str))
        return cards


# Module-level convenience calculator
_default_calculator = EquityCalculator()


def equity(
    hand1: Hand | str | HandRange,
    hand2: Hand | str | HandRange,
    board: Optional[list[Card] | str] = None,
) -> EquityResult:
    """
    Convenience function to calculate equity.

    Args:
        hand1: First hand/range
        hand2: Second hand/range
        board: Community cards

    Returns:
        EquityResult
    """
    if isinstance(hand1, HandRange) or (isinstance(hand1, str) and ("," in hand1 or "+" in hand1)):
        if isinstance(hand2, HandRange) or (isinstance(hand2, str) and ("," in hand2 or "+" in hand2)):
            return _default_calculator.range_vs_range(hand1, hand2, board)
        return _default_calculator.hand_vs_range(hand2, hand1, board)
    elif isinstance(hand2, HandRange) or (isinstance(hand2, str) and ("," in hand2 or "+" in hand2)):
        return _default_calculator.hand_vs_range(hand1, hand2, board)
    return _default_calculator.hand_vs_hand(hand1, hand2, board)
