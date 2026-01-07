"""Hand evaluation for Texas Hold'em poker using the treys library."""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from treys import Evaluator as TreysEvaluator
from treys import Card as TreysCard

if TYPE_CHECKING:
    from pokerbot.core.card import Card
    from pokerbot.core.hand import Hand


class HandRank(IntEnum):
    """
    Poker hand rankings from highest to lowest.

    Note: Lower values are BETTER (1 = best possible hand).
    """
    ROYAL_FLUSH = 1
    STRAIGHT_FLUSH = 2
    FOUR_OF_A_KIND = 3
    FULL_HOUSE = 4
    FLUSH = 5
    STRAIGHT = 6
    THREE_OF_A_KIND = 7
    TWO_PAIR = 8
    ONE_PAIR = 9
    HIGH_CARD = 10

    @classmethod
    def from_treys_class(cls, treys_class: int) -> HandRank:
        """Convert treys hand class to HandRank."""
        # Treys uses 1-9 where 1 is straight flush (includes royal)
        # and 9 is high card
        mapping = {
            1: cls.STRAIGHT_FLUSH,  # Includes royal flush
            2: cls.FOUR_OF_A_KIND,
            3: cls.FULL_HOUSE,
            4: cls.FLUSH,
            5: cls.STRAIGHT,
            6: cls.THREE_OF_A_KIND,
            7: cls.TWO_PAIR,
            8: cls.ONE_PAIR,
            9: cls.HIGH_CARD,
        }
        return mapping.get(treys_class, cls.HIGH_CARD)


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a poker hand."""

    score: int  # Lower is better (1-7462)
    rank: HandRank
    description: str

    @property
    def percentile(self) -> float:
        """
        Return the percentile of this hand (0-100).

        100 = best possible hand (royal flush)
        0 = worst possible hand (7-high)
        """
        # Treys scores range from 1 (best) to 7462 (worst)
        return (1 - (self.score - 1) / 7461) * 100

    def __lt__(self, other: EvaluationResult) -> bool:
        # Lower score is better
        return self.score < other.score

    def __gt__(self, other: EvaluationResult) -> bool:
        return self.score > other.score

    def beats(self, other: EvaluationResult) -> bool:
        """Check if this hand beats another."""
        return self.score < other.score

    def ties(self, other: EvaluationResult) -> bool:
        """Check if this hand ties another."""
        return self.score == other.score


class HandEvaluator:
    """
    Evaluates poker hands using the treys library.

    The treys library uses efficient lookup tables for fast evaluation.
    This class provides a convenient wrapper with our card types.
    """

    # Singleton evaluator instance (treys evaluator is stateless)
    _evaluator = TreysEvaluator()

    @classmethod
    def evaluate(cls, hole_cards: Hand | list[Card], board: list[Card]) -> EvaluationResult:
        """
        Evaluate a 5-7 card poker hand.

        Args:
            hole_cards: Player's hole cards (Hand or list of Cards)
            board: Community cards (3-5 cards)

        Returns:
            EvaluationResult with score, rank, and description
        """
        from pokerbot.core.hand import Hand

        # Convert to treys format
        if isinstance(hole_cards, Hand):
            treys_hole = hole_cards.to_treys()
        else:
            treys_hole = [c.to_treys() for c in hole_cards]

        treys_board = [c.to_treys() for c in board]

        # Evaluate
        score = cls._evaluator.evaluate(treys_board, treys_hole)
        rank_class = cls._evaluator.get_rank_class(score)
        rank = HandRank.from_treys_class(rank_class)
        description = cls._evaluator.class_to_string(rank_class)

        # Check for royal flush (straight flush with A-high)
        if rank == HandRank.STRAIGHT_FLUSH and score == 1:
            rank = HandRank.ROYAL_FLUSH
            description = "Royal Flush"

        return EvaluationResult(score=score, rank=rank, description=description)

    @classmethod
    def evaluate_board(cls, board: list[Card]) -> EvaluationResult:
        """
        Evaluate just the board (useful for analyzing board texture).

        Args:
            board: 5 community cards

        Returns:
            EvaluationResult for the board alone
        """
        if len(board) != 5:
            raise ValueError("Board must have exactly 5 cards")

        treys_board = [c.to_treys() for c in board]

        # Evaluate board as a 5-card hand
        score = cls._evaluator.evaluate(treys_board, [])
        rank_class = cls._evaluator.get_rank_class(score)
        rank = HandRank.from_treys_class(rank_class)
        description = cls._evaluator.class_to_string(rank_class)

        return EvaluationResult(score=score, rank=rank, description=description)

    @classmethod
    def compare(
        cls,
        hand1: Hand | list[Card],
        hand2: Hand | list[Card],
        board: list[Card],
    ) -> int:
        """
        Compare two hands on the same board.

        Args:
            hand1: First player's hole cards
            hand2: Second player's hole cards
            board: Community cards

        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        result1 = cls.evaluate(hand1, board)
        result2 = cls.evaluate(hand2, board)

        if result1.beats(result2):
            return 1
        elif result2.beats(result1):
            return -1
        else:
            return 0

    @classmethod
    def winners(
        cls,
        hands: list[Hand | list[Card]],
        board: list[Card],
    ) -> list[int]:
        """
        Determine the winner(s) among multiple hands.

        Args:
            hands: List of hole cards for each player
            board: Community cards

        Returns:
            List of indices of winning players (multiple if tie)
        """
        if not hands:
            return []

        results = [cls.evaluate(hand, board) for hand in hands]
        best_score = min(r.score for r in results)

        return [i for i, r in enumerate(results) if r.score == best_score]


# Convenience functions
def evaluate(hole_cards: Hand | list[Card], board: list[Card]) -> EvaluationResult:
    """Evaluate a poker hand. Shortcut for HandEvaluator.evaluate()."""
    return HandEvaluator.evaluate(hole_cards, board)


def compare(
    hand1: Hand | list[Card],
    hand2: Hand | list[Card],
    board: list[Card],
) -> int:
    """Compare two hands. Shortcut for HandEvaluator.compare()."""
    return HandEvaluator.compare(hand1, hand2, board)
