"""
Game abstraction for CFR solver.

Poker has too many states to solve exactly. Abstraction groups similar
situations together to make the problem tractable.

Two types of abstraction:
1. Hand Abstraction: Group hands by strength/potential
2. Action Abstraction: Limit bet sizes to discrete options
"""

from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.core.card import Card, Rank
from pokerbot.core.evaluator import HandEvaluator, HandRank


class HandBucket(IntEnum):
    """
    Hand strength buckets for abstraction.

    Reduces ~1326 starting hands to manageable categories.
    """
    # Preflop buckets (based on hand percentile)
    PREMIUM = 0       # Top 5%: AA, KK, QQ, AKs, AKo
    STRONG = 1        # 5-12%: JJ, TT, AQs, AQo, AJs, KQs
    GOOD = 2          # 12-25%: 99, 88, ATs, KJs, QJs, etc.
    PLAYABLE = 3      # 25-40%: 77, 66, suited connectors, etc.
    MARGINAL = 4      # 40-60%: Weak suited, small pairs
    TRASH = 5         # 60%+: Junk hands

    # Postflop buckets (based on made hand + draws)
    NUTS = 10         # Nuts or near-nuts (straights, flushes, sets)
    VERY_STRONG = 11  # Two pair+, overpairs
    TOP_PAIR = 12     # Top pair good kicker
    MEDIUM_PAIR = 13  # Middle pair, top pair weak kicker
    WEAK_MADE = 14    # Bottom pair, weak hands
    DRAW = 15         # Flush/straight draws
    WEAK_DRAW = 16    # Gutshots, backdoor draws
    AIR = 17          # No made hand, no draw


class ActionBucket(IntEnum):
    """
    Discrete bet sizing buckets.

    Instead of continuous bet sizes, use these discrete options.
    """
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_SMALL = 3     # 25-33% pot
    BET_MEDIUM = 4    # 50-67% pot
    BET_LARGE = 5     # 75-100% pot
    BET_OVERBET = 6   # 125%+ pot
    ALL_IN = 7


@dataclass
class ActionAbstraction:
    """
    Maps continuous bet sizes to discrete buckets.
    """
    # Pot percentage ranges for each bucket
    SIZING_RANGES = {
        ActionBucket.BET_SMALL: (0.20, 0.40),
        ActionBucket.BET_MEDIUM: (0.40, 0.70),
        ActionBucket.BET_LARGE: (0.70, 1.10),
        ActionBucket.BET_OVERBET: (1.10, 3.00),
    }

    # Default sizing for each bucket (as pot fraction)
    DEFAULT_SIZING = {
        ActionBucket.BET_SMALL: 0.33,
        ActionBucket.BET_MEDIUM: 0.67,
        ActionBucket.BET_LARGE: 0.80,
        ActionBucket.BET_OVERBET: 1.50,
    }

    @classmethod
    def get_bucket(cls, bet_size: float, pot: float) -> ActionBucket:
        """Map a bet size to its bucket."""
        if pot == 0:
            return ActionBucket.BET_MEDIUM

        ratio = bet_size / pot

        for bucket, (low, high) in cls.SIZING_RANGES.items():
            if low <= ratio < high:
                return bucket

        if ratio >= 3.0:
            return ActionBucket.ALL_IN
        return ActionBucket.BET_MEDIUM

    @classmethod
    def get_bet_size(cls, bucket: ActionBucket, pot: float, stack: float) -> float:
        """Get actual bet size from bucket."""
        if bucket == ActionBucket.ALL_IN:
            return stack
        if bucket == ActionBucket.FOLD:
            return 0
        if bucket == ActionBucket.CHECK:
            return 0
        if bucket == ActionBucket.CALL:
            return 0  # Call amount determined by game state

        sizing = cls.DEFAULT_SIZING.get(bucket, 0.67)
        bet = pot * sizing
        return min(bet, stack)

    @classmethod
    def get_available_buckets(cls, can_check: bool, can_bet: bool,
                               can_raise: bool, stack: float, pot: float) -> list[ActionBucket]:
        """Get available action buckets for current state."""
        buckets = []

        if can_check:
            buckets.append(ActionBucket.CHECK)
        else:
            buckets.append(ActionBucket.FOLD)
            buckets.append(ActionBucket.CALL)

        if can_bet or can_raise:
            # Add betting options based on stack
            buckets.append(ActionBucket.BET_SMALL)
            buckets.append(ActionBucket.BET_MEDIUM)
            buckets.append(ActionBucket.BET_LARGE)

            # Only add overbet if deep enough
            if stack > pot * 1.2:
                buckets.append(ActionBucket.BET_OVERBET)

            buckets.append(ActionBucket.ALL_IN)

        return buckets


@dataclass
class HandAbstraction:
    """
    Maps hands to buckets based on strength and potential.
    """

    def __post_init__(self):
        pass  # No initialization needed - uses static HandEvaluator

    def get_preflop_bucket(self, hand: Hand) -> HandBucket:
        """Get preflop hand bucket based on percentile."""
        from pokerbot.equity.range import get_hand_percentile

        percentile = get_hand_percentile(hand)

        if percentile <= 5:
            return HandBucket.PREMIUM
        elif percentile <= 12:
            return HandBucket.STRONG
        elif percentile <= 25:
            return HandBucket.GOOD
        elif percentile <= 40:
            return HandBucket.PLAYABLE
        elif percentile <= 60:
            return HandBucket.MARGINAL
        else:
            return HandBucket.TRASH

    def get_postflop_bucket(self, hand: Hand, board: list[Card]) -> HandBucket:
        """Get postflop hand bucket based on made hand and draws."""
        if len(board) < 3:
            return self.get_preflop_bucket(hand)

        # Evaluate made hand
        result = HandEvaluator.evaluate(hand, board)
        rank = result.rank

        # Check for strong made hands
        # Note: HandRank values are 1=best (royal flush) to 10=worst (high card)
        # So <= means "at least as good as"
        if rank <= HandRank.STRAIGHT:  # Straight or better
            return HandBucket.NUTS
        if rank == HandRank.THREE_OF_A_KIND:
            return HandBucket.VERY_STRONG
        if rank == HandRank.TWO_PAIR:
            return HandBucket.VERY_STRONG

        # Analyze pair situations
        if rank == HandRank.ONE_PAIR:
            pair_bucket = self._categorize_pair(hand, board)
            if pair_bucket:
                return pair_bucket

        # Check for draws
        draw_bucket = self._categorize_draws(hand, board)
        if draw_bucket:
            return draw_bucket

        # High card
        return HandBucket.AIR

    def _categorize_pair(self, hand: Hand, board: list[Card]) -> Optional[HandBucket]:
        """Categorize pair hands."""
        hand_ranks = [c.rank for c in hand.cards]
        board_ranks = sorted([c.rank for c in board], reverse=True)

        # Check for top pair
        if hand_ranks[0] == board_ranks[0] or hand_ranks[1] == board_ranks[0]:
            # Check kicker
            kicker = max(hand_ranks)
            if kicker.value >= Rank.TEN.value:
                return HandBucket.TOP_PAIR
            else:
                return HandBucket.MEDIUM_PAIR

        # Middle pair
        if len(board_ranks) >= 2:
            if hand_ranks[0] == board_ranks[1] or hand_ranks[1] == board_ranks[1]:
                return HandBucket.MEDIUM_PAIR

        # Pocket pair
        if hand_ranks[0] == hand_ranks[1]:
            if hand_ranks[0].value > board_ranks[0].value:
                return HandBucket.TOP_PAIR  # Overpair
            else:
                return HandBucket.MEDIUM_PAIR

        return HandBucket.WEAK_MADE

    def _categorize_draws(self, hand: Hand, board: list[Card]) -> Optional[HandBucket]:
        """Categorize drawing hands."""
        all_cards = list(hand.cards) + board

        # Check flush draw
        suits = {}
        for card in all_cards:
            suits[card.suit] = suits.get(card.suit, 0) + 1

        max_suited = max(suits.values())
        if max_suited >= 4:
            return HandBucket.DRAW

        # Check straight draw
        ranks = sorted(set([c.rank.value for c in all_cards]))

        # Open-ended straight draw
        for i in range(len(ranks) - 3):
            window = ranks[i:i+4]
            if window[-1] - window[0] == 3:
                return HandBucket.DRAW

        # Gutshot
        for i in range(len(ranks) - 3):
            window = ranks[i:i+4]
            if window[-1] - window[0] == 4:
                return HandBucket.WEAK_DRAW

        # Backdoor draws (3 to flush or straight)
        if max_suited >= 3:
            return HandBucket.WEAK_DRAW

        return None

    def get_bucket(self, hand: Hand, board: Optional[list[Card]] = None) -> HandBucket:
        """Get appropriate bucket for hand."""
        if board is None or len(board) == 0:
            return self.get_preflop_bucket(hand)
        else:
            return self.get_postflop_bucket(hand, board)


def create_info_set_key(
    hand_bucket: HandBucket,
    street: int,  # 0=preflop, 1=flop, 2=turn, 3=river
    action_history: tuple,
    pot_ratio: float,  # pot / starting_stack, bucketed
) -> str:
    """
    Create unique key for an information set.

    Information sets group together states that are indistinguishable
    to the player (same hand, same action history, similar pot size).
    """
    # Bucket pot ratio into categories
    if pot_ratio < 0.1:
        pot_bucket = "tiny"
    elif pot_ratio < 0.3:
        pot_bucket = "small"
    elif pot_ratio < 0.6:
        pot_bucket = "medium"
    elif pot_ratio < 1.0:
        pot_bucket = "large"
    else:
        pot_bucket = "huge"

    # Convert action history to string
    history_str = "_".join(str(a) for a in action_history)

    return f"{hand_bucket.name}|S{street}|{pot_bucket}|{history_str}"
