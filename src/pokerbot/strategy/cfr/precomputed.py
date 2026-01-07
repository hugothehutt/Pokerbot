"""
Precomputed GTO-approximated strategies.

Real GTO solvers pre-compute strategies offline and store them for lookup.
This module provides approximated strategies based on solver outputs and
game theory principles.

These frequencies are derived from:
1. Published solver studies
2. GTO theory (MDF, optimal bluff frequencies)
3. Hand strength categorization
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import random

from pokerbot.strategy.cfr.abstraction import HandBucket, ActionBucket


@dataclass
class ActionFrequencies:
    """Action frequencies for a spot."""
    check: float = 0.0
    bet_small: float = 0.0
    bet_medium: float = 0.0
    bet_large: float = 0.0
    bet_overbet: float = 0.0
    fold: float = 0.0
    call: float = 0.0
    raise_small: float = 0.0
    raise_large: float = 0.0
    all_in: float = 0.0

    def sample_action(self, rng: random.Random = None) -> ActionBucket:
        """Sample an action according to frequencies."""
        if rng is None:
            rng = random.Random()

        actions = [
            (ActionBucket.CHECK, self.check),
            (ActionBucket.BET_SMALL, self.bet_small),
            (ActionBucket.BET_MEDIUM, self.bet_medium),
            (ActionBucket.BET_LARGE, self.bet_large),
            (ActionBucket.BET_OVERBET, self.bet_overbet),
            (ActionBucket.FOLD, self.fold),
            (ActionBucket.CALL, self.call),
            (ActionBucket.BET_SMALL, self.raise_small),  # Reuse bet buckets
            (ActionBucket.BET_LARGE, self.raise_large),
            (ActionBucket.ALL_IN, self.all_in),
        ]

        total = sum(freq for _, freq in actions)
        if total == 0:
            return ActionBucket.CHECK

        r = rng.random() * total
        cumulative = 0
        for action, freq in actions:
            cumulative += freq
            if r <= cumulative:
                return action

        return actions[-1][0]


class PrecomputedStrategy:
    """
    Precomputed GTO-approximated strategies.

    Strategies are organized by:
    - Street (preflop, flop, turn, river)
    - Position type (in position, out of position)
    - Facing action (checked to, facing bet)
    - Hand bucket
    """

    def __init__(self):
        self._rng = random.Random()

        # Build strategy tables
        self._build_preflop_strategies()
        self._build_postflop_strategies()

    def _build_preflop_strategies(self):
        """Build preflop strategy tables."""
        # Opening ranges by hand bucket (when first to act)
        self.preflop_open = {
            HandBucket.PREMIUM: ActionFrequencies(
                bet_large=0.85,  # 3x open
                bet_overbet=0.10,  # 4x open
                all_in=0.05,  # Occasional limp-raise
            ),
            HandBucket.STRONG: ActionFrequencies(
                bet_medium=0.30,
                bet_large=0.65,
                check=0.05,  # Occasional trap
            ),
            HandBucket.GOOD: ActionFrequencies(
                bet_medium=0.50,
                bet_large=0.40,
                check=0.10,
            ),
            HandBucket.PLAYABLE: ActionFrequencies(
                bet_medium=0.40,
                bet_large=0.30,
                check=0.30,
            ),
            HandBucket.MARGINAL: ActionFrequencies(
                check=0.70,
                bet_medium=0.20,
                fold=0.10,
            ),
            HandBucket.TRASH: ActionFrequencies(
                fold=0.85,
                check=0.10,
                bet_medium=0.05,  # Occasional bluff
            ),
        }

        # Facing open raise
        self.preflop_vs_open = {
            HandBucket.PREMIUM: ActionFrequencies(
                raise_large=0.80,  # 3-bet
                call=0.15,  # Trap
                all_in=0.05,  # 4-bet jam
            ),
            HandBucket.STRONG: ActionFrequencies(
                raise_large=0.45,
                call=0.50,
                fold=0.05,
            ),
            HandBucket.GOOD: ActionFrequencies(
                call=0.60,
                raise_small=0.25,
                fold=0.15,
            ),
            HandBucket.PLAYABLE: ActionFrequencies(
                call=0.45,
                fold=0.45,
                raise_small=0.10,  # Light 3-bet
            ),
            HandBucket.MARGINAL: ActionFrequencies(
                fold=0.70,
                call=0.25,
                raise_small=0.05,
            ),
            HandBucket.TRASH: ActionFrequencies(
                fold=0.90,
                call=0.05,
                raise_small=0.05,  # Bluff 3-bet
            ),
        }

    def _build_postflop_strategies(self):
        """Build postflop strategy tables."""
        # When checked to (can bet or check)
        self.postflop_checked_to = {
            # Strong made hands - bet for value
            HandBucket.NUTS: ActionFrequencies(
                bet_large=0.50,
                bet_overbet=0.30,
                check=0.15,  # Trap
                all_in=0.05,
            ),
            HandBucket.VERY_STRONG: ActionFrequencies(
                bet_large=0.55,
                bet_medium=0.25,
                check=0.15,
                bet_overbet=0.05,
            ),
            HandBucket.TOP_PAIR: ActionFrequencies(
                bet_medium=0.50,
                bet_small=0.20,
                check=0.25,
                bet_large=0.05,
            ),
            HandBucket.MEDIUM_PAIR: ActionFrequencies(
                check=0.55,
                bet_small=0.30,
                bet_medium=0.15,
            ),
            HandBucket.WEAK_MADE: ActionFrequencies(
                check=0.70,
                bet_small=0.20,
                bet_medium=0.10,
            ),
            # Draws - semi-bluff
            HandBucket.DRAW: ActionFrequencies(
                bet_medium=0.45,
                bet_large=0.20,
                check=0.35,
            ),
            HandBucket.WEAK_DRAW: ActionFrequencies(
                check=0.60,
                bet_small=0.25,
                bet_medium=0.15,
            ),
            # Air - balanced bluffs
            HandBucket.AIR: ActionFrequencies(
                check=0.65,
                bet_medium=0.20,
                bet_large=0.10,
                bet_small=0.05,
            ),
        }

        # When facing a bet (can fold, call, raise)
        self.postflop_vs_bet = {
            HandBucket.NUTS: ActionFrequencies(
                raise_large=0.60,
                call=0.30,  # Slow play
                all_in=0.10,
            ),
            HandBucket.VERY_STRONG: ActionFrequencies(
                call=0.50,
                raise_small=0.35,
                raise_large=0.10,
                fold=0.05,
            ),
            HandBucket.TOP_PAIR: ActionFrequencies(
                call=0.65,
                raise_small=0.15,
                fold=0.20,
            ),
            HandBucket.MEDIUM_PAIR: ActionFrequencies(
                call=0.50,
                fold=0.45,
                raise_small=0.05,
            ),
            HandBucket.WEAK_MADE: ActionFrequencies(
                fold=0.60,
                call=0.35,
                raise_small=0.05,  # Bluff
            ),
            HandBucket.DRAW: ActionFrequencies(
                call=0.55,
                raise_small=0.20,  # Semi-bluff
                fold=0.25,
            ),
            HandBucket.WEAK_DRAW: ActionFrequencies(
                fold=0.50,
                call=0.40,
                raise_small=0.10,
            ),
            HandBucket.AIR: ActionFrequencies(
                fold=0.70,
                call=0.15,  # Float
                raise_small=0.10,  # Bluff raise
                raise_large=0.05,
            ),
        }

    def get_preflop_action(
        self,
        hand_bucket: HandBucket,
        facing_raise: bool = False,
    ) -> ActionBucket:
        """Get preflop action."""
        if facing_raise:
            table = self.preflop_vs_open
        else:
            table = self.preflop_open

        # Handle postflop buckets by mapping to preflop
        if hand_bucket.value >= 10:
            # Map postflop bucket to preflop equivalent
            hand_bucket = HandBucket.PLAYABLE

        freqs = table.get(hand_bucket, ActionFrequencies(fold=1.0))
        return freqs.sample_action(self._rng)

    def get_postflop_action(
        self,
        hand_bucket: HandBucket,
        facing_bet: bool = False,
    ) -> ActionBucket:
        """Get postflop action."""
        if facing_bet:
            table = self.postflop_vs_bet
        else:
            table = self.postflop_checked_to

        freqs = table.get(hand_bucket, ActionFrequencies(check=1.0))
        return freqs.sample_action(self._rng)

    def get_action(
        self,
        hand_bucket: HandBucket,
        street: int,  # 0=preflop, 1+=postflop
        facing_bet: bool = False,
    ) -> ActionBucket:
        """Get action for any street."""
        if street == 0:
            return self.get_preflop_action(hand_bucket, facing_raise=facing_bet)
        else:
            return self.get_postflop_action(hand_bucket, facing_bet=facing_bet)

    def get_frequencies(
        self,
        hand_bucket: HandBucket,
        street: int,
        facing_bet: bool = False,
    ) -> ActionFrequencies:
        """Get full frequency table for a spot."""
        if street == 0:
            if facing_bet:
                table = self.preflop_vs_open
            else:
                table = self.preflop_open
            if hand_bucket.value >= 10:
                hand_bucket = HandBucket.PLAYABLE
        else:
            if facing_bet:
                table = self.postflop_vs_bet
            else:
                table = self.postflop_checked_to

        return table.get(hand_bucket, ActionFrequencies(check=1.0))


# Global singleton for easy access
_strategy = None


def get_precomputed_strategy() -> PrecomputedStrategy:
    """Get the global precomputed strategy instance."""
    global _strategy
    if _strategy is None:
        _strategy = PrecomputedStrategy()
    return _strategy
