"""GTO-inspired bet sizing strategies."""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from pokerbot.game.state import GameState, Street
from pokerbot.core.evaluator import HandEvaluator, HandRank


class SizingStrategy(Enum):
    """Common bet sizing strategies."""
    SMALL = auto()      # 25-33% pot
    MEDIUM = auto()     # 50-67% pot
    LARGE = auto()      # 75-100% pot
    OVERBET = auto()    # 125%+ pot
    GEOMETRIC = auto()  # Size to get all-in over multiple streets
    POLARIZED = auto()  # Large with value/bluffs, small with medium hands


@dataclass
class BetSize:
    """Represents a bet size with context."""
    amount: float
    pot_percentage: float
    strategy: SizingStrategy
    reasoning: str


class BetSizer:
    """
    GTO-inspired bet sizing calculator.

    Determines optimal bet sizes based on:
    - Stack-to-pot ratio (SPR)
    - Board texture
    - Position
    - Street
    - Hand strength category
    """

    # Standard sizing percentages
    SIZING = {
        SizingStrategy.SMALL: (0.25, 0.33),
        SizingStrategy.MEDIUM: (0.50, 0.67),
        SizingStrategy.LARGE: (0.75, 1.00),
        SizingStrategy.OVERBET: (1.25, 2.00),
    }

    def __init__(self, game_state: GameState):
        """Initialize with game state."""
        self.state = game_state

    def get_bet_size(
        self,
        strategy: SizingStrategy,
        use_max: bool = False,
    ) -> float:
        """
        Get bet size for a strategy.

        Args:
            strategy: Sizing strategy to use
            use_max: Use maximum of range (default: minimum)

        Returns:
            Bet amount
        """
        pot = self.state.pot.total
        index = 1 if use_max else 0

        if strategy in self.SIZING:
            pct = self.SIZING[strategy][index]
            return pot * pct

        return pot * 0.67  # Default to medium

    def get_geometric_size(self, streets_remaining: int) -> float:
        """
        Calculate geometric sizing to get all-in.

        Sizes bets so that betting the same pot fraction each street
        results in being all-in on the river.

        Args:
            streets_remaining: Number of streets left (1-3)

        Returns:
            Bet amount for this street
        """
        if streets_remaining <= 0:
            return self.state.pot.total

        player = self.state.players[self.state.action_on]
        stack = player.stack
        pot = self.state.pot.total

        # Calculate geometric multiplier
        # After n bets of size x*pot, stack should be 0
        # pot * (1 + 2x)^n = stack + pot
        # Solve for x
        target_pot = stack + pot
        ratio = target_pot / pot

        # (1 + 2x)^n = ratio
        # 1 + 2x = ratio^(1/n)
        growth = ratio ** (1 / streets_remaining)
        bet_fraction = (growth - 1) / 2

        return pot * bet_fraction

    def recommend_sizing(
        self,
        hand_strength: Optional[HandRank] = None,
        is_bluff: bool = False,
    ) -> BetSize:
        """
        Recommend a bet size based on context.

        Args:
            hand_strength: Strength of hero's hand
            is_bluff: Whether this is a bluff

        Returns:
            BetSize with recommended amount and reasoning
        """
        pot = self.state.pot.total
        spr = self.state.get_stack_to_pot_ratio()
        street = self.state.street

        # Preflop sizing
        if street == Street.PREFLOP:
            return self._preflop_sizing()

        # Postflop sizing based on SPR and board
        if spr > 10:
            # Deep stacked - use smaller sizes to preserve maneuverability
            strategy = SizingStrategy.SMALL
            amount = self.get_bet_size(strategy)
            reasoning = "Deep SPR - smaller sizes maintain flexibility"

        elif spr < 3:
            # Short stacked - often best to just jam
            player = self.state.players[self.state.action_on]
            strategy = SizingStrategy.LARGE
            amount = min(pot * 1.5, player.stack)
            reasoning = "Low SPR - committing size or all-in"

        else:
            # Medium SPR - standard sizing
            if is_bluff or (hand_strength and hand_strength <= HandRank.THREE_OF_A_KIND):
                # Polarized range - use larger sizes
                strategy = SizingStrategy.LARGE
                amount = self.get_bet_size(strategy)
                reasoning = "Polarized range - larger sizing with value/bluffs"
            else:
                # Medium strength - smaller sizes
                strategy = SizingStrategy.MEDIUM
                amount = self.get_bet_size(strategy)
                reasoning = "Medium strength - standard sizing"

        pct = amount / pot if pot > 0 else 0
        return BetSize(
            amount=round(amount, 2),
            pot_percentage=pct * 100,
            strategy=strategy,
            reasoning=reasoning,
        )

    def _preflop_sizing(self) -> BetSize:
        """Calculate preflop open raise size."""
        bb = self.state.big_blind
        pot = self.state.pot.total

        # Standard open: 2.5-3x BB
        # Add 1 BB per limper
        limpers = sum(
            1 for p in self.state.players
            if p.bet_this_round == bb and p.is_active
        )

        base_raise = 2.5 * bb
        per_limper = 1.0 * bb
        amount = base_raise + (limpers * per_limper)

        return BetSize(
            amount=round(amount, 2),
            pot_percentage=(amount / pot * 100) if pot > 0 else 0,
            strategy=SizingStrategy.MEDIUM,
            reasoning=f"Standard open: 2.5x + {limpers} limper(s)",
        )

    def get_raise_size(
        self,
        current_bet: float,
        strategy: SizingStrategy = SizingStrategy.MEDIUM,
    ) -> float:
        """
        Calculate raise size.

        Standard raise sizing is 2.5-3x the previous bet.

        Args:
            current_bet: Current bet to raise
            strategy: Sizing strategy

        Returns:
            Total raise amount
        """
        if strategy == SizingStrategy.SMALL:
            return current_bet * 2.2
        elif strategy == SizingStrategy.MEDIUM:
            return current_bet * 3.0
        elif strategy == SizingStrategy.LARGE:
            return current_bet * 4.0
        else:
            return current_bet * 3.0

    def get_3bet_size(self, open_raise: float, in_position: bool) -> float:
        """
        Calculate 3-bet size.

        Args:
            open_raise: Original raise amount
            in_position: Whether we're in position

        Returns:
            3-bet amount
        """
        # Out of position: 4x open
        # In position: 3x open
        multiplier = 3.0 if in_position else 4.0
        return open_raise * multiplier


def calculate_bet_size(
    pot: float,
    percentage: float = 67,
) -> float:
    """
    Simple bet size calculator.

    Args:
        pot: Current pot size
        percentage: Pot percentage (0-100)

    Returns:
        Bet amount
    """
    return pot * (percentage / 100)


def calculate_raise_to(
    pot: float,
    current_bet: float,
    percentage: float = 67,
) -> float:
    """
    Calculate raise-to amount.

    Args:
        pot: Current pot size
        current_bet: Current bet amount
        percentage: Pot percentage for raise size

    Returns:
        Raise-to amount
    """
    # Raise = current_bet + (pot + current_bet) * percentage
    raise_amount = (pot + current_bet) * (percentage / 100)
    return current_bet + raise_amount
