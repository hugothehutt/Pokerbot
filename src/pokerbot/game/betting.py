"""Betting round management for Texas Hold'em poker."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto

from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType


class BettingPattern(Enum):
    """Common betting patterns for analysis."""
    CHECK = auto()
    BET_SMALL = auto()   # < 40% pot
    BET_MEDIUM = auto()  # 40-70% pot
    BET_LARGE = auto()   # 70-100% pot
    OVERBET = auto()     # > 100% pot
    ALL_IN = auto()
    RAISE_SMALL = auto()
    RAISE_LARGE = auto()


@dataclass
class BetSizing:
    """
    Bet sizing utilities based on pot size.

    Provides standard sizing options used in modern poker strategy.
    """

    @staticmethod
    def pot_percentage(pot: float, percentage: float) -> float:
        """Calculate bet size as percentage of pot."""
        return pot * (percentage / 100)

    @staticmethod
    def quarter_pot(pot: float) -> float:
        """25% pot bet."""
        return pot * 0.25

    @staticmethod
    def third_pot(pot: float) -> float:
        """33% pot bet."""
        return pot * 0.33

    @staticmethod
    def half_pot(pot: float) -> float:
        """50% pot bet."""
        return pot * 0.50

    @staticmethod
    def two_thirds_pot(pot: float) -> float:
        """67% pot bet."""
        return pot * 0.67

    @staticmethod
    def three_quarters_pot(pot: float) -> float:
        """75% pot bet."""
        return pot * 0.75

    @staticmethod
    def pot_sized(pot: float) -> float:
        """100% pot bet."""
        return pot

    @staticmethod
    def overbet(pot: float, multiplier: float = 1.5) -> float:
        """Overbet (> pot)."""
        return pot * multiplier

    @classmethod
    def classify_bet(cls, bet: float, pot: float) -> BettingPattern:
        """Classify a bet size relative to pot."""
        if pot == 0:
            return BettingPattern.BET_SMALL

        ratio = bet / pot
        if ratio < 0.4:
            return BettingPattern.BET_SMALL
        elif ratio < 0.7:
            return BettingPattern.BET_MEDIUM
        elif ratio <= 1.0:
            return BettingPattern.BET_LARGE
        else:
            return BettingPattern.OVERBET


@dataclass
class BettingRound:
    """
    Manages a single betting round.

    Tracks actions and provides analysis of betting patterns.
    """
    street: Street
    starting_pot: float
    actions: list[Action] = field(default_factory=list)
    num_raises: int = 0
    num_bets: int = 0

    def record_action(self, action: Action) -> None:
        """Record an action in this betting round."""
        self.actions.append(action)

        if action.action_type == ActionType.BET:
            self.num_bets += 1
        elif action.action_type == ActionType.RAISE:
            self.num_raises += 1

    @property
    def total_aggression(self) -> int:
        """Total aggressive actions (bets + raises)."""
        return self.num_bets + self.num_raises

    @property
    def is_checked_around(self) -> bool:
        """Check if the round was checked around."""
        return all(a.action_type == ActionType.CHECK for a in self.actions)

    @property
    def is_multiway(self) -> bool:
        """Check if multiple players saw action."""
        players_acted = set()
        for action in self.actions:
            if action.action_type != ActionType.FOLD:
                players_acted.add(action.player_index)
        return len(players_acted) > 2

    def get_aggressor(self) -> Optional[int]:
        """Get the player who made the last aggressive action."""
        for action in reversed(self.actions):
            if action.is_aggressive:
                return action.player_index
        return None

    def get_action_sequence(self) -> str:
        """Get a string representation of the action sequence."""
        symbols = {
            ActionType.FOLD: "f",
            ActionType.CHECK: "x",
            ActionType.CALL: "c",
            ActionType.BET: "b",
            ActionType.RAISE: "r",
            ActionType.ALL_IN: "a",
        }
        return "".join(symbols.get(a.action_type, "?") for a in self.actions)


@dataclass
class HandHistory:
    """
    Complete history of a poker hand.

    Used for analysis and replay of hands.
    """
    # Game info
    small_blind: float
    big_blind: float
    num_players: int

    # Player info
    player_names: list[str]
    starting_stacks: list[float]
    positions: list[str]

    # Hole cards (if known)
    hole_cards: dict[int, str] = field(default_factory=dict)

    # Board
    flop: Optional[str] = None
    turn: Optional[str] = None
    river: Optional[str] = None

    # Betting rounds
    preflop_actions: list[str] = field(default_factory=list)
    flop_actions: list[str] = field(default_factory=list)
    turn_actions: list[str] = field(default_factory=list)
    river_actions: list[str] = field(default_factory=list)

    # Results
    pot_size: float = 0.0
    winners: list[int] = field(default_factory=list)
    winnings: dict[int, float] = field(default_factory=dict)

    def to_string(self) -> str:
        """Convert hand history to readable string format."""
        lines = []
        lines.append(f"Blinds: {self.small_blind}/{self.big_blind}")
        lines.append(f"Players: {self.num_players}")
        lines.append("")

        for i, name in enumerate(self.player_names):
            pos = self.positions[i]
            stack = self.starting_stacks[i]
            hole = self.hole_cards.get(i, "??")
            lines.append(f"{pos}: {name} ({stack:.2f}) [{hole}]")

        lines.append("")

        if self.preflop_actions:
            lines.append("Preflop:")
            lines.append("  " + ", ".join(self.preflop_actions))

        if self.flop:
            lines.append(f"\nFlop: {self.flop}")
            if self.flop_actions:
                lines.append("  " + ", ".join(self.flop_actions))

        if self.turn:
            lines.append(f"\nTurn: {self.turn}")
            if self.turn_actions:
                lines.append("  " + ", ".join(self.turn_actions))

        if self.river:
            lines.append(f"\nRiver: {self.river}")
            if self.river_actions:
                lines.append("  " + ", ".join(self.river_actions))

        lines.append(f"\nPot: {self.pot_size:.2f}")
        if self.winners:
            winner_names = [self.player_names[i] for i in self.winners]
            lines.append(f"Winner(s): {', '.join(winner_names)}")

        return "\n".join(lines)


def calculate_pot_odds(pot: float, to_call: float) -> float:
    """
    Calculate pot odds as a ratio.

    Args:
        pot: Current pot size
        to_call: Amount needed to call

    Returns:
        Pot odds ratio (e.g., 3.0 means 3:1)
    """
    if to_call <= 0:
        return float('inf')
    return pot / to_call


def calculate_implied_odds(
    pot: float,
    to_call: float,
    expected_future_winnings: float
) -> float:
    """
    Calculate implied odds.

    Args:
        pot: Current pot size
        to_call: Amount needed to call
        expected_future_winnings: Expected additional winnings if you hit

    Returns:
        Implied odds ratio
    """
    if to_call <= 0:
        return float('inf')
    return (pot + expected_future_winnings) / to_call


def required_equity(pot: float, to_call: float) -> float:
    """
    Calculate required equity to profitably call.

    Args:
        pot: Current pot size
        to_call: Amount needed to call

    Returns:
        Required equity as a decimal (0-1)
    """
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call)
