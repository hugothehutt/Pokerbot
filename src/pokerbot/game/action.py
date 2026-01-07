"""Player actions for Texas Hold'em poker."""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class ActionType(Enum):
    """Types of actions a player can take."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Action:
    """
    Represents a player action in a betting round.

    Attributes:
        action_type: The type of action taken
        amount: The amount bet/raised (0 for fold/check)
        player_index: Index of the player taking the action
    """
    action_type: ActionType
    amount: float = 0.0
    player_index: int = -1

    def __str__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return str(self.action_type)
        elif self.action_type == ActionType.CALL:
            return f"call {self.amount:.2f}"
        elif self.action_type == ActionType.BET:
            return f"bet {self.amount:.2f}"
        elif self.action_type == ActionType.RAISE:
            return f"raise to {self.amount:.2f}"
        elif self.action_type == ActionType.ALL_IN:
            return f"all-in {self.amount:.2f}"
        return str(self.action_type)

    @classmethod
    def fold(cls, player_index: int = -1) -> Action:
        """Create a fold action."""
        return cls(ActionType.FOLD, 0, player_index)

    @classmethod
    def check(cls, player_index: int = -1) -> Action:
        """Create a check action."""
        return cls(ActionType.CHECK, 0, player_index)

    @classmethod
    def call(cls, amount: float, player_index: int = -1) -> Action:
        """Create a call action."""
        return cls(ActionType.CALL, amount, player_index)

    @classmethod
    def bet(cls, amount: float, player_index: int = -1) -> Action:
        """Create a bet action."""
        return cls(ActionType.BET, amount, player_index)

    @classmethod
    def raise_to(cls, amount: float, player_index: int = -1) -> Action:
        """Create a raise action."""
        return cls(ActionType.RAISE, amount, player_index)

    @classmethod
    def all_in(cls, amount: float, player_index: int = -1) -> Action:
        """Create an all-in action."""
        return cls(ActionType.ALL_IN, amount, player_index)

    @property
    def is_aggressive(self) -> bool:
        """Check if this is an aggressive action (bet/raise/all-in)."""
        return self.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)

    @property
    def is_passive(self) -> bool:
        """Check if this is a passive action (check/call)."""
        return self.action_type in (ActionType.CHECK, ActionType.CALL)

    @property
    def is_fold(self) -> bool:
        """Check if this is a fold action."""
        return self.action_type == ActionType.FOLD


@dataclass
class AvailableActions:
    """
    Represents the actions available to a player at a decision point.

    Attributes:
        can_fold: Whether folding is an option
        can_check: Whether checking is an option
        can_call: Whether calling is an option
        call_amount: Amount required to call
        can_bet: Whether betting is an option
        min_bet: Minimum bet amount
        can_raise: Whether raising is an option
        min_raise: Minimum raise amount
        max_raise: Maximum raise amount (stack size)
    """
    can_fold: bool = True
    can_check: bool = False
    can_call: bool = False
    call_amount: float = 0.0
    can_bet: bool = False
    min_bet: float = 0.0
    can_raise: bool = False
    min_raise: float = 0.0
    max_raise: float = 0.0

    def validate_action(self, action: Action) -> bool:
        """Check if an action is valid given available actions."""
        if action.action_type == ActionType.FOLD:
            return self.can_fold
        elif action.action_type == ActionType.CHECK:
            return self.can_check
        elif action.action_type == ActionType.CALL:
            return self.can_call and action.amount >= self.call_amount
        elif action.action_type == ActionType.BET:
            return self.can_bet and action.amount >= self.min_bet
        elif action.action_type == ActionType.RAISE:
            return self.can_raise and self.min_raise <= action.amount <= self.max_raise
        elif action.action_type == ActionType.ALL_IN:
            return True  # All-in is always valid if you have chips
        return False

    def get_valid_actions(self) -> list[ActionType]:
        """Get list of valid action types."""
        actions = []
        if self.can_fold:
            actions.append(ActionType.FOLD)
        if self.can_check:
            actions.append(ActionType.CHECK)
        if self.can_call:
            actions.append(ActionType.CALL)
        if self.can_bet:
            actions.append(ActionType.BET)
        if self.can_raise:
            actions.append(ActionType.RAISE)
        actions.append(ActionType.ALL_IN)  # Always can go all-in
        return actions
