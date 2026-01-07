"""Player representation for Texas Hold'em poker."""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.action import Action


class Position(IntEnum):
    """
    Table positions for Texas Hold'em.

    Positions are numbered from the button (0) clockwise.
    For 6-max:
        BTN (0) -> SB (1) -> BB (2) -> UTG (3) -> HJ (4) -> CO (5)
    """
    BTN = 0   # Button (dealer)
    SB = 1    # Small Blind
    BB = 2    # Big Blind
    UTG = 3   # Under The Gun
    UTG1 = 4  # UTG+1 / Hijack in 6-max
    UTG2 = 5  # UTG+2 / Cutoff in 6-max
    LJ = 6    # Lojack
    HJ = 7    # Hijack
    CO = 8    # Cutoff

    @property
    def abbreviation(self) -> str:
        """Get position abbreviation."""
        return self.name

    @property
    def is_blinds(self) -> bool:
        """Check if this position is a blind."""
        return self in (Position.SB, Position.BB)

    @property
    def is_early(self) -> bool:
        """Check if this is an early position."""
        return self in (Position.UTG, Position.UTG1, Position.UTG2)

    @property
    def is_late(self) -> bool:
        """Check if this is a late position."""
        return self in (Position.HJ, Position.CO, Position.BTN)

    @classmethod
    def for_table_size(cls, num_players: int) -> list[Position]:
        """
        Get positions for a given table size.

        Args:
            num_players: Number of players (2-9)

        Returns:
            List of positions for the table
        """
        if num_players == 2:
            return [cls.BTN, cls.BB]  # Heads up - BTN is SB
        elif num_players <= 6:
            # 6-max positions
            positions = [cls.BTN, cls.SB, cls.BB, cls.UTG, cls.HJ, cls.CO]
            return positions[:num_players]
        else:
            # Full ring
            all_positions = [
                cls.BTN, cls.SB, cls.BB, cls.UTG, cls.UTG1, cls.UTG2,
                cls.LJ, cls.HJ, cls.CO
            ]
            return all_positions[:num_players]


@dataclass
class Player:
    """
    Represents a player at the poker table.

    Attributes:
        name: Player name/identifier
        stack: Current chip stack
        position: Current table position
        hole_cards: Player's hole cards (None if folded/not dealt)
        is_active: Whether player is still in the hand
        is_all_in: Whether player is all-in
        bet_this_round: Amount bet in current betting round
        total_invested: Total amount invested in current hand
        actions: List of actions taken this hand
    """
    name: str
    stack: float
    position: Optional[Position] = None
    hole_cards: Optional[Hand] = None
    is_active: bool = True
    is_all_in: bool = False
    bet_this_round: float = 0.0
    total_invested: float = 0.0
    actions: list[Action] = field(default_factory=list)

    def bet(self, amount: float) -> float:
        """
        Make a bet, returning the actual amount bet.

        If the bet is more than the stack, goes all-in.

        Args:
            amount: Amount to bet

        Returns:
            Actual amount bet (may be less if all-in)
        """
        actual = min(amount, self.stack)
        self.stack -= actual
        self.bet_this_round += actual
        self.total_invested += actual

        if self.stack == 0:
            self.is_all_in = True

        return actual

    def fold(self) -> None:
        """Fold the hand."""
        self.is_active = False
        self.hole_cards = None

    def reset_for_new_round(self) -> None:
        """Reset betting state for a new betting round."""
        self.bet_this_round = 0.0

    def reset_for_new_hand(self) -> None:
        """Reset all state for a new hand."""
        self.hole_cards = None
        self.is_active = True
        self.is_all_in = False
        self.bet_this_round = 0.0
        self.total_invested = 0.0
        self.actions = []

    def can_act(self) -> bool:
        """Check if player can take an action."""
        return self.is_active and not self.is_all_in

    @property
    def effective_stack(self) -> float:
        """Get effective stack (stack + amount already bet)."""
        return self.stack + self.bet_this_round

    def __str__(self) -> str:
        pos = f"({self.position.abbreviation})" if self.position else ""
        return f"{self.name}{pos}: {self.stack:.2f}"

    def __repr__(self) -> str:
        return f"Player(name='{self.name}', stack={self.stack}, position={self.position})"


@dataclass
class PlayerStats:
    """
    Statistics for tracking player tendencies.

    Used for opponent modeling and exploitative play.
    """
    hands_played: int = 0
    vpip_hands: int = 0  # Voluntarily put $ in pot
    pfr_hands: int = 0   # Preflop raise
    aggression_actions: int = 0  # Bets and raises
    passive_actions: int = 0     # Checks and calls
    three_bet_opportunities: int = 0
    three_bets: int = 0
    cbet_opportunities: int = 0
    cbets: int = 0
    fold_to_cbet: int = 0
    fold_to_cbet_opportunities: int = 0

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.vpip_hands / self.hands_played * 100

    @property
    def pfr(self) -> float:
        """Preflop Raise percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.pfr_hands / self.hands_played * 100

    @property
    def aggression_factor(self) -> float:
        """Aggression Factor (AF) = (Bets + Raises) / Calls."""
        if self.passive_actions == 0:
            return float('inf') if self.aggression_actions > 0 else 0.0
        return self.aggression_actions / self.passive_actions

    @property
    def three_bet_pct(self) -> float:
        """3-bet percentage."""
        if self.three_bet_opportunities == 0:
            return 0.0
        return self.three_bets / self.three_bet_opportunities * 100

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        if self.cbet_opportunities == 0:
            return 0.0
        return self.cbets / self.cbet_opportunities * 100

    @property
    def fold_to_cbet_pct(self) -> float:
        """Fold to continuation bet percentage."""
        if self.fold_to_cbet_opportunities == 0:
            return 0.0
        return self.fold_to_cbet / self.fold_to_cbet_opportunities * 100

    def player_type(self) -> str:
        """
        Classify player type based on stats.

        Returns one of: 'rock', 'nit', 'tag', 'lag', 'fish', 'unknown'
        """
        if self.hands_played < 20:
            return 'unknown'

        vpip = self.vpip
        pfr = self.pfr

        if vpip < 15 and pfr < 10:
            return 'nit'  # Very tight, rarely raises
        elif vpip < 22 and pfr > 15:
            return 'tag'  # Tight-aggressive
        elif vpip > 30 and pfr > 25:
            return 'lag'  # Loose-aggressive
        elif vpip > 35 and pfr < 15:
            return 'fish'  # Loose-passive
        elif vpip < 18:
            return 'rock'  # Very tight
        else:
            return 'unknown'
