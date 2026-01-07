"""Game state management for Texas Hold'em poker."""

from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from pokerbot.core.card import Card
from pokerbot.core.deck import Deck
from pokerbot.core.hand import Hand
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.game.player import Player, Position
from pokerbot.game.action import Action, ActionType, AvailableActions


class Street(IntEnum):
    """Betting streets in Texas Hold'em."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

    def __str__(self) -> str:
        return self.name.capitalize()

    @property
    def num_board_cards(self) -> int:
        """Number of community cards on this street."""
        return {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
            Street.SHOWDOWN: 5,
        }[self]


@dataclass
class Pot:
    """Represents the pot(s) in a poker hand."""
    main_pot: float = 0.0
    side_pots: list[tuple[float, list[int]]] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Total amount in all pots."""
        return self.main_pot + sum(sp[0] for sp in self.side_pots)

    def add(self, amount: float) -> None:
        """Add chips to the main pot."""
        self.main_pot += amount


@dataclass
class GameState:
    """
    Complete state of a Texas Hold'em hand.

    This class tracks all information needed to make decisions
    and evaluate the game state.
    """
    # Game configuration
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    num_players: int = 6

    # Players
    players: list[Player] = field(default_factory=list)
    button_position: int = 0

    # Hand state
    deck: Optional[Deck] = None
    board: list[Card] = field(default_factory=list)
    pot: Pot = field(default_factory=Pot)
    street: Street = Street.PREFLOP
    current_bet: float = 0.0
    min_raise: float = 0.0
    last_raiser: int = -1
    action_on: int = 0  # Index of player to act

    # History
    action_history: list[list[Action]] = field(default_factory=lambda: [[], [], [], []])

    def __post_init__(self):
        if not self.players:
            self.players = [Player(f"Player_{i}", 100.0) for i in range(self.num_players)]
        self._assign_positions()

    def _assign_positions(self) -> None:
        """Assign positions to players based on button position."""
        positions = Position.for_table_size(len(self.players))
        for i, player in enumerate(self.players):
            # Position is relative to button
            pos_index = (i - self.button_position) % len(self.players)
            player.position = positions[pos_index]

    def start_hand(self, deck: Optional[Deck] = None) -> None:
        """Initialize a new hand."""
        # Reset players
        for player in self.players:
            player.reset_for_new_hand()

        # Reset state
        self.deck = deck or Deck()
        self.board = []
        self.pot = Pot()
        self.street = Street.PREFLOP
        self.current_bet = self.big_blind
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.action_history = [[], [], [], []]

        # Collect antes
        if self.ante > 0:
            for player in self.players:
                self.pot.add(player.bet(self.ante))

        # Post blinds
        sb_idx = self._get_position_index(Position.SB)
        bb_idx = self._get_position_index(Position.BB)

        self.pot.add(self.players[sb_idx].bet(self.small_blind))
        self.pot.add(self.players[bb_idx].bet(self.big_blind))

        # Deal hole cards
        for player in self.players:
            cards = self.deck.deal(2)
            player.hole_cards = Hand(cards)

        # Action starts left of BB (UTG)
        self.action_on = (bb_idx + 1) % len(self.players)

    def _get_position_index(self, position: Position) -> int:
        """Get player index for a position."""
        for i, player in enumerate(self.players):
            if player.position == position:
                return i
        raise ValueError(f"No player at position {position}")

    def get_available_actions(self, player_idx: Optional[int] = None) -> AvailableActions:
        """Get available actions for a player."""
        if player_idx is None:
            player_idx = self.action_on

        player = self.players[player_idx]

        if not player.can_act():
            return AvailableActions(can_fold=False)

        to_call = self.current_bet - player.bet_this_round

        available = AvailableActions(
            can_fold=True,
            can_check=(to_call == 0),
            can_call=(to_call > 0 and to_call < player.stack),
            call_amount=min(to_call, player.stack),
        )

        # Betting/raising
        if self.current_bet == 0:
            # No bet yet - can bet
            available.can_bet = player.stack >= self.big_blind
            available.min_bet = self.big_blind
        else:
            # Bet exists - can raise
            min_raise_to = self.current_bet + self.min_raise
            available.can_raise = player.stack > to_call
            available.min_raise = min_raise_to
            available.max_raise = player.stack + player.bet_this_round

        return available

    def apply_action(self, action: Action) -> None:
        """Apply a player action to the game state."""
        player = self.players[self.action_on]

        if action.action_type == ActionType.FOLD:
            player.fold()

        elif action.action_type == ActionType.CHECK:
            pass  # No change

        elif action.action_type == ActionType.CALL:
            to_call = self.current_bet - player.bet_this_round
            self.pot.add(player.bet(to_call))

        elif action.action_type == ActionType.BET:
            self.pot.add(player.bet(action.amount))
            self.current_bet = action.amount
            self.min_raise = action.amount
            self.last_raiser = self.action_on

        elif action.action_type == ActionType.RAISE:
            raise_amount = action.amount - self.current_bet
            self.min_raise = max(self.min_raise, raise_amount)
            to_call = self.current_bet - player.bet_this_round
            total_bet = to_call + raise_amount + (self.current_bet - player.bet_this_round - to_call)
            self.pot.add(player.bet(action.amount - player.bet_this_round))
            self.current_bet = action.amount
            self.last_raiser = self.action_on

        elif action.action_type == ActionType.ALL_IN:
            amount = player.bet(player.stack)
            self.pot.add(amount)
            if player.bet_this_round > self.current_bet:
                raise_amount = player.bet_this_round - self.current_bet
                if raise_amount >= self.min_raise:
                    self.min_raise = raise_amount
                    self.last_raiser = self.action_on
                self.current_bet = player.bet_this_round

        # Record action
        player.actions.append(action)
        self.action_history[self.street.value].append(
            Action(action.action_type, action.amount, self.action_on)
        )

        # Move to next player
        self._advance_action()

    def _advance_action(self) -> None:
        """Advance to the next player to act or next street."""
        # Find next active player
        start = self.action_on
        for _ in range(len(self.players)):
            self.action_on = (self.action_on + 1) % len(self.players)
            player = self.players[self.action_on]

            if player.can_act():
                # Check if betting round is complete
                if self._is_round_complete():
                    self._end_betting_round()
                return

        # No active players - end round
        self._end_betting_round()

    def _is_round_complete(self) -> bool:
        """Check if the betting round is complete."""
        active_players = [p for p in self.players if p.is_active]

        if len(active_players) <= 1:
            return True

        # All active players must have matched the bet or be all-in
        for player in active_players:
            if player.can_act():
                if player.bet_this_round < self.current_bet:
                    return False

        # Check if everyone has had a chance to act
        # (action has gone around since last raise)
        return True

    def _end_betting_round(self) -> None:
        """End the current betting round and move to next street."""
        active_players = [p for p in self.players if p.is_active]

        # Reset for new round
        for player in self.players:
            player.reset_for_new_round()
        self.current_bet = 0.0
        self.last_raiser = -1

        # Check for hand end
        if len(active_players) <= 1 or self.street == Street.RIVER:
            self.street = Street.SHOWDOWN
            return

        # Deal next street
        self.street = Street(self.street.value + 1)

        if self.street == Street.FLOP:
            self.board.extend(self.deck.deal(3))
        elif self.street in (Street.TURN, Street.RIVER):
            self.board.extend(self.deck.deal(1))

        # Action starts with first active player after button
        self._set_first_to_act()

    def _set_first_to_act(self) -> None:
        """Set action to first active player after button."""
        for i in range(len(self.players)):
            idx = (self.button_position + i + 1) % len(self.players)
            if self.players[idx].can_act():
                self.action_on = idx
                return

    @property
    def is_hand_over(self) -> bool:
        """Check if the hand is over."""
        if self.street == Street.SHOWDOWN:
            return True
        active = sum(1 for p in self.players if p.is_active)
        return active <= 1

    def get_winners(self) -> list[int]:
        """Get indices of winning players."""
        active_players = [(i, p) for i, p in enumerate(self.players) if p.is_active]

        if len(active_players) == 1:
            return [active_players[0][0]]

        # Showdown
        hands = [p.hole_cards for _, p in active_players]
        indices = [i for i, _ in active_players]

        winner_positions = HandEvaluator.winners(hands, self.board)
        return [indices[pos] for pos in winner_positions]

    def get_pot_odds(self, player_idx: Optional[int] = None) -> float:
        """
        Calculate pot odds for calling.

        Returns the ratio of pot to call amount.
        """
        if player_idx is None:
            player_idx = self.action_on

        player = self.players[player_idx]
        to_call = self.current_bet - player.bet_this_round

        if to_call <= 0:
            return float('inf')

        return self.pot.total / to_call

    def get_stack_to_pot_ratio(self, player_idx: Optional[int] = None) -> float:
        """Calculate stack-to-pot ratio (SPR)."""
        if player_idx is None:
            player_idx = self.action_on

        if self.pot.total == 0:
            return float('inf')

        return self.players[player_idx].stack / self.pot.total

    def copy(self) -> GameState:
        """Create a deep copy of the game state."""
        return deepcopy(self)

    def __str__(self) -> str:
        board_str = " ".join(str(c) for c in self.board) if self.board else "[]"
        return (
            f"Street: {self.street} | Board: {board_str} | "
            f"Pot: {self.pot.total:.2f} | To Act: {self.players[self.action_on].name}"
        )
