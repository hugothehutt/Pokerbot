"""Poker table management for 6-max No-Limit Hold'em simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto

from pokerbot.core.card import Card
from pokerbot.core.deck import Deck
from pokerbot.core.hand import Hand
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType, AvailableActions
from pokerbot.game.player import Player, Position
from pokerbot.bot.base import BaseBot


class SeatStatus(Enum):
    """Status of a seat at the table."""
    EMPTY = auto()
    OCCUPIED = auto()
    SITTING_OUT = auto()


@dataclass
class Seat:
    """Represents a seat at the poker table."""
    index: int
    bot: Optional[BaseBot] = None
    is_human: bool = False
    stack: float = 0.0
    status: SeatStatus = SeatStatus.EMPTY
    name: str = ""

    def __post_init__(self):
        if self.bot and not self.name:
            self.name = self.bot.name
        elif self.is_human and not self.name:
            self.name = "Hero"

    @property
    def is_active(self) -> bool:
        """Check if seat is occupied and has chips."""
        return self.status == SeatStatus.OCCUPIED and self.stack > 0

    def bot_type(self) -> str:
        """Get the bot type name."""
        if self.is_human:
            return "HUMAN"
        if self.bot:
            return self.bot.__class__.__name__.replace("Bot", "").upper()
        return "-"


class PokerTable:
    """
    Manages a 6-max No-Limit Hold'em poker table.

    Handles:
    - Seating players (bots and human)
    - Dealer button rotation
    - Blind collection
    - Hand orchestration (deal → betting → showdown → payout)
    - Chip movement
    """

    def __init__(
        self,
        num_seats: int = 6,
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        ante: float = 0.0,
    ):
        """
        Initialize the poker table.

        Args:
            num_seats: Number of seats (default 6)
            small_blind: Small blind amount
            big_blind: Big blind amount
            ante: Ante amount (0 for no ante)
        """
        self.num_seats = num_seats
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante

        # Seats
        self.seats: list[Seat] = [Seat(index=i) for i in range(num_seats)]

        # Table state
        self.button_position: int = 0
        self.hand_number: int = 0

        # Current hand state
        self.game_state: Optional[GameState] = None
        self.deck: Optional[Deck] = None
        self.board: list[Card] = []
        self.pot: float = 0.0
        self.current_bet: float = 0.0
        self.min_raise: float = 0.0
        self.street: Street = Street.PREFLOP

        # Action tracking for current hand
        self.action_history: list[tuple[int, Action]] = []  # (seat_index, action)
        self.hole_cards: dict[int, Hand] = {}  # seat_index -> Hand

        # Player states for current hand
        self.in_hand: list[bool] = [False] * num_seats
        self.all_in: list[bool] = [False] * num_seats
        self.bet_this_round: list[float] = [0.0] * num_seats
        self.total_invested: list[float] = [0.0] * num_seats

        # Callbacks
        self.on_action_callback: Optional[Callable[[int, Action], None]] = None

    def seat_player(
        self,
        seat_index: int,
        bot: Optional[BaseBot] = None,
        is_human: bool = False,
        stack: float = 100.0,
        name: str = "",
    ) -> bool:
        """
        Seat a player at the table.

        Args:
            seat_index: Seat index (0 to num_seats-1)
            bot: Bot instance (None for human)
            is_human: Whether this is a human player
            stack: Starting stack
            name: Player name

        Returns:
            True if successful
        """
        if seat_index < 0 or seat_index >= self.num_seats:
            return False

        seat = self.seats[seat_index]
        seat.bot = bot
        seat.is_human = is_human
        seat.stack = stack
        seat.status = SeatStatus.OCCUPIED
        seat.name = name or (bot.name if bot else "Hero" if is_human else f"Seat{seat_index}")

        return True

    def remove_player(self, seat_index: int) -> None:
        """Remove a player from a seat."""
        if 0 <= seat_index < self.num_seats:
            seat = self.seats[seat_index]
            seat.bot = None
            seat.is_human = False
            seat.stack = 0.0
            seat.status = SeatStatus.EMPTY
            seat.name = ""

    def get_active_seats(self) -> list[Seat]:
        """Get list of seats with active players."""
        return [s for s in self.seats if s.is_active]

    def get_players_in_hand(self) -> list[int]:
        """Get seat indices of players still in the hand."""
        return [i for i in range(self.num_seats) if self.in_hand[i]]

    def advance_button(self) -> None:
        """Move the dealer button to the next active player."""
        active = self.get_active_seats()
        if len(active) < 2:
            return

        # Find next active seat after current button
        for _ in range(self.num_seats):
            self.button_position = (self.button_position + 1) % self.num_seats
            if self.seats[self.button_position].is_active:
                break

    def start_hand(self) -> bool:
        """
        Start a new hand.

        Returns:
            True if hand started successfully
        """
        active = self.get_active_seats()
        if len(active) < 2:
            return False

        self.hand_number += 1
        self.deck = Deck()
        self.board = []
        self.pot = 0.0
        self.current_bet = self.big_blind
        self.min_raise = self.big_blind
        self.street = Street.PREFLOP
        self.action_history = []
        self.hole_cards = {}

        # Reset player states
        self.in_hand = [s.is_active for s in self.seats]
        self.all_in = [False] * self.num_seats
        self.bet_this_round = [0.0] * self.num_seats
        self.total_invested = [0.0] * self.num_seats

        # Collect antes
        if self.ante > 0:
            for i, seat in enumerate(self.seats):
                if seat.is_active:
                    ante_amount = min(self.ante, seat.stack)
                    seat.stack -= ante_amount
                    self.pot += ante_amount
                    self.total_invested[i] += ante_amount

        # Post blinds
        self._post_blinds()

        # Deal hole cards
        self._deal_hole_cards()

        return True

    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        active_indices = [i for i, s in enumerate(self.seats) if s.is_active]
        if len(active_indices) < 2:
            return

        # Find SB and BB positions
        btn_idx = self.button_position

        if len(active_indices) == 2:
            # Heads-up: button is SB
            sb_idx = btn_idx
            bb_idx = self._next_active_seat(sb_idx)
        else:
            sb_idx = self._next_active_seat(btn_idx)
            bb_idx = self._next_active_seat(sb_idx)

        # Post SB
        sb_amount = min(self.small_blind, self.seats[sb_idx].stack)
        self.seats[sb_idx].stack -= sb_amount
        self.bet_this_round[sb_idx] = sb_amount
        self.total_invested[sb_idx] += sb_amount
        self.pot += sb_amount
        if self.seats[sb_idx].stack == 0:
            self.all_in[sb_idx] = True

        # Post BB
        bb_amount = min(self.big_blind, self.seats[bb_idx].stack)
        self.seats[bb_idx].stack -= bb_amount
        self.bet_this_round[bb_idx] = bb_amount
        self.total_invested[bb_idx] += bb_amount
        self.pot += bb_amount
        if self.seats[bb_idx].stack == 0:
            self.all_in[bb_idx] = True

    def _deal_hole_cards(self) -> None:
        """Deal 2 cards to each active player."""
        for i, seat in enumerate(self.seats):
            if self.in_hand[i]:
                cards = self.deck.deal(2)
                self.hole_cards[i] = Hand(cards)
                if seat.bot:
                    seat.bot.set_hole_cards(self.hole_cards[i])

    def _next_active_seat(self, from_seat: int) -> int:
        """Get next active seat after given position (has chips, for dealing/setup)."""
        for _ in range(self.num_seats):
            from_seat = (from_seat + 1) % self.num_seats
            if self.seats[from_seat].is_active:
                return from_seat
        return from_seat

    def _next_in_hand_seat(self, from_seat: int) -> int:
        """Get next seat still in the current hand (not folded)."""
        for _ in range(self.num_seats):
            from_seat = (from_seat + 1) % self.num_seats
            if self.in_hand[from_seat]:
                return from_seat
        return from_seat

    def _get_first_to_act(self) -> int:
        """Get the first player to act on current street."""
        active = [i for i in range(self.num_seats) if self.in_hand[i] and not self.all_in[i]]
        if not active:
            return -1

        active_count = len([s for s in self.seats if s.is_active])

        if self.street == Street.PREFLOP:
            if active_count == 2:
                # Heads-up: Button/SB acts first preflop
                start = self.button_position
            else:
                # Multi-way: First to act is UTG (left of BB)
                sb_idx = self._next_active_seat(self.button_position)
                bb_idx = self._next_active_seat(sb_idx)
                start = self._next_active_seat(bb_idx)
        else:
            # Postflop: First active player after button
            if active_count == 2:
                # Heads-up postflop: BB (non-button) acts first
                # Button is SB in heads-up, so BB is next active after button
                start = self._next_active_seat(self.button_position)
            else:
                # Multi-way postflop: SB acts first (first active after button)
                start = self._next_active_seat(self.button_position)

        # Find first non-all-in player
        current = start
        for _ in range(self.num_seats):
            if self.in_hand[current] and not self.all_in[current]:
                return current
            current = self._next_active_seat(current)

        return -1

    def get_available_actions(self, seat_index: int) -> AvailableActions:
        """Get available actions for a player."""
        if not self.in_hand[seat_index] or self.all_in[seat_index]:
            return AvailableActions(can_fold=False)

        seat = self.seats[seat_index]
        to_call = self.current_bet - self.bet_this_round[seat_index]

        available = AvailableActions(
            can_fold=True,
            can_check=(to_call == 0),
            can_call=(to_call > 0 and to_call < seat.stack),
            call_amount=min(to_call, seat.stack),
        )

        # Betting/raising
        if self.current_bet == 0:
            available.can_bet = seat.stack >= self.big_blind
            available.min_bet = self.big_blind
        else:
            min_raise_to = self.current_bet + self.min_raise
            available.can_raise = seat.stack > to_call
            available.min_raise = min_raise_to
            available.max_raise = seat.stack + self.bet_this_round[seat_index]

        return available

    def apply_action(self, seat_index: int, action: Action) -> bool:
        """
        Apply a player action.

        Args:
            seat_index: Index of acting player
            action: Action to apply

        Returns:
            True if action was valid and applied
        """
        seat = self.seats[seat_index]
        if not self.in_hand[seat_index]:
            return False

        if action.action_type == ActionType.FOLD:
            self.in_hand[seat_index] = False

        elif action.action_type == ActionType.CHECK:
            pass  # No chip movement

        elif action.action_type == ActionType.CALL:
            to_call = min(self.current_bet - self.bet_this_round[seat_index], seat.stack)
            seat.stack -= to_call
            self.bet_this_round[seat_index] += to_call
            self.total_invested[seat_index] += to_call
            self.pot += to_call
            if seat.stack == 0:
                self.all_in[seat_index] = True

        elif action.action_type == ActionType.BET:
            bet_amount = min(action.amount, seat.stack)
            seat.stack -= bet_amount
            self.bet_this_round[seat_index] = bet_amount
            self.total_invested[seat_index] += bet_amount
            self.pot += bet_amount
            self.current_bet = bet_amount
            self.min_raise = bet_amount
            if seat.stack == 0:
                self.all_in[seat_index] = True

        elif action.action_type == ActionType.RAISE:
            raise_to = min(action.amount, seat.stack + self.bet_this_round[seat_index])
            raise_amount = raise_to - self.bet_this_round[seat_index]
            seat.stack -= raise_amount
            self.min_raise = max(self.min_raise, raise_to - self.current_bet)
            self.current_bet = raise_to
            self.bet_this_round[seat_index] = raise_to
            self.total_invested[seat_index] += raise_amount
            self.pot += raise_amount
            if seat.stack == 0:
                self.all_in[seat_index] = True

        elif action.action_type == ActionType.ALL_IN:
            all_in_amount = seat.stack
            new_total = self.bet_this_round[seat_index] + all_in_amount

            if new_total > self.current_bet:
                raise_amount = new_total - self.current_bet
                if raise_amount >= self.min_raise:
                    self.min_raise = raise_amount
                self.current_bet = new_total

            seat.stack = 0
            self.pot += all_in_amount
            self.total_invested[seat_index] += all_in_amount
            self.bet_this_round[seat_index] = new_total
            self.all_in[seat_index] = True

        # Record action
        self.action_history.append((seat_index, action))

        # Callback
        if self.on_action_callback:
            self.on_action_callback(seat_index, action)

        return True

    def is_betting_round_complete(self, last_aggressor: int, current_actor: int) -> bool:
        """Check if the betting round is complete."""
        active = [i for i in range(self.num_seats) if self.in_hand[i] and not self.all_in[i]]

        if len(active) <= 1:
            return True

        # All active players must have matched the bet
        for i in active:
            if self.bet_this_round[i] < self.current_bet:
                return False

        return True

    def deal_street(self, street: Street) -> None:
        """Deal community cards for the next street."""
        self.street = street

        # Reset betting
        self.current_bet = 0.0
        self.bet_this_round = [0.0] * self.num_seats

        if street == Street.FLOP:
            self.board.extend(self.deck.deal(3))
        elif street in (Street.TURN, Street.RIVER):
            self.board.extend(self.deck.deal(1))

    def get_winners(self) -> list[tuple[int, float]]:
        """
        Determine winners and their share of the pot.

        Returns:
            List of (seat_index, amount_won) tuples
        """
        in_hand_indices = self.get_players_in_hand()

        if len(in_hand_indices) == 1:
            # Everyone else folded
            return [(in_hand_indices[0], self.pot)]

        # Showdown - evaluate hands
        hands = [(i, self.hole_cards[i]) for i in in_hand_indices]
        winner_indices = HandEvaluator.winners(
            [h for _, h in hands],
            self.board,
        )

        # Split pot among winners
        win_amount = self.pot / len(winner_indices)
        return [(hands[i][0], win_amount) for i in winner_indices]

    def payout_winners(self, winners: list[tuple[int, float]]) -> None:
        """Pay out the pot to winners."""
        for seat_idx, amount in winners:
            self.seats[seat_idx].stack += amount

    def is_hand_over(self) -> bool:
        """Check if the current hand is over."""
        in_hand = self.get_players_in_hand()

        # Only one player left
        if len(in_hand) <= 1:
            return True

        # All streets complete and betting done
        if self.street == Street.RIVER:
            active = [i for i in in_hand if not self.all_in[i]]
            if len(active) <= 1:
                return True
            # Check if betting is complete
            for i in active:
                if self.bet_this_round[i] < self.current_bet:
                    return False
            return True

        return False

    def get_position_name(self, seat_index: int) -> str:
        """Get position name for a seat (BTN, SB, BB, etc.)."""
        if not self.seats[seat_index].is_active:
            return "-"

        active_indices = [i for i in range(self.num_seats) if self.seats[i].is_active]
        if len(active_indices) < 2:
            return "-"

        btn_idx = self.button_position

        if len(active_indices) == 2:
            # Heads-up
            if seat_index == btn_idx:
                return "BTN/SB"
            return "BB"

        # Calculate position relative to button
        positions = []
        current = btn_idx
        for _ in range(len(active_indices)):
            positions.append(current)
            current = self._next_active_seat(current)

        if seat_index not in positions:
            return "-"

        pos_idx = positions.index(seat_index)
        names = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]

        if len(active_indices) <= 6:
            # Map positions for smaller tables
            if pos_idx < len(names):
                return names[pos_idx]

        return f"S{pos_idx}"

    def get_table_state_string(self) -> str:
        """Get a string representation of the current table state."""
        lines = []
        lines.append(f"Hand #{self.hand_number} | Pot: ${self.pot:.2f} | Street: {self.street.name}")

        if self.board:
            board_str = " ".join(c.pretty for c in self.board)
            lines.append(f"Board: {board_str}")

        lines.append("")
        lines.append(f"{'Pos':<6} {'Name':<12} {'Type':<8} {'Stack':>8} {'Bet':>8} {'Status':<10}")
        lines.append("-" * 60)

        for i, seat in enumerate(self.seats):
            if seat.status == SeatStatus.EMPTY:
                continue

            pos = self.get_position_name(i)
            name = seat.name[:11]
            bot_type = seat.bot_type()
            stack = f"${seat.stack:.2f}"
            bet = f"${self.bet_this_round[i]:.2f}" if self.bet_this_round[i] > 0 else "-"

            if not self.in_hand[i]:
                status = "Folded"
            elif self.all_in[i]:
                status = "All-in"
            else:
                status = "Active"

            lines.append(f"{pos:<6} {name:<12} {bot_type:<8} {stack:>8} {bet:>8} {status:<10}")

        return "\n".join(lines)
