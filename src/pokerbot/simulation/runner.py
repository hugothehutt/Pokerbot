"""Game runner for interactive poker simulation."""

from __future__ import annotations
import os
import sys
from typing import Optional, Callable
from dataclasses import dataclass

from pokerbot.core.hand import Hand
from pokerbot.game.state import Street
from pokerbot.game.action import Action, ActionType, AvailableActions
from pokerbot.simulation.table import PokerTable, SeatStatus
from pokerbot.simulation.tracker import StatTracker, PlayerType
from pokerbot.equity.calculator import EquityCalculator
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot


@dataclass
class GameConfig:
    """Configuration for the game runner."""
    num_seats: int = 6
    small_blind: float = 0.5
    big_blind: float = 1.0
    ante: float = 0.0
    starting_stack: float = 100.0
    show_equity: bool = True
    show_stats: bool = True
    auto_advance: bool = False


class GameRunner:
    """
    Runs interactive poker games with visual display.

    Features:
    - Beautiful ASCII display
    - Human player interaction
    - AI opponents with different styles
    - Real-time statistics
    - Equity calculations
    """

    def __init__(self, config: Optional[GameConfig] = None):
        """Initialize the game runner."""
        self.config = config or GameConfig()

        self.table = PokerTable(
            num_seats=self.config.num_seats,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            ante=self.config.ante,
        )

        self.tracker = StatTracker(num_seats=self.config.num_seats)
        self.equity_calc = EquityCalculator(default_simulations=3000)

        self.human_seat: Optional[int] = None
        self.running = False

        # Display settings
        self.width = 66

    def setup_table(
        self,
        bots: list[BaseBot],
        human_seat: int = 0,
        human_name: str = "Hero",
    ) -> None:
        """
        Set up the table with bots and human player.

        Args:
            bots: List of bots to seat (will fill remaining seats)
            human_seat: Seat index for human player
            human_name: Name for human player
        """
        self.human_seat = human_seat

        # Seat the human
        self.table.seat_player(
            human_seat,
            is_human=True,
            stack=self.config.starting_stack,
            name=human_name,
        )
        self.tracker.register_player(human_seat, human_name)

        # Seat the bots
        bot_idx = 0
        for i in range(self.config.num_seats):
            if i == human_seat:
                continue
            if bot_idx < len(bots):
                bot = bots[bot_idx]
                self.table.seat_player(
                    i,
                    bot=bot,
                    stack=self.config.starting_stack,
                    name=bot.name,
                )
                self.tracker.register_player(i, bot.name)
                bot_idx += 1

    def run(self, num_hands: int = 0) -> None:
        """
        Run the game loop.

        Args:
            num_hands: Number of hands to play (0 = unlimited)
        """
        self.running = True
        hands_played = 0

        print(self._header())
        print("\nWelcome to Pokerbot! Type 'h' for help at any prompt.\n")

        while self.running:
            # Check if we should stop
            if num_hands > 0 and hands_played >= num_hands:
                break

            # Check if human has chips
            if self.human_seat is not None:
                if self.table.seats[self.human_seat].stack <= 0:
                    print("\nYou're out of chips! Game over.")
                    break

            # Start new hand
            if not self.table.start_hand():
                print("\nNot enough players to continue.")
                break

            # Register active players with tracker
            active = [i for i in range(self.config.num_seats) if self.table.in_hand[i]]
            positions = {i: self.table.get_position_name(i) for i in active}
            self.tracker.start_hand(active, positions)

            # Play the hand
            self._play_hand()

            hands_played += 1

            # Ask to continue
            if not self.config.auto_advance:
                response = input("\nPress Enter for next hand (q to quit): ").strip().lower()
                if response == 'q':
                    self.running = False

            # Advance button
            self.table.advance_button()

        print("\n" + self._footer())
        self._show_final_stats()

    def _play_hand(self) -> None:
        """Play a single hand to completion."""
        streets = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]

        for street in streets:
            if self.table.is_hand_over():
                break

            if street != Street.PREFLOP:
                self.table.deal_street(street)

            self._play_betting_round()

        # Showdown and payout
        self._showdown()

    def _play_betting_round(self) -> None:
        """Play a single betting round."""
        first_actor = self.table._get_first_to_act()
        if first_actor == -1:
            return

        current = first_actor
        last_aggressor = -1
        acted = set()

        while True:
            # Skip all-in players (but not folded - we use _next_in_hand_seat)
            if self.table.all_in[current]:
                current = self.table._next_in_hand_seat(current)
                continue

            # Check if betting round is complete
            if current in acted:
                # Check if all players have matched the bet or are folded/all-in
                all_matched = all(
                    self.table.bet_this_round[i] >= self.table.current_bet or
                    not self.table.in_hand[i] or self.table.all_in[i]
                    for i in range(self.config.num_seats)
                )
                if all_matched:
                    break

            # Display game state
            self._display_table(current)

            # Get action
            available = self.table.get_available_actions(current)
            seat = self.table.seats[current]

            if seat.is_human:
                action = self._get_human_action(current, available)
            else:
                action = self._get_bot_action(current, seat.bot, available)
                self._display_action(current, action)

            # Apply action
            self.table.apply_action(current, action)

            # Track stats
            is_facing = self.table.current_bet > 0
            self.tracker.record_action(
                current, action, self.table.street,
                is_facing_bet=is_facing,
                pot_size=self.table.pot,
            )

            # Update aggressor - when someone bets/raises, everyone else needs to act again
            if action.is_aggressive:
                last_aggressor = current
                acted.clear()  # Reset - everyone needs to act again

            acted.add(current)

            # Check if only one player remains
            in_hand = [i for i in range(self.config.num_seats) if self.table.in_hand[i]]
            if len(in_hand) <= 1:
                break

            # Next player still in the hand
            current = self.table._next_in_hand_seat(current)

    def _display_table(self, action_on: int) -> None:
        """Display the current table state."""
        os.system('cls' if os.name == 'nt' else 'clear')

        lines = []
        lines.append(self._header())
        lines.append(f"  HAND #{self.table.hand_number}  |  Pot: ${self.table.pot:.2f}  |  "
                     f"Street: {self.table.street.name}")
        lines.append(self._separator())

        # Board
        if self.table.board:
            board_str = "  ".join(c.pretty for c in self.table.board)
            lines.append(f"  Board: {board_str}")
        else:
            lines.append("  Board: [Preflop]")
        lines.append("")

        # Player table header
        lines.append(f"  {'Pos':<5} {'Name':<12} {'Type':<10} {'Stack':>8} "
                     f"{'Bet':>8} {'Status':<10}")
        lines.append("  " + "-" * 60)

        # Players
        for i, seat in enumerate(self.table.seats):
            if seat.status == SeatStatus.EMPTY:
                continue

            marker = ">" if i == action_on else " "
            pos = self.table.get_position_name(i)
            name = seat.name[:11]
            bot_type = seat.bot_type()
            stack = f"${seat.stack:.2f}"
            bet = f"${self.table.bet_this_round[i]:.2f}" if self.table.bet_this_round[i] > 0 else "-"

            if not self.table.in_hand[i]:
                status = "Folded"
            elif self.table.all_in[i]:
                status = "All-in"
            elif i == action_on:
                status = "Action"
            else:
                status = "Waiting"

            lines.append(f"{marker} {pos:<5} {name:<12} {bot_type:<10} {stack:>8} "
                         f"{bet:>8} {status:<10}")

        lines.append(self._separator())

        # Human's hole cards
        if self.human_seat is not None and self.human_seat in self.table.hole_cards:
            hand = self.table.hole_cards[self.human_seat]
            lines.append(f"  Your Hand: {hand.cards[0].pretty}  {hand.cards[1].pretty}")

            # Show equity if enabled
            if self.config.show_equity and self.table.board:
                equity = self._calculate_equity(hand)
                lines.append(f"  Equity vs range: ~{equity*100:.0f}%")

            # Pot odds if facing bet
            if self.human_seat == action_on:
                to_call = self.table.current_bet - self.table.bet_this_round[self.human_seat]
                if to_call > 0:
                    pot_odds = self.table.pot / (self.table.pot + to_call)
                    required = self.equity_calc.required_equity_to_call(self.table.pot, to_call)
                    lines.append(f"  Pot Odds: {pot_odds*100:.1f}%  |  Need: {required*100:.1f}%")

        lines.append(self._footer())

        print("\n".join(lines))

    def _get_human_action(self, seat_index: int, available: AvailableActions) -> Action:
        """Get action from human player."""
        while True:
            prompt = self._build_action_prompt(available)
            response = input(prompt).strip().lower()

            if response == 'h':
                self._show_help()
                continue
            elif response == 's':
                self._show_stats_for_seat()
                continue
            elif response == 'q':
                self.running = False
                return Action.fold(seat_index)

            action = self._parse_action(response, seat_index, available)
            if action:
                return action

            print("Invalid action. Type 'h' for help.")

    def _build_action_prompt(self, available: AvailableActions) -> str:
        """Build the action prompt string."""
        options = []

        if available.can_fold:
            options.append("[F]old")
        if available.can_check:
            options.append("[C]heck")
        if available.can_call:
            options.append(f"[C]all ${available.call_amount:.2f}")
        if available.can_bet:
            options.append(f"[B]et (min ${available.min_bet:.2f})")
        if available.can_raise:
            options.append(f"[R]aise (${available.min_raise:.2f}-${available.max_raise:.2f})")
        options.append("[A]ll-in")

        return f"\n{' | '.join(options)}\n> "

    def _parse_action(
        self,
        response: str,
        seat_index: int,
        available: AvailableActions,
    ) -> Optional[Action]:
        """Parse user input into an action."""
        if not response:
            return None

        cmd = response.split()[0]
        amount = None

        if len(response.split()) > 1:
            try:
                amount = float(response.split()[1])
            except ValueError:
                pass

        if cmd in ('f', 'fold'):
            if available.can_fold:
                return Action.fold(seat_index)

        elif cmd in ('c', 'check', 'call'):
            if available.can_check:
                return Action.check(seat_index)
            if available.can_call:
                return Action.call(available.call_amount, seat_index)

        elif cmd in ('b', 'bet'):
            if available.can_bet:
                bet_amount = amount if amount else available.min_bet
                bet_amount = max(available.min_bet, bet_amount)
                return Action.bet(bet_amount, seat_index)

        elif cmd in ('r', 'raise'):
            if available.can_raise:
                raise_amount = amount if amount else available.min_raise
                raise_amount = max(available.min_raise, min(raise_amount, available.max_raise))
                return Action.raise_to(raise_amount, seat_index)

        elif cmd in ('a', 'all', 'allin', 'all-in'):
            stack = self.table.seats[seat_index].stack
            return Action.all_in(stack, seat_index)

        return None

    def _get_bot_action(
        self,
        seat_index: int,
        bot: BaseBot,
        available: AvailableActions,
    ) -> Action:
        """Get action from a bot."""
        # Build a simplified game state for the bot
        from pokerbot.game.state import GameState
        from pokerbot.game.player import Player

        # Create minimal game state
        # Map seat indices to player list indices
        players = []
        seat_to_player_idx = {}
        player_idx = 0
        for i, seat in enumerate(self.table.seats):
            if seat.status != SeatStatus.EMPTY:
                player = Player(
                    name=seat.name,
                    stack=seat.stack,
                    bet_this_round=self.table.bet_this_round[i],
                    is_active=self.table.in_hand[i],
                    is_all_in=self.table.all_in[i],
                )
                if i in self.table.hole_cards:
                    player.hole_cards = self.table.hole_cards[i]
                players.append(player)
                seat_to_player_idx[i] = player_idx
                player_idx += 1

        # Convert seat index to player list index
        action_on_player = seat_to_player_idx.get(seat_index, 0)

        game_state = GameState(
            players=players,
            board=list(self.table.board),
            street=self.table.street,
            current_bet=self.table.current_bet,
            min_raise=self.table.min_raise,
            action_on=action_on_player,
            big_blind=self.table.big_blind,
        )
        # Set the pot - critical for bet sizing calculations
        game_state.pot.main_pot = self.table.pot

        # Set bot's hole cards
        if seat_index in self.table.hole_cards:
            bot.set_hole_cards(self.table.hole_cards[seat_index])

        # Get action
        action = bot.get_action(game_state)

        # Validate action
        return self._validate_bot_action(action, seat_index, available)

    def _validate_bot_action(
        self,
        action: Action,
        seat_index: int,
        available: AvailableActions,
    ) -> Action:
        """Validate and fix bot action if needed."""
        if action.action_type == ActionType.FOLD and available.can_check:
            return Action.check(seat_index)

        if action.action_type == ActionType.BET:
            if not available.can_bet:
                if available.can_check:
                    return Action.check(seat_index)
                return Action.fold(seat_index)
            return Action.bet(max(available.min_bet, action.amount), seat_index)

        if action.action_type == ActionType.RAISE:
            if not available.can_raise:
                if available.can_call:
                    return Action.call(available.call_amount, seat_index)
                return Action.fold(seat_index)
            amount = max(available.min_raise, min(action.amount, available.max_raise))
            return Action.raise_to(amount, seat_index)

        if action.action_type == ActionType.CALL:
            if available.can_call:
                return Action.call(available.call_amount, seat_index)
            if available.can_check:
                return Action.check(seat_index)
            return Action.fold(seat_index)

        return action

    def _display_action(self, seat_index: int, action: Action) -> None:
        """Display a bot's action."""
        seat = self.table.seats[seat_index]
        print(f"\n  {seat.name} ({self.table.get_position_name(seat_index)}): {action}")

        if not self.config.auto_advance:
            import time
            time.sleep(0.5)

    def _showdown(self) -> None:
        """Handle showdown and payouts."""
        winners = self.table.get_winners()

        if len(winners) == 0:
            return

        print("\n" + self._separator())
        print("  SHOWDOWN")
        print(self._separator())

        # Show remaining hands
        for seat_idx in self.table.get_players_in_hand():
            if seat_idx in self.table.hole_cards:
                hand = self.table.hole_cards[seat_idx]
                name = self.table.seats[seat_idx].name
                print(f"  {name}: {hand.cards[0].pretty}  {hand.cards[1].pretty}")

        # Announce winners
        print()
        for seat_idx, amount in winners:
            name = self.table.seats[seat_idx].name
            print(f"  {name} wins ${amount:.2f}")

            # Track result
            self.tracker.record_hand_result(
                seat_idx, amount, self.table.total_invested[seat_idx]
            )

        # Payout
        self.table.payout_winners(winners)

        print(self._separator())

    def _calculate_equity(self, hand: Hand) -> float:
        """Calculate approximate equity vs a range."""
        if not self.table.board:
            return get_hand_percentile(hand) / 100

        # Simplified equity calculation
        try:
            villain_range = top_percent_range(30)  # Assume 30% range
            result = self.equity_calc.hand_vs_range(
                str(hand),
                str(villain_range),
                " ".join(str(c) for c in self.table.board),
            )
            return result.equity
        except Exception:
            return 0.5

    def _show_help(self) -> None:
        """Show help information."""
        print("\n" + "=" * 40)
        print("  COMMANDS")
        print("=" * 40)
        print("  f, fold     - Fold your hand")
        print("  c, check    - Check (if no bet)")
        print("  c, call     - Call the current bet")
        print("  b, bet X    - Bet amount X")
        print("  r, raise X  - Raise to amount X")
        print("  a, all-in   - Go all-in")
        print()
        print("  h           - Show this help")
        print("  s           - Show player stats")
        print("  q           - Quit the game")
        print("=" * 40)

    def _show_stats_for_seat(self) -> None:
        """Show stats for all players."""
        print("\n" + "=" * 50)
        print("  PLAYER STATISTICS")
        print("=" * 50)

        for i, seat in enumerate(self.table.seats):
            if seat.status == SeatStatus.EMPTY:
                continue

            stats = self.tracker.get_stats(i)
            if stats and stats.hands_played > 0:
                player_type = stats.classify()
                print(f"\n  {seat.name} ({seat.bot_type()}):")
                print(f"    Type: {player_type.name}")
                print(f"    Hands: {stats.hands_played}")
                print(f"    VPIP: {stats.vpip:.1f}% | PFR: {stats.pfr:.1f}%")
                print(f"    AF: {stats.aggression_factor:.2f}")
                profit = stats.total_won - stats.total_invested
                print(f"    Profit: ${profit:+.2f}")

        print("=" * 50)

    def _show_final_stats(self) -> None:
        """Show final session statistics."""
        print("\n" + "=" * 50)
        print("  FINAL RESULTS")
        print("=" * 50)

        results = []
        for i, seat in enumerate(self.table.seats):
            if seat.status == SeatStatus.EMPTY:
                continue

            profit = seat.stack - self.config.starting_stack
            results.append((seat.name, seat.stack, profit))

        # Sort by profit
        results.sort(key=lambda x: x[2], reverse=True)

        for name, stack, profit in results:
            sign = "+" if profit >= 0 else ""
            print(f"  {name:<15} ${stack:>8.2f}  ({sign}{profit:.2f})")

        print("=" * 50)

    def _header(self) -> str:
        """Generate header line."""
        return "+" + "=" * (self.width - 2) + "+"

    def _footer(self) -> str:
        """Generate footer line."""
        return "+" + "=" * (self.width - 2) + "+"

    def _separator(self) -> str:
        """Generate separator line."""
        return "+" + "-" * (self.width - 2) + "+"


def create_default_game() -> GameRunner:
    """Create a game with default settings and mixed bot opponents."""
    from pokerbot.bot.nit_bot import NitBot
    from pokerbot.bot.fish_bot import FishBot
    from pokerbot.bot.lag_bot import LAGBot
    from pokerbot.bot.maniac_bot import ManiacBot
    from pokerbot.bot.calling_station import CallingStationBot
    from pokerbot.bot.gto_bot import GTOBot

    config = GameConfig(
        num_seats=6,
        small_blind=0.5,
        big_blind=1.0,
        starting_stack=100.0,
        show_equity=True,
        show_stats=True,
    )

    runner = GameRunner(config)

    # Create diverse table
    bots = [
        NitBot(),
        FishBot(),
        LAGBot(),
        ManiacBot(),
        CallingStationBot(),
    ]

    runner.setup_table(bots, human_seat=0, human_name="Hero")

    return runner
