"""Real-time player statistics tracking for poker simulation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from pokerbot.game.action import Action, ActionType
from pokerbot.game.state import Street


class PlayerType(Enum):
    """Classification of player types based on observed behavior."""
    UNKNOWN = auto()
    NIT = auto()           # Very tight, rarely plays
    ROCK = auto()          # Tight, straightforward
    TAG = auto()           # Tight-aggressive
    LAG = auto()           # Loose-aggressive
    FISH = auto()          # Loose-passive
    MANIAC = auto()        # Hyper-aggressive
    CALLING_STATION = auto()  # Calls everything

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class PlayerStats:
    """
    Statistics for a single player.

    Tracks all key poker metrics for player classification.
    """
    # Identification
    seat_index: int
    name: str = ""

    # Hand counts
    hands_played: int = 0
    hands_won: int = 0

    # Preflop stats
    vpip_hands: int = 0      # Voluntarily put $ in pot
    pfr_hands: int = 0       # Preflop raise
    three_bet_opportunities: int = 0
    three_bet_made: int = 0
    fold_to_three_bet_opportunities: int = 0
    fold_to_three_bet: int = 0

    # Postflop stats
    cbet_opportunities: int = 0
    cbet_made: int = 0
    fold_to_cbet_opportunities: int = 0
    fold_to_cbet: int = 0

    # Aggression tracking
    bets_made: int = 0
    raises_made: int = 0
    calls_made: int = 0
    checks_made: int = 0
    folds_made: int = 0

    # Showdown stats
    went_to_showdown: int = 0
    won_at_showdown: int = 0

    # Money stats
    total_won: float = 0.0
    total_invested: float = 0.0
    biggest_pot_won: float = 0.0

    # Position stats
    hands_by_position: dict[str, int] = field(default_factory=dict)
    vpip_by_position: dict[str, int] = field(default_factory=dict)

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot percentage."""
        if self.hands_played == 0:
            return 0.0
        return (self.vpip_hands / self.hands_played) * 100

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise percentage."""
        if self.hands_played == 0:
            return 0.0
        return (self.pfr_hands / self.hands_played) * 100

    @property
    def three_bet_pct(self) -> float:
        """3-bet percentage."""
        if self.three_bet_opportunities == 0:
            return 0.0
        return (self.three_bet_made / self.three_bet_opportunities) * 100

    @property
    def fold_to_three_bet_pct(self) -> float:
        """Fold to 3-bet percentage."""
        if self.fold_to_three_bet_opportunities == 0:
            return 0.0
        return (self.fold_to_three_bet / self.fold_to_three_bet_opportunities) * 100

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        if self.cbet_opportunities == 0:
            return 0.0
        return (self.cbet_made / self.cbet_opportunities) * 100

    @property
    def fold_to_cbet_pct(self) -> float:
        """Fold to c-bet percentage."""
        if self.fold_to_cbet_opportunities == 0:
            return 0.0
        return (self.fold_to_cbet / self.fold_to_cbet_opportunities) * 100

    @property
    def aggression_factor(self) -> float:
        """Aggression Factor (AF) = (bets + raises) / calls."""
        if self.calls_made == 0:
            return float('inf') if (self.bets_made + self.raises_made) > 0 else 0.0
        return (self.bets_made + self.raises_made) / self.calls_made

    @property
    def aggression_frequency(self) -> float:
        """Aggression Frequency = (bets + raises) / total actions."""
        total = self.bets_made + self.raises_made + self.calls_made + self.checks_made
        if total == 0:
            return 0.0
        return ((self.bets_made + self.raises_made) / total) * 100

    @property
    def wtsd(self) -> float:
        """Went To ShowDown percentage."""
        if self.vpip_hands == 0:
            return 0.0
        return (self.went_to_showdown / self.vpip_hands) * 100

    @property
    def wsd(self) -> float:
        """Won at ShowDown percentage."""
        if self.went_to_showdown == 0:
            return 0.0
        return (self.won_at_showdown / self.went_to_showdown) * 100

    @property
    def winrate_bb_100(self) -> float:
        """Winrate in big blinds per 100 hands."""
        if self.hands_played == 0:
            return 0.0
        # Assuming 1BB = 1.0 for simplicity
        return ((self.total_won - self.total_invested) / self.hands_played) * 100

    def classify(self) -> PlayerType:
        """
        Classify player type based on statistics.

        Uses VPIP/PFR/AF to determine player style.
        """
        if self.hands_played < 10:
            return PlayerType.UNKNOWN

        vpip = self.vpip
        pfr = self.pfr
        af = self.aggression_factor

        # Classification thresholds
        if vpip < 12:
            return PlayerType.NIT
        elif vpip < 18:
            if pfr > vpip * 0.7:
                return PlayerType.TAG
            return PlayerType.ROCK
        elif vpip < 28:
            if pfr > vpip * 0.7 and af > 2.5:
                return PlayerType.TAG
            elif pfr > vpip * 0.7:
                return PlayerType.LAG
            return PlayerType.FISH
        elif vpip < 45:
            if pfr > vpip * 0.6 and af > 3.0:
                return PlayerType.LAG
            elif af < 1.0:
                return PlayerType.CALLING_STATION
            return PlayerType.FISH
        else:
            if pfr > vpip * 0.5 and af > 3.5:
                return PlayerType.MANIAC
            elif af < 0.8:
                return PlayerType.CALLING_STATION
            return PlayerType.FISH

    def get_summary(self) -> str:
        """Get a formatted summary of player stats."""
        player_type = self.classify()
        lines = [
            f"Player: {self.name} (Seat {self.seat_index})",
            f"Type: {player_type}",
            f"Hands: {self.hands_played}",
            f"",
            f"VPIP: {self.vpip:.1f}%",
            f"PFR: {self.pfr:.1f}%",
            f"3-Bet: {self.three_bet_pct:.1f}%",
            f"",
            f"AF: {self.aggression_factor:.2f}",
            f"Agg%: {self.aggression_frequency:.1f}%",
            f"C-Bet: {self.cbet_pct:.1f}%",
            f"",
            f"WTSD: {self.wtsd:.1f}%",
            f"W$SD: {self.wsd:.1f}%",
            f"",
            f"Profit: ${self.total_won - self.total_invested:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class HandState:
    """Tracks the state of actions within a single hand for stat calculation."""
    preflop_raiser: Optional[int] = None
    has_bet_preflop: dict[int, bool] = field(default_factory=dict)
    is_facing_raise: dict[int, bool] = field(default_factory=dict)
    flop_aggressor: Optional[int] = None
    has_vpip: dict[int, bool] = field(default_factory=dict)


class StatTracker:
    """
    Tracks player statistics across a poker session.

    Records actions, calculates statistics, and classifies players.
    """

    def __init__(self, num_seats: int = 6):
        """
        Initialize the stat tracker.

        Args:
            num_seats: Number of seats at the table
        """
        self.num_seats = num_seats
        self.player_stats: dict[int, PlayerStats] = {}
        self.hand_state: HandState = HandState()
        self.current_street: Street = Street.PREFLOP

    def register_player(self, seat_index: int, name: str) -> None:
        """Register a player for tracking."""
        if seat_index not in self.player_stats:
            self.player_stats[seat_index] = PlayerStats(
                seat_index=seat_index,
                name=name,
            )
        else:
            self.player_stats[seat_index].name = name

    def start_hand(self, active_seats: list[int], positions: dict[int, str]) -> None:
        """
        Start tracking a new hand.

        Args:
            active_seats: List of seat indices in the hand
            positions: Mapping of seat index to position name
        """
        self.hand_state = HandState()
        self.current_street = Street.PREFLOP

        for seat_idx in active_seats:
            if seat_idx in self.player_stats:
                self.player_stats[seat_idx].hands_played += 1
                pos = positions.get(seat_idx, "?")
                self.player_stats[seat_idx].hands_by_position[pos] = \
                    self.player_stats[seat_idx].hands_by_position.get(pos, 0) + 1

    def record_action(
        self,
        seat_index: int,
        action: Action,
        street: Street,
        is_facing_bet: bool = False,
        bet_amount: float = 0.0,
        pot_size: float = 0.0,
    ) -> None:
        """
        Record a player action for statistics.

        Args:
            seat_index: Index of acting player
            action: The action taken
            street: Current betting street
            is_facing_bet: Whether player was facing a bet/raise
            bet_amount: Size of the bet faced
            pot_size: Current pot size
        """
        if seat_index not in self.player_stats:
            return

        stats = self.player_stats[seat_index]
        self.current_street = street

        # Track basic action counts
        if action.action_type == ActionType.FOLD:
            stats.folds_made += 1
        elif action.action_type == ActionType.CHECK:
            stats.checks_made += 1
        elif action.action_type == ActionType.CALL:
            stats.calls_made += 1
        elif action.action_type == ActionType.BET:
            stats.bets_made += 1
        elif action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            if is_facing_bet:
                stats.raises_made += 1
            else:
                stats.bets_made += 1

        # Preflop stats
        if street == Street.PREFLOP:
            self._record_preflop_action(seat_index, action, is_facing_bet)
        else:
            self._record_postflop_action(seat_index, action, street, is_facing_bet)

    def _record_preflop_action(
        self,
        seat_index: int,
        action: Action,
        is_facing_bet: bool,
    ) -> None:
        """Record preflop-specific statistics."""
        stats = self.player_stats[seat_index]

        # VPIP - voluntarily putting money in pot
        if action.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            if not self.hand_state.has_vpip.get(seat_index, False):
                stats.vpip_hands += 1
                self.hand_state.has_vpip[seat_index] = True

        # PFR - preflop raise
        if action.action_type in (ActionType.RAISE, ActionType.ALL_IN) and action.is_aggressive:
            if self.hand_state.preflop_raiser is None:
                # This is an open raise
                stats.pfr_hands += 1
                self.hand_state.preflop_raiser = seat_index
            else:
                # This is a 3-bet or higher
                stats.three_bet_made += 1

        # 3-bet opportunities
        if self.hand_state.preflop_raiser is not None and seat_index != self.hand_state.preflop_raiser:
            if not self.hand_state.is_facing_raise.get(seat_index, False):
                stats.three_bet_opportunities += 1
                self.hand_state.is_facing_raise[seat_index] = True

                # Fold to 3-bet (if facing a 3-bet and folding)
                if action.action_type == ActionType.FOLD:
                    # Check if this is fold to a 3-bet
                    pass  # Simplified for now

    def _record_postflop_action(
        self,
        seat_index: int,
        action: Action,
        street: Street,
        is_facing_bet: bool,
    ) -> None:
        """Record postflop-specific statistics."""
        stats = self.player_stats[seat_index]

        # C-bet tracking on flop
        if street == Street.FLOP:
            # If this player was the preflop raiser
            if seat_index == self.hand_state.preflop_raiser:
                if self.hand_state.flop_aggressor is None:
                    # Opportunity to c-bet
                    stats.cbet_opportunities += 1
                    if action.action_type in (ActionType.BET, ActionType.ALL_IN):
                        stats.cbet_made += 1
                        self.hand_state.flop_aggressor = seat_index

            # Fold to c-bet
            if is_facing_bet and self.hand_state.flop_aggressor == self.hand_state.preflop_raiser:
                stats.fold_to_cbet_opportunities += 1
                if action.action_type == ActionType.FOLD:
                    stats.fold_to_cbet += 1

    def record_showdown(
        self,
        seat_indices: list[int],
        winner_indices: list[int],
        amounts: dict[int, float],
    ) -> None:
        """
        Record showdown results.

        Args:
            seat_indices: Players who went to showdown
            winner_indices: Players who won
            amounts: Mapping of seat index to amount won
        """
        for seat_idx in seat_indices:
            if seat_idx in self.player_stats:
                stats = self.player_stats[seat_idx]
                stats.went_to_showdown += 1
                if seat_idx in winner_indices:
                    stats.won_at_showdown += 1

    def record_hand_result(
        self,
        seat_index: int,
        amount_won: float,
        amount_invested: float,
    ) -> None:
        """
        Record the result of a hand for a player.

        Args:
            seat_index: Player's seat index
            amount_won: Total amount won in the hand
            amount_invested: Total amount invested in the hand
        """
        if seat_index in self.player_stats:
            stats = self.player_stats[seat_index]
            stats.total_won += amount_won
            stats.total_invested += amount_invested
            if amount_won > 0:
                stats.hands_won += 1
                if amount_won > stats.biggest_pot_won:
                    stats.biggest_pot_won = amount_won

    def get_stats(self, seat_index: int) -> Optional[PlayerStats]:
        """Get statistics for a specific player."""
        return self.player_stats.get(seat_index)

    def get_player_type(self, seat_index: int) -> PlayerType:
        """Get the classified player type for a seat."""
        stats = self.player_stats.get(seat_index)
        if stats:
            return stats.classify()
        return PlayerType.UNKNOWN

    def get_all_stats(self) -> dict[int, PlayerStats]:
        """Get statistics for all players."""
        return self.player_stats

    def get_stats_display(self, seat_index: int) -> str:
        """Get a compact stats display for a player."""
        stats = self.player_stats.get(seat_index)
        if not stats or stats.hands_played == 0:
            return "No data"

        player_type = stats.classify()
        return (
            f"{player_type.name}: "
            f"VPIP {stats.vpip:.0f}% / "
            f"PFR {stats.pfr:.0f}% / "
            f"AF {stats.aggression_factor:.1f}"
        )

    def get_hud_display(self, seat_index: int) -> dict[str, str]:
        """
        Get HUD-style statistics display.

        Returns:
            Dictionary with stat names and formatted values
        """
        stats = self.player_stats.get(seat_index)
        if not stats or stats.hands_played == 0:
            return {"Hands": "0"}

        return {
            "Type": stats.classify().name,
            "Hands": str(stats.hands_played),
            "VPIP": f"{stats.vpip:.1f}%",
            "PFR": f"{stats.pfr:.1f}%",
            "3-Bet": f"{stats.three_bet_pct:.1f}%",
            "AF": f"{stats.aggression_factor:.1f}",
            "C-Bet": f"{stats.cbet_pct:.1f}%",
            "WTSD": f"{stats.wtsd:.1f}%",
            "W$SD": f"{stats.wsd:.1f}%",
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.player_stats.clear()
        self.hand_state = HandState()
