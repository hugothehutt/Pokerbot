"""Opponent modeling for exploitative play."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from pokerbot.game.player import PlayerStats
from pokerbot.equity.range import HandRange


class PlayerType(Enum):
    """Classification of opponent playing styles."""
    UNKNOWN = auto()
    NIT = auto()        # Very tight, rarely bluffs
    ROCK = auto()       # Tight, solid player
    TAG = auto()        # Tight-aggressive
    LAG = auto()        # Loose-aggressive
    FISH = auto()       # Loose-passive
    MANIAC = auto()     # Very loose-aggressive
    CALLING_STATION = auto()  # Calls too much


@dataclass
class OpponentProfile:
    """
    Profile of an opponent's tendencies.

    Used to adjust strategy for exploitative play.
    """
    player_type: PlayerType = PlayerType.UNKNOWN
    stats: PlayerStats = field(default_factory=PlayerStats)

    # Estimated ranges by situation
    open_range: Optional[HandRange] = None
    three_bet_range: Optional[HandRange] = None
    call_range: Optional[HandRange] = None
    cbet_range: Optional[HandRange] = None

    # Tendencies (0-1 scale)
    aggression: float = 0.5  # How aggressive postflop
    bluff_frequency: float = 0.3  # How often they bluff
    fold_to_aggression: float = 0.5  # How often they fold to bets/raises
    slowplay_frequency: float = 0.1  # How often they trap

    # Notes
    notes: list[str] = field(default_factory=list)

    def update_from_stats(self) -> None:
        """Update tendencies based on stats."""
        if self.stats.hands_played < 10:
            return

        vpip = self.stats.vpip
        pfr = self.stats.pfr
        af = self.stats.aggression_factor

        # Classify player type
        if vpip < 15 and pfr < 10:
            self.player_type = PlayerType.NIT
        elif vpip < 20 and pfr < 15:
            self.player_type = PlayerType.ROCK
        elif vpip < 25 and pfr > 18:
            self.player_type = PlayerType.TAG
        elif vpip > 30 and pfr > 25:
            self.player_type = PlayerType.LAG
        elif vpip > 40 and pfr > 35:
            self.player_type = PlayerType.MANIAC
        elif vpip > 40 and pfr < 20:
            self.player_type = PlayerType.FISH
        elif vpip > 35 and af < 1:
            self.player_type = PlayerType.CALLING_STATION

        # Update tendencies
        self.aggression = min(1.0, af / 3) if af != float('inf') else 0.8
        self.bluff_frequency = max(0.1, min(0.6, (pfr - 10) / 30))
        self.fold_to_aggression = self.stats.fold_to_cbet_pct / 100 if self.stats.fold_to_cbet_opportunities > 0 else 0.5

    def get_adjusted_range(self, base_range: HandRange, action: str) -> HandRange:
        """
        Adjust a GTO range based on opponent tendencies.

        Args:
            base_range: GTO starting range
            action: Type of action ('open', '3bet', 'call', etc.)

        Returns:
            Adjusted range for this opponent
        """
        if self.player_type == PlayerType.UNKNOWN:
            return base_range

        # Adjust based on player type
        adjustment_factor = {
            PlayerType.NIT: 0.6,
            PlayerType.ROCK: 0.8,
            PlayerType.TAG: 1.0,
            PlayerType.LAG: 1.3,
            PlayerType.FISH: 1.5,
            PlayerType.MANIAC: 1.8,
            PlayerType.CALLING_STATION: 1.4,
        }.get(self.player_type, 1.0)

        # For now, return base range
        # In a full implementation, we'd expand/contract the range
        return base_range


class OpponentModel:
    """
    Models opponent behavior for exploitative adjustments.

    Tracks stats and provides exploitative recommendations.
    """

    def __init__(self):
        self.profiles: dict[str, OpponentProfile] = {}

    def get_profile(self, player_name: str) -> OpponentProfile:
        """Get or create profile for a player."""
        if player_name not in self.profiles:
            self.profiles[player_name] = OpponentProfile()
        return self.profiles[player_name]

    def update_stats(self, player_name: str, stats: PlayerStats) -> None:
        """Update stats for a player."""
        profile = self.get_profile(player_name)
        profile.stats = stats
        profile.update_from_stats()

    def estimate_range(
        self,
        player_name: str,
        situation: str,
        default_range: HandRange,
    ) -> HandRange:
        """
        Estimate opponent's range in a situation.

        Args:
            player_name: Player to estimate
            situation: Description of situation
            default_range: GTO default range

        Returns:
            Estimated range based on opponent profile
        """
        profile = self.get_profile(player_name)
        return profile.get_adjusted_range(default_range, situation)

    def get_exploit_recommendation(
        self,
        player_name: str,
        situation: str,
    ) -> str:
        """
        Get exploitative recommendation vs opponent.

        Args:
            player_name: Opponent name
            situation: Current situation

        Returns:
            Strategy recommendation
        """
        profile = self.get_profile(player_name)
        ptype = profile.player_type

        recommendations = {
            PlayerType.NIT: "Respect their bets. Steal blinds often. Don't bluff into them.",
            PlayerType.ROCK: "Value bet thin. Don't bluff often. Steal in position.",
            PlayerType.TAG: "Standard GTO approach. Look for small edges.",
            PlayerType.LAG: "Tighten up preflop. Trap with strong hands. Call down lighter.",
            PlayerType.FISH: "Value bet relentlessly. Don't bluff. Isolate preflop.",
            PlayerType.MANIAC: "Let them hang themselves. Call with medium strength. Trap big hands.",
            PlayerType.CALLING_STATION: "Never bluff. Value bet thin and often. Don't fold to their bets.",
            PlayerType.UNKNOWN: "Use standard GTO approach until you have more information.",
        }

        return recommendations.get(ptype, "Use balanced strategy.")

    def get_fold_equity_estimate(
        self,
        player_name: str,
        bet_size_pct: float,
    ) -> float:
        """
        Estimate fold equity against an opponent.

        Args:
            player_name: Opponent name
            bet_size_pct: Bet size as % of pot

        Returns:
            Estimated fold equity (0-1)
        """
        profile = self.get_profile(player_name)

        # Base fold equity on tendencies
        base_fold = profile.fold_to_aggression

        # Adjust for bet size
        # Larger bets = more folds
        size_adjustment = (bet_size_pct / 100) * 0.2

        # Adjust for player type
        type_adjustment = {
            PlayerType.NIT: 0.2,
            PlayerType.ROCK: 0.1,
            PlayerType.TAG: 0.0,
            PlayerType.LAG: -0.1,
            PlayerType.FISH: -0.15,
            PlayerType.MANIAC: -0.2,
            PlayerType.CALLING_STATION: -0.3,
            PlayerType.UNKNOWN: 0.0,
        }.get(profile.player_type, 0.0)

        return max(0.0, min(1.0, base_fold + size_adjustment + type_adjustment))

    def get_bluff_frequency_estimate(self, player_name: str) -> float:
        """Estimate how often opponent is bluffing."""
        profile = self.get_profile(player_name)

        type_bluff_freq = {
            PlayerType.NIT: 0.1,
            PlayerType.ROCK: 0.2,
            PlayerType.TAG: 0.33,
            PlayerType.LAG: 0.45,
            PlayerType.FISH: 0.25,
            PlayerType.MANIAC: 0.55,
            PlayerType.CALLING_STATION: 0.2,
            PlayerType.UNKNOWN: 0.33,
        }

        return type_bluff_freq.get(profile.player_type, 0.33)
