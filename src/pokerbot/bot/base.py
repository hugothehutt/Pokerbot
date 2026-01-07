"""Base bot interface for poker bots."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState
from pokerbot.game.action import Action
from pokerbot.game.player import Player


@dataclass
class BotConfig:
    """Configuration for a poker bot."""

    # Identification
    name: str = "Bot"

    # Monte Carlo settings
    simulations: int = 5000
    high_accuracy_simulations: int = 50000

    # Strategy settings
    aggression: float = 1.0  # Multiplier for aggression (1.0 = balanced)
    tightness: float = 1.0   # Multiplier for hand selection (1.0 = GTO)

    # Opponent modeling
    use_opponent_modeling: bool = True
    default_opponent_vpip: float = 25.0  # Assumed VPIP if unknown

    # Decision settings
    use_randomization: bool = True  # Mix strategies
    verbose: bool = False  # Print decision details


@dataclass
class BotStats:
    """Statistics tracked by the bot."""
    hands_played: int = 0
    hands_won: int = 0
    total_profit: float = 0.0

    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bet_hands: int = 0
    cbet_hands: int = 0

    biggest_pot_won: float = 0.0
    biggest_pot_lost: float = 0.0

    @property
    def vpip(self) -> float:
        """Voluntarily put money in pot percentage."""
        return (self.vpip_hands / self.hands_played * 100) if self.hands_played > 0 else 0

    @property
    def pfr(self) -> float:
        """Preflop raise percentage."""
        return (self.pfr_hands / self.hands_played * 100) if self.hands_played > 0 else 0

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        return (self.hands_won / self.hands_played * 100) if self.hands_played > 0 else 0

    @property
    def bb_per_100(self) -> float:
        """Big blinds won per 100 hands."""
        if self.hands_played == 0:
            return 0.0
        return (self.total_profit / self.hands_played) * 100


class BaseBot(ABC):
    """
    Abstract base class for poker bots.

    Subclasses must implement the get_action method.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the bot with configuration."""
        self.config = config or BotConfig()
        self.stats = BotStats()
        self._hole_cards: Optional[Hand] = None
        self._current_game: Optional[GameState] = None

    @property
    def name(self) -> str:
        """Get bot name."""
        return self.config.name

    @property
    def hole_cards(self) -> Optional[Hand]:
        """Get current hole cards."""
        return self._hole_cards

    def set_hole_cards(self, cards: Hand) -> None:
        """Set hole cards for current hand."""
        self._hole_cards = cards

    def set_game_state(self, state: GameState) -> None:
        """Set current game state."""
        self._current_game = state

    @abstractmethod
    def get_action(self, game_state: GameState) -> Action:
        """
        Get the bot's action for the current game state.

        Args:
            game_state: Current state of the game

        Returns:
            Action to take
        """
        pass

    def on_hand_start(self, hole_cards: Hand, game_state: GameState) -> None:
        """
        Called at the start of each hand.

        Args:
            hole_cards: Bot's hole cards
            game_state: Initial game state
        """
        self._hole_cards = hole_cards
        self._current_game = game_state
        self.stats.hands_played += 1

    def on_hand_end(self, game_state: GameState, won: bool, profit: float) -> None:
        """
        Called at the end of each hand.

        Args:
            game_state: Final game state
            won: Whether bot won the hand
            profit: Profit/loss for the hand
        """
        if won:
            self.stats.hands_won += 1
            if profit > self.stats.biggest_pot_won:
                self.stats.biggest_pot_won = profit
        else:
            if abs(profit) > self.stats.biggest_pot_lost:
                self.stats.biggest_pot_lost = abs(profit)

        self.stats.total_profit += profit
        self._hole_cards = None

    def on_action_taken(self, action: Action) -> None:
        """
        Called after bot takes an action.

        Updates internal statistics.
        """
        from pokerbot.game.action import ActionType
        from pokerbot.game.state import Street

        if self._current_game and self._current_game.street == Street.PREFLOP:
            if action.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                self.stats.vpip_hands += 1
            if action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                self.stats.pfr_hands += 1

    def reset_stats(self) -> None:
        """Reset bot statistics."""
        self.stats = BotStats()

    def get_stats_summary(self) -> str:
        """Get a summary of bot statistics."""
        return (
            f"Bot: {self.name}\n"
            f"Hands: {self.stats.hands_played}\n"
            f"Win Rate: {self.stats.win_rate:.1f}%\n"
            f"Profit: {self.stats.total_profit:.2f}\n"
            f"BB/100: {self.stats.bb_per_100:.2f}\n"
            f"VPIP: {self.stats.vpip:.1f}%\n"
            f"PFR: {self.stats.pfr:.1f}%"
        )

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
