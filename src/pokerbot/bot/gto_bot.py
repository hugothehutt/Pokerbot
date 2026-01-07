"""GTO-inspired poker bot implementation."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.game.player import Position
from pokerbot.equity.range import HandRange
from pokerbot.strategy.decision import DecisionEngine, DecisionContext, Decision, Confidence
from pokerbot.strategy.opponent import OpponentModel
from pokerbot.bot.base import BaseBot, BotConfig


class GTOBot(BaseBot):
    """
    A poker bot that uses GTO-inspired strategy.

    Features:
    - Position-aware preflop ranges
    - Monte Carlo equity calculations
    - GTO bet sizing
    - Optional opponent modeling for exploitative adjustments
    - Strategy randomization for balanced play
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the GTO bot."""
        if config is None:
            config = BotConfig(name="GTO")
        
        super().__init__(config)

        self.decision_engine = DecisionEngine(
            simulations=self.config.simulations,
            use_opponent_modeling=self.config.use_opponent_modeling,
        )

        self.opponent_model = OpponentModel() if self.config.use_opponent_modeling else None
        self._rng = random.Random()

        # Villain range estimates by street
        self._villain_range: Optional[HandRange] = None
        self._last_decision: Optional[Decision] = None

    def get_action(self, game_state: GameState) -> Action:
        """
        Get the bot's action for the current game state.

        Args:
            game_state: Current state of the game

        Returns:
            Action to take
        """
        self._current_game = game_state

        # Get our position and available actions
        player_idx = game_state.action_on
        player = game_state.players[player_idx]
        position = player.position
        available = game_state.get_available_actions()

        # Ensure we have hole cards
        if self._hole_cards is None:
            self._hole_cards = player.hole_cards

        if self._hole_cards is None:
            # No cards - fold
            return Action.fold(player_idx)

        # Build decision context
        context = DecisionContext(
            game_state=game_state,
            hero_hand=self._hole_cards,
            hero_position=position,
            villain_range=self._get_villain_range(game_state),
            opponent_profile=None,  # Could be enhanced with opponent modeling
        )

        # Get decision from engine
        decision = self.decision_engine.make_decision(context)
        self._last_decision = decision

        # Apply randomization if enabled
        action = self._apply_randomization(decision, available)

        # Log if verbose
        if self.config.verbose:
            self._log_decision(decision, action)

        # Track action
        self.on_action_taken(action)

        return action

    def _get_villain_range(self, game_state: GameState) -> HandRange:
        """Estimate villain's range based on action."""
        street = game_state.street

        if street == Street.PREFLOP:
            # Default wide range, narrow based on action
            if game_state.current_bet > game_state.big_blind * 3:
                # Facing 3bet+ - tighter range
                return HandRange("JJ+,AKs,AQs,AKo")
            elif game_state.current_bet > game_state.big_blind:
                # Facing open - standard range
                return HandRange("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,65s,54s,A9o+,KTo+,QTo+,JTo")
            else:
                # No raise - very wide
                return HandRange("22+,A2s+,K2s+,Q2s+,J5s+,T6s+,96s+,86s+,76s,65s,54s,A2o+,K5o+,Q7o+,J8o+,T8o+,98o")
        else:
            # Postflop - use previous estimate or default
            if self._villain_range:
                return self._villain_range
            return HandRange("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,A9o+,KTo+,QTo+,JTo")

    def _apply_randomization(self, decision: Decision, available) -> Action:
        """
        Apply randomization to maintain balanced strategy.

        Occasionally takes alternative actions based on confidence level.
        """
        if not self.config.use_randomization:
            return self._validate_action(decision.action, available)

        action = decision.action

        # Higher confidence = less randomization
        rand_threshold = {
            Confidence.HIGH: 0.05,
            Confidence.MEDIUM: 0.15,
            Confidence.LOW: 0.25,
        }.get(decision.confidence, 0.1)

        if self._rng.random() < rand_threshold and decision.alternatives:
            # Pick random alternative
            alt_action, _ = self._rng.choice(decision.alternatives)
            action = alt_action

        return self._validate_action(action, available)

    def _validate_action(self, action: Action, available) -> Action:
        """Validate and adjust action to be legal."""
        action_type = action.action_type
        amount = action.amount
        player_idx = self._current_game.action_on if self._current_game else -1

        # Check if action is available
        if action_type == ActionType.FOLD:
            if available.can_fold:
                return Action.fold(player_idx)
            elif available.can_check:
                return Action.check(player_idx)
            return Action.fold(player_idx)

        elif action_type == ActionType.CHECK:
            if available.can_check:
                return Action.check(player_idx)
            elif available.can_fold:
                return Action.fold(player_idx)
            return Action.fold(player_idx)

        elif action_type == ActionType.CALL:
            if available.can_call:
                return Action.call(available.call_amount, player_idx)
            elif available.can_check:
                return Action.check(player_idx)
            return Action.fold(player_idx)

        elif action_type == ActionType.BET:
            if available.can_bet:
                bet_amount = max(available.min_bet, amount)
                return Action.bet(bet_amount, player_idx)
            elif available.can_check:
                return Action.check(player_idx)
            return Action.fold(player_idx)

        elif action_type == ActionType.RAISE:
            if available.can_raise:
                raise_amount = max(available.min_raise, min(amount, available.max_raise))
                return Action.raise_to(raise_amount, player_idx)
            elif available.can_call:
                return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        elif action_type == ActionType.ALL_IN:
            if self._current_game:
                player = self._current_game.players[player_idx]
                return Action.all_in(player.stack, player_idx)
            return Action.fold(player_idx)

        return Action.fold(player_idx)

    def _log_decision(self, decision: Decision, action: Action) -> None:
        """Log decision details."""
        print(f"\n{'='*50}")
        print(f"Hand: {self._hole_cards}")
        if self._current_game:
            print(f"Board: {' '.join(str(c) for c in self._current_game.board)}")
            print(f"Pot: {self._current_game.pot.total:.2f}")
        print(f"Decision: {decision}")
        print(f"Action: {action}")
        print(f"{'='*50}\n")

    def analyze_hand(
        self,
        hole_cards: str | Hand,
        board: str = "",
        villain_range: str = "22+,A2s+,K9s+,Q9s+,J9s+,A9o+,KTo+",
        pot: float = 10.0,
        to_call: float = 0.0,
    ) -> dict:
        """
        Analyze a specific hand situation.

        Useful for studying specific spots.

        Args:
            hole_cards: Hero's hand
            board: Community cards (space-separated)
            villain_range: Villain's estimated range
            pot: Current pot size
            to_call: Amount to call

        Returns:
            Analysis dictionary with recommendations
        """
        if isinstance(hole_cards, str):
            hole_cards = Hand(hole_cards)

        return self.decision_engine.analyze_spot(
            hand=hole_cards,
            board=board,
            villain_range=HandRange(villain_range),
            pot=pot,
            to_call=to_call,
        )

    def get_preflop_action(
        self,
        hole_cards: str | Hand,
        position: str | Position,
        facing_raise: bool = False,
        raise_size: float = 0.0,
    ) -> str:
        """
        Get preflop action recommendation.

        Args:
            hole_cards: Hero's hand
            position: Table position
            facing_raise: Whether facing a raise
            raise_size: Size of raise facing (in BBs)

        Returns:
            Action recommendation string
        """
        if isinstance(hole_cards, str):
            hole_cards = Hand(hole_cards)
        if isinstance(position, str):
            position = Position[position.upper()]

        from pokerbot.strategy.gto.ranges import get_opening_range, get_3bet_range, get_calling_range

        hand_notation = hole_cards.notation()

        if not facing_raise:
            # Check if in opening range
            opening_range = get_opening_range(position)
            if hole_cards in opening_range:
                return f"RAISE - {hand_notation} is in opening range for {position.name}"
            else:
                return f"FOLD - {hand_notation} is not in opening range for {position.name}"
        else:
            # Facing raise - check 3bet and calling ranges
            # Assume CO opened (simplification)
            opener = Position.CO

            three_bet_range = get_3bet_range(position, opener)
            calling_range = get_calling_range(position, opener)

            if hole_cards in three_bet_range:
                return f"3-BET - {hand_notation} is in 3-betting range vs {opener.name}"
            elif hole_cards in calling_range:
                return f"CALL - {hand_notation} is in calling range vs {opener.name}"
            else:
                return f"FOLD - {hand_notation} is not in defending range vs {opener.name}"


def create_bot(
    name: str = "GTOBot",
    aggression: float = 1.0,
    tightness: float = 1.0,
    simulations: int = 5000,
) -> GTOBot:
    """
    Create a configured GTO bot.

    Args:
        name: Bot name
        aggression: Aggression multiplier (>1 = more aggressive)
        tightness: Tightness multiplier (>1 = tighter)
        simulations: Monte Carlo simulations

    Returns:
        Configured GTOBot instance
    """
    config = BotConfig(
        name=name,
        aggression=aggression,
        tightness=tightness,
        simulations=simulations,
    )
    return GTOBot(config)
