"""Nit bot - ultra-tight player that only plays premium hands."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot, BotConfig


class NitBot(BaseBot):
    """
    Ultra-tight player that only plays premium hands.

    Characteristics:
    - VPIP: ~10%
    - PFR: ~8%
    - Only plays top 10% of hands
    - Rarely bluffs
    - Folds to aggression unless very strong
    - Small bet sizing (33-50% pot)
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the nit bot."""
        if config is None:
            config = BotConfig(name="Nit")
        super().__init__(config)

        self._rng = random.Random()
        # Preflop range: top 10% of hands
        self._opening_range = top_percent_range(10)

    def get_action(self, game_state: GameState) -> Action:
        """Get action - very tight, small bets, folds to pressure."""
        self._current_game = game_state

        player_idx = game_state.action_on
        available = game_state.get_available_actions()

        if self._hole_cards is None:
            return Action.fold(player_idx)

        hand_percentile = get_hand_percentile(self._hole_cards)

        if game_state.street == Street.PREFLOP:
            return self._preflop_action(game_state, player_idx, available, hand_percentile)
        else:
            return self._postflop_action(game_state, player_idx, available, hand_percentile)

    def _preflop_action(
        self,
        game_state: GameState,
        player_idx: int,
        available,
        hand_percentile: float,
    ) -> Action:
        """Preflop: Only play top 10%, fold to 3-bets without premiums."""
        # Only play top 10% of hands
        if hand_percentile > 10:
            return Action.fold(player_idx)

        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round

        # Facing a 3-bet or larger
        if to_call > game_state.big_blind * 6:
            # Only continue with top 3%
            if hand_percentile <= 3:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing a raise
        if to_call > game_state.big_blind:
            # Continue with top 8%
            if hand_percentile <= 8:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # No raise - open with small sizing
        if available.can_bet:
            # Small open: 2.2x BB
            bet_size = game_state.big_blind * 2.2
            return Action.bet(max(available.min_bet, bet_size), player_idx)

        if available.can_raise:
            # Small raise: 2.5x
            raise_size = game_state.current_bet * 2.5
            raise_size = max(available.min_raise, min(raise_size, available.max_raise))
            return Action.raise_to(raise_size, player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _postflop_action(
        self,
        game_state: GameState,
        player_idx: int,
        available,
        hand_percentile: float,
    ) -> Action:
        """Postflop: Only bet with strong hands, fold to aggression."""
        pot = game_state.pot.total
        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round

        # Estimate hand strength based on board
        # Nit only continues with top pair+
        has_strong_hand = self._evaluate_hand_strength(game_state)

        # Facing a bet
        if to_call > 0:
            # Fold without a strong hand
            if not has_strong_hand:
                return Action.fold(player_idx)

            # Facing large bet - need very strong hand
            if to_call > pot * 0.5:
                if hand_percentile <= 5:  # Only with premium starting hands
                    if available.can_call:
                        return Action.call(available.call_amount, player_idx)
                return Action.fold(player_idx)

            # Facing smaller bet - call with strong hand
            if available.can_call:
                return Action.call(available.call_amount, player_idx)

        # Check/bet decision
        if has_strong_hand:
            # Small value bet (33% pot)
            if available.can_bet:
                bet_size = pot * 0.33
                return Action.bet(max(available.min_bet, bet_size), player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _evaluate_hand_strength(self, game_state: GameState) -> bool:
        """
        Evaluate if we have a strong enough hand to continue.

        Nit requires at least top pair to continue.
        """
        if not self._hole_cards or not game_state.board:
            return True  # No board yet

        # Simple check: do we have a pair with the board?
        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Check for pair with board
        for rank in hand_ranks:
            if rank in board_ranks:
                # Check if it's top pair
                if rank == max(board_ranks):
                    return True
                # Second pair with top kicker (ace)
                from pokerbot.core.card import Rank
                if Rank.ACE in hand_ranks:
                    return True

        # Check for overpair
        top_board = max(board_ranks)
        if hand_ranks[0] == hand_ranks[1] and hand_ranks[0] > top_board:
            return True

        return False
