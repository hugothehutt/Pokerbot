"""Fish bot - loose-passive player that calls too much."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot, BotConfig


class FishBot(BaseBot):
    """
    Loose-passive player that calls too much and rarely raises.

    Characteristics:
    - VPIP: ~50%
    - PFR: ~12%
    - Plays way too many hands
    - Calls frequently (calling station tendencies)
    - Rarely raises or bluffs
    - Small, weak bets when betting
    - Chases draws without pot odds
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the fish bot."""
        if config is None:
            config = BotConfig(name="Fish")
        super().__init__(config)

        self._rng = random.Random()
        # Plays top 50% of hands
        self._opening_range = top_percent_range(50)

    def get_action(self, game_state: GameState) -> Action:
        """Get action - call often, rarely raise, weak bets."""
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
        """Preflop: Play 50% of hands, mostly calling."""
        # Fold the worst 50%
        if hand_percentile > 50:
            return Action.fold(player_idx)

        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round

        # Facing a big raise - only continue with decent hands
        if to_call > game_state.big_blind * 8:
            if hand_percentile <= 15:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing a raise - call with most of our range
        if to_call > game_state.big_blind:
            if hand_percentile <= 40:  # Still too loose
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # No raise - occasionally raise with premium, mostly limp/call
        if hand_percentile <= 12:
            # Raise with good hands (but not always)
            if self._rng.random() < 0.7:
                if available.can_raise:
                    raise_size = game_state.big_blind * 2.5
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)
                if available.can_bet:
                    return Action.bet(game_state.big_blind * 2.5, player_idx)

        # Limp/call with the rest
        if available.can_call:
            return Action.call(available.call_amount, player_idx)
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
        """Postflop: Call too much, bet weakly."""
        pot = game_state.pot.total
        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round
        stack = game_state.players[player_idx].stack

        has_piece = self._has_any_piece(game_state)
        has_draw = self._has_draw(game_state)
        has_strong = self._has_strong_hand(game_state)

        # Facing a bet
        if to_call > 0:
            # Call with any piece of the board
            if has_piece or has_draw:
                # Call almost any bet (ignoring pot odds)
                if to_call < stack * 0.5:
                    if available.can_call:
                        return Action.call(available.call_amount, player_idx)

                # Only fold big bets without strong hand
                if has_strong:
                    if available.can_call:
                        return Action.call(available.call_amount, player_idx)

            # Sometimes call anyway (loose passive)
            if self._rng.random() < 0.2:
                if available.can_call and to_call < pot:
                    return Action.call(available.call_amount, player_idx)

            return Action.fold(player_idx)

        # Not facing bet - bet weakly with strong hands
        if has_strong:
            if available.can_bet:
                # Small, weak bet (25-40% pot)
                bet_size = pot * (0.25 + self._rng.random() * 0.15)
                return Action.bet(max(available.min_bet, bet_size), player_idx)

        # Sometimes bet with any piece (weak lead)
        if has_piece and self._rng.random() < 0.2:
            if available.can_bet:
                bet_size = pot * 0.25
                return Action.bet(max(available.min_bet, bet_size), player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _has_any_piece(self, game_state: GameState) -> bool:
        """Check if we have any pair or draw."""
        if not self._hole_cards or not game_state.board:
            return False

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Any pair with board
        for rank in hand_ranks:
            if rank in board_ranks:
                return True

        # Pocket pair
        if hand_ranks[0] == hand_ranks[1]:
            return True

        return self._has_draw(game_state)

    def _has_draw(self, game_state: GameState) -> bool:
        """Check if we have a flush or straight draw."""
        if not self._hole_cards or len(game_state.board) < 3:
            return False

        all_cards = list(self._hole_cards.cards) + list(game_state.board)

        # Flush draw
        suits = {}
        for card in all_cards:
            suits[card.suit] = suits.get(card.suit, 0) + 1
        if max(suits.values()) >= 4:
            return True

        # Straight draw (simplified)
        ranks = sorted([c.rank.value for c in all_cards])
        for i in range(len(ranks) - 3):
            window = ranks[i:i+4]
            if window[-1] - window[0] <= 4:
                return True

        return False

    def _has_strong_hand(self, game_state: GameState) -> bool:
        """Check for two pair or better."""
        if not self._hole_cards or not game_state.board:
            return False

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        pair_count = 0
        for rank in hand_ranks:
            if rank in board_ranks:
                pair_count += 1

        # Two pair or trips
        if pair_count >= 2 or hand_ranks[0] == hand_ranks[1]:
            return True

        return False
