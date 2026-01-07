"""Maniac bot - hyper-aggressive player that raises constantly."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot, BotConfig


class ManiacBot(BaseBot):
    """
    Hyper-aggressive player that raises constantly.

    Characteristics:
    - VPIP: ~60%
    - PFR: ~45%
    - Plays almost any hand
    - Raises and re-raises constantly
    - Overbets frequently
    - High bluff frequency
    - Massive bet sizing (100-200% pot)
    - Never gives up without a fight
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the maniac bot."""
        if config is None:
            config = BotConfig(name="Maniac")
        super().__init__(config)

        self._rng = random.Random()
        # Plays top 60% of hands
        self._opening_range = top_percent_range(60)

    def get_action(self, game_state: GameState) -> Action:
        """Get action - constant aggression regardless of cards."""
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
        """Preflop: Raise almost everything."""
        stack = game_state.players[player_idx].stack

        # Play 60% of hands, occasionally even more
        if hand_percentile > 60:
            if self._rng.random() < 0.15:
                pass  # Still play some junk
            else:
                return Action.fold(player_idx)

        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round

        # Facing a 4-bet+ - still fight back sometimes
        if to_call > game_state.big_blind * 20:
            if hand_percentile <= 8 or self._rng.random() < 0.15:
                if available.can_raise and self._rng.random() < 0.5:
                    # 5-bet jam
                    return Action.all_in(stack, player_idx)
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing a 3-bet - 4-bet frequently
        if to_call > game_state.big_blind * 6:
            if hand_percentile <= 20 or self._rng.random() < 0.3:
                if available.can_raise:
                    raise_size = to_call * 2.8
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing an open - 3-bet very frequently
        if to_call > game_state.big_blind:
            # 3-bet 60%+ of continuing range
            if self._rng.random() < 0.65:
                if available.can_raise:
                    raise_size = to_call * 3.5
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)

            if available.can_call:
                return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Open raise - always raise, never limp
        if available.can_raise:
            raise_size = game_state.big_blind * 3.5
            raise_size = max(available.min_raise, min(raise_size, available.max_raise))
            return Action.raise_to(raise_size, player_idx)

        if available.can_bet:
            return Action.bet(game_state.big_blind * 3.5, player_idx)

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
        """Postflop: Overbets, constant aggression."""
        pot = game_state.pot.total
        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round
        stack = game_state.players[player_idx].stack

        has_any_hand = self._has_any_hand(game_state)

        # Facing a bet
        if to_call > 0:
            # Raise frequently regardless of hand
            raise_freq = 0.5 if has_any_hand else 0.35

            if self._rng.random() < raise_freq:
                if available.can_raise:
                    # Big raise (pot+ sizing)
                    raise_size = to_call + pot * (1.0 + self._rng.random() * 0.5)
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)

            # Call somewhat wide
            call_freq = 0.6 if has_any_hand else 0.35
            if self._rng.random() < call_freq and to_call < stack * 0.5:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)

            # All-in bluff sometimes
            if self._rng.random() < 0.1:
                return Action.all_in(stack, player_idx)

            return Action.fold(player_idx)

        # Not facing bet - bet/raise very frequently
        # Bet almost always
        bet_freq = 0.85 if has_any_hand else 0.7

        if self._rng.random() < bet_freq:
            if available.can_bet:
                # Overbet (100-200% pot)
                bet_size = pot * (1.0 + self._rng.random())
                return Action.bet(max(available.min_bet, min(bet_size, stack)), player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _has_any_hand(self, game_state: GameState) -> bool:
        """Check for any piece of the board or draw."""
        if not self._hole_cards or not game_state.board:
            return True

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Any pair
        for rank in hand_ranks:
            if rank in board_ranks:
                return True

        # Pocket pair
        if hand_ranks[0] == hand_ranks[1]:
            return True

        # Overcards
        if max(hand_ranks) > max(board_ranks):
            return True

        # Any draw
        if len(game_state.board) >= 3:
            all_cards = list(self._hole_cards.cards) + list(game_state.board)

            # Flush draw
            suits = {}
            for card in all_cards:
                suits[card.suit] = suits.get(card.suit, 0) + 1
            if max(suits.values()) >= 4:
                return True

            # Any connected cards for straight potential
            ranks = sorted([c.rank.value for c in all_cards])
            for i in range(len(ranks) - 2):
                if ranks[i+2] - ranks[i] <= 4:
                    return True

        return False
