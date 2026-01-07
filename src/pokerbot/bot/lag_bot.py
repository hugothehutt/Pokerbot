"""LAG bot - loose-aggressive player that applies constant pressure."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot, BotConfig


class LAGBot(BaseBot):
    """
    Loose-Aggressive player that applies constant pressure.

    Characteristics:
    - VPIP: ~35%
    - PFR: ~28%
    - Plays wide range aggressively
    - Frequent raises and re-raises
    - High c-bet frequency
    - Barrels multiple streets
    - Large bet sizing (75-125% pot)
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the LAG bot."""
        if config is None:
            config = BotConfig(name="LAG")
        super().__init__(config)

        self._rng = random.Random()
        # Plays top 35% of hands
        self._opening_range = top_percent_range(35)
        self._was_preflop_aggressor = False

    def get_action(self, game_state: GameState) -> Action:
        """Get action - aggressive with wide range."""
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
        """Preflop: Wide range with aggressive raises."""
        # Play top 35% of hands
        if hand_percentile > 35:
            # Sometimes still 3-bet light
            if hand_percentile <= 45 and self._rng.random() < 0.15:
                pass  # Continue with light 3-bet
            else:
                return Action.fold(player_idx)

        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round

        # Facing a 4-bet+
        if to_call > game_state.big_blind * 15:
            if hand_percentile <= 5:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing a 3-bet
        if to_call > game_state.big_blind * 6:
            if hand_percentile <= 15:
                # 4-bet sometimes
                if hand_percentile <= 5 or self._rng.random() < 0.3:
                    if available.can_raise:
                        raise_size = to_call * 2.5
                        raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                        return Action.raise_to(raise_size, player_idx)
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing an open
        if to_call > game_state.big_blind:
            # 3-bet frequently
            if hand_percentile <= 25:
                if self._rng.random() < 0.6:  # 60% 3-bet frequency
                    if available.can_raise:
                        raise_size = to_call * 3
                        raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                        self._was_preflop_aggressor = True
                        return Action.raise_to(raise_size, player_idx)

            if hand_percentile <= 35:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Open raise with standard sizing
        if available.can_raise:
            raise_size = game_state.big_blind * 2.8
            raise_size = max(available.min_raise, min(raise_size, available.max_raise))
            self._was_preflop_aggressor = True
            return Action.raise_to(raise_size, player_idx)

        if available.can_bet:
            self._was_preflop_aggressor = True
            return Action.bet(game_state.big_blind * 2.8, player_idx)

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
        """Postflop: High aggression, barrel multiple streets."""
        pot = game_state.pot.total
        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round
        stack = game_state.players[player_idx].stack

        has_equity = self._has_equity(game_state)
        has_strong = self._has_strong_hand(game_state)

        # Facing a bet
        if to_call > 0:
            # With strong hand, raise for value
            if has_strong:
                if available.can_raise:
                    raise_size = to_call + pot * 0.8
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)

            # With equity, float or semi-bluff raise
            if has_equity:
                # Raise as bluff sometimes
                if self._rng.random() < 0.25:
                    if available.can_raise and raise_size < stack * 0.4:
                        raise_size = to_call + pot * 0.75
                        raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                        return Action.raise_to(raise_size, player_idx)

                # Float with position
                if to_call < pot * 0.6:
                    if available.can_call:
                        return Action.call(available.call_amount, player_idx)

            # Fold without equity on large bets
            if to_call > pot * 0.5:
                return Action.fold(player_idx)

            # Call smaller bets with any backdoor potential
            if available.can_call and to_call < pot * 0.4:
                if self._rng.random() < 0.35:
                    return Action.call(available.call_amount, player_idx)

            return Action.fold(player_idx)

        # Not facing bet - bet aggressively
        # C-bet as preflop aggressor
        if self._was_preflop_aggressor or has_equity or has_strong:
            cbet_freq = 0.75 if game_state.street == Street.FLOP else 0.55

            if has_strong or self._rng.random() < cbet_freq:
                if available.can_bet:
                    # Large sizing (75-100% pot)
                    bet_size = pot * (0.75 + self._rng.random() * 0.25)
                    return Action.bet(max(available.min_bet, bet_size), player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _has_equity(self, game_state: GameState) -> bool:
        """Check if we have any equity (pair, draw, overcards)."""
        if not self._hole_cards or not game_state.board:
            return True

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Pair with board
        for rank in hand_ranks:
            if rank in board_ranks:
                return True

        # Overcards
        top_board = max(board_ranks)
        if hand_ranks[0] > top_board or hand_ranks[1] > top_board:
            return True

        # Pocket pair
        if hand_ranks[0] == hand_ranks[1]:
            return True

        # Draws
        return self._has_draw(game_state)

    def _has_draw(self, game_state: GameState) -> bool:
        """Check for flush or straight draw."""
        if not self._hole_cards or len(game_state.board) < 3:
            return False

        all_cards = list(self._hole_cards.cards) + list(game_state.board)

        # Flush draw
        suits = {}
        for card in all_cards:
            suits[card.suit] = suits.get(card.suit, 0) + 1
        if max(suits.values()) >= 4:
            return True

        # Straight draw
        ranks = sorted(set([c.rank.value for c in all_cards]))
        for i in range(len(ranks) - 3):
            window = ranks[i:i+4]
            if window[-1] - window[0] <= 4:
                return True

        return False

    def _has_strong_hand(self, game_state: GameState) -> bool:
        """Check for strong made hand."""
        if not self._hole_cards or not game_state.board:
            return False

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Top pair with good kicker
        top_board = max(board_ranks)
        from pokerbot.core.card import Rank
        for rank in hand_ranks:
            if rank == top_board and max(hand_ranks) >= Rank.TEN:
                return True

        # Overpair
        if hand_ranks[0] == hand_ranks[1] and hand_ranks[0] > top_board:
            return True

        # Two pair or better (simplified)
        pair_count = sum(1 for r in hand_ranks if r in board_ranks)
        if pair_count >= 2:
            return True

        return False

    def on_hand_start(self, hole_cards: Hand, game_state: GameState) -> None:
        """Reset aggressor flag at start of hand."""
        super().on_hand_start(hole_cards, game_state)
        self._was_preflop_aggressor = False
