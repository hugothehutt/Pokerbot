"""Calling Station bot - passive player that calls everything."""

from __future__ import annotations
import random
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType
from pokerbot.equity.range import top_percent_range, get_hand_percentile
from pokerbot.bot.base import BaseBot, BotConfig


class CallingStationBot(BaseBot):
    """
    Passive player that calls almost everything but rarely raises.

    Characteristics:
    - VPIP: ~55%
    - PFR: ~8%
    - Plays many hands but very passively
    - Calls everything, never folds postflop with any piece
    - Almost never raises or bluffs
    - Min-bets when betting
    - Will call any bet size
    - "Sheriff" mentality - must see showdown
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """Initialize the calling station bot."""
        if config is None:
            config = BotConfig(name="CallStation")
        super().__init__(config)

        self._rng = random.Random()
        # Plays top 55% of hands
        self._opening_range = top_percent_range(55)

    def get_action(self, game_state: GameState) -> Action:
        """Get action - call everything, rarely raise."""
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
        """Preflop: Call with many hands, rarely raise."""
        # Play 55% of hands
        if hand_percentile > 55:
            return Action.fold(player_idx)

        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round
        stack = game_state.players[player_idx].stack

        # Facing a huge raise - still call with decent hands
        if to_call > game_state.big_blind * 15:
            if hand_percentile <= 15:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing a 3-bet - call if we can
        if to_call > game_state.big_blind * 5:
            if hand_percentile <= 30:
                if available.can_call:
                    return Action.call(available.call_amount, player_idx)
            return Action.fold(player_idx)

        # Facing an open - just call (never 3-bet)
        if to_call > 0:
            if available.can_call:
                return Action.call(available.call_amount, player_idx)

        # No raise - occasionally raise with premium, mostly check/call
        if hand_percentile <= 5:
            # Very rarely raise even with premiums
            if self._rng.random() < 0.3:
                if available.can_raise:
                    raise_size = game_state.big_blind * 2.5
                    raise_size = max(available.min_raise, min(raise_size, available.max_raise))
                    return Action.raise_to(raise_size, player_idx)

        # Just check or call the blind
        if available.can_check:
            return Action.check(player_idx)
        if available.can_call:
            return Action.call(available.call_amount, player_idx)

        return Action.fold(player_idx)

    def _postflop_action(
        self,
        game_state: GameState,
        player_idx: int,
        available,
        hand_percentile: float,
    ) -> Action:
        """Postflop: Call almost everything, min-bet when betting."""
        pot = game_state.pot.total
        to_call = game_state.current_bet - game_state.players[player_idx].bet_this_round
        stack = game_state.players[player_idx].stack

        has_anything = self._has_anything(game_state)
        has_strong = self._has_strong_hand(game_state)

        # Facing a bet
        if to_call > 0:
            # Call with any piece - the calling station way
            if has_anything:
                # Call almost any bet size
                if to_call <= stack:
                    if available.can_call:
                        return Action.call(available.call_amount, player_idx)

            # Even without a hand, sometimes call (can't be bluffed easily)
            if self._rng.random() < 0.25:
                if available.can_call and to_call < pot:
                    return Action.call(available.call_amount, player_idx)

            return Action.fold(player_idx)

        # Not facing bet - rarely bet, usually check
        # Only bet with strong hands, and even then small
        if has_strong and self._rng.random() < 0.4:
            if available.can_bet:
                # Min-bet or tiny bet
                bet_size = max(available.min_bet, pot * 0.2)
                return Action.bet(bet_size, player_idx)

        if available.can_check:
            return Action.check(player_idx)

        return Action.fold(player_idx)

    def _has_anything(self, game_state: GameState) -> bool:
        """Check if we have literally any reason to call."""
        if not self._hole_cards or not game_state.board:
            return True  # No board = always call preflop

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Any pair with board (any pair, not just top pair)
        for rank in hand_ranks:
            if rank in board_ranks:
                return True

        # Any pocket pair
        if hand_ranks[0] == hand_ranks[1]:
            return True

        # High cards (even unconnected)
        from pokerbot.core.card import Rank
        if max(hand_ranks) >= Rank.JACK:
            return True

        # Any draw (very liberally defined)
        if len(game_state.board) >= 3:
            all_cards = list(self._hole_cards.cards) + list(game_state.board)

            # Any 3 of same suit
            suits = {}
            for card in all_cards:
                suits[card.suit] = suits.get(card.suit, 0) + 1
            if max(suits.values()) >= 3:
                return True

            # Any 3 cards within 5 ranks (very loose straight draw)
            ranks = sorted([c.rank.value for c in all_cards])
            for i in range(len(ranks) - 2):
                if ranks[i+2] - ranks[i] <= 5:
                    return True

        return False

    def _has_strong_hand(self, game_state: GameState) -> bool:
        """Check for genuinely strong hand - rare for calling station to bet."""
        if not self._hole_cards or not game_state.board:
            return False

        hand_ranks = [c.rank for c in self._hole_cards.cards]
        board_ranks = [c.rank for c in game_state.board]

        # Top pair or overpair
        top_board = max(board_ranks)
        for rank in hand_ranks:
            if rank == top_board:
                return True

        if hand_ranks[0] == hand_ranks[1] and hand_ranks[0] > top_board:
            return True

        # Two pair or better
        pair_count = sum(1 for r in hand_ranks if r in board_ranks)
        if pair_count >= 2:
            return True

        # Flush on board
        if len(game_state.board) >= 3:
            all_cards = list(self._hole_cards.cards) + list(game_state.board)
            suits = {}
            for card in all_cards:
                suits[card.suit] = suits.get(card.suit, 0) + 1
            if max(suits.values()) >= 5:
                return True

        return False
