"""
Counterfactual Regret Minimization (CFR) Algorithm.

This implements vanilla CFR for poker. The algorithm:
1. Iteratively traverses the game tree
2. Computes counterfactual regrets for each action
3. Updates strategy proportionally to positive regrets
4. Converges to Nash equilibrium over many iterations

Based on:
- Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information"
- Brown & Sandholm (2019) "Superhuman AI for multiplayer poker" (Pluribus)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import random

from pokerbot.core.hand import Hand
from pokerbot.core.card import Card
from pokerbot.core.deck import Deck
from pokerbot.strategy.cfr.abstraction import (
    HandBucket, ActionBucket, HandAbstraction, ActionAbstraction,
    create_info_set_key
)
from pokerbot.strategy.cfr.strategy import CFRStrategy, StrategyProfile


@dataclass
class GameNode:
    """
    Represents a node in the game tree.
    """
    is_terminal: bool = False
    is_chance: bool = False
    player: int = 0  # 0 or 1
    pot: float = 0.0
    stacks: tuple = (100.0, 100.0)
    street: int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
    board: list = None
    hands: dict = None  # {player: Hand}
    action_history: tuple = ()
    current_bet: float = 0.0
    to_call: float = 0.0

    # Terminal node payoffs
    payoffs: tuple = None  # (player0_payoff, player1_payoff)


class CFRSolver:
    """
    CFR solver for heads-up poker.

    Uses external sampling MCCFR for efficiency.
    """

    def __init__(self, stack_size: float = 100.0, big_blind: float = 1.0):
        self.stack_size = stack_size
        self.big_blind = big_blind
        self.small_blind = big_blind / 2

        self.hand_abstraction = HandAbstraction()
        self.strategy = StrategyProfile(num_players=2)

        self._rng = random.Random()
        self._deck = Deck()

    def train(self, iterations: int = 10000, verbose: bool = True):
        """
        Train the strategy via self-play.

        Args:
            iterations: Number of training iterations
            verbose: Print progress
        """
        for i in range(iterations):
            # Reset deck by shuffling
            self._deck.shuffle()

            hands = {
                0: self._deal_hand(),
                1: self._deal_hand(),
            }

            # Run CFR for both players
            for player in [0, 1]:
                self._cfr(
                    player=player,
                    hands=hands,
                    board=[],
                    pot=self.small_blind + self.big_blind,
                    stacks=(
                        self.stack_size - self.small_blind,
                        self.stack_size - self.big_blind,
                    ),
                    street=0,
                    action_history=(),
                    reach_probs=(1.0, 1.0),
                )

            if verbose and (i + 1) % 1000 == 0:
                exploit = self.strategy.get_strategy(0).get_exploitability()
                print(f"Iteration {i+1}: exploitability={exploit:.6f}")

    def _deal_hand(self) -> Hand:
        """Deal a 2-card hand."""
        cards = self._deck.deal(2)
        return Hand(cards)

    def _cfr(
        self,
        player: int,
        hands: dict,
        board: list,
        pot: float,
        stacks: tuple,
        street: int,
        action_history: tuple,
        reach_probs: tuple,
        deck: Deck = None,
    ) -> float:
        """
        Recursive CFR traversal.

        Returns expected value for the traversing player.
        """
        # Use provided deck or create new one with dealt cards removed
        if deck is None:
            deck = Deck()
            for hand in hands.values():
                deck.remove(list(hand.cards))
            for card in board:
                deck.remove(card)

        # Check for terminal states
        if self._is_terminal(action_history, stacks):
            return self._get_terminal_value(player, hands, board, pot, action_history, deck)

        # Check if we need to deal community cards
        if self._should_deal(action_history, street):
            return self._chance_node(player, hands, board, pot, stacks, street, action_history, reach_probs, deck)

        # Determine acting player
        acting_player = self._get_acting_player(action_history)

        # Get info set key
        hand_bucket = self.hand_abstraction.get_bucket(hands[acting_player], board)
        info_key = create_info_set_key(
            hand_bucket, street, action_history, pot / self.stack_size
        )

        # Get available actions
        available = self._get_available_actions(pot, stacks, action_history)

        # Get strategy for this info set
        info_set = self.strategy.get_strategy(acting_player).get_info_set(info_key)

        # Initialize regrets for all actions
        for action in available:
            if action not in info_set.regret_sum:
                info_set.regret_sum[action] = 0

        strategy = info_set.get_strategy()

        # Calculate action values
        action_values = {}
        node_value = 0.0

        for action in available:
            action_prob = strategy.get(action, 1.0 / len(available))

            # Get new game state after action
            new_pot, new_stacks, new_history, new_street = self._apply_action(
                action, pot, stacks, action_history, street
            )

            # Update reach probabilities
            if acting_player == 0:
                new_reach = (reach_probs[0] * action_prob, reach_probs[1])
            else:
                new_reach = (reach_probs[0], reach_probs[1] * action_prob)

            # Recurse with a copy of the deck
            action_value = self._cfr(
                player, hands, board, new_pot, new_stacks,
                new_street, new_history, new_reach, deck.copy()
            )

            action_values[action] = action_value
            node_value += action_prob * action_value

        # Update regrets for traversing player
        if acting_player == player:
            opponent = 1 - player
            opponent_reach = reach_probs[opponent]

            for action in available:
                regret = action_values[action] - node_value
                info_set.update_regret(action, opponent_reach * regret)

        # Update strategy sum
        player_reach = reach_probs[acting_player]
        for action in available:
            action_prob = strategy.get(action, 1.0 / len(available))
            info_set.update_strategy(action, action_prob, player_reach)

        info_set.visit_count += 1

        return node_value

    def _chance_node(
        self,
        player: int,
        hands: dict,
        board: list,
        pot: float,
        stacks: tuple,
        street: int,
        action_history: tuple,
        reach_probs: tuple,
        deck: Deck,
    ) -> float:
        """Handle chance nodes (dealing community cards)."""
        # Deal cards based on street
        if street == 0:
            # Deal flop (3 cards)
            new_board = board + deck.deal(3)
            new_street = 1
        elif street == 1:
            # Deal turn
            new_board = board + deck.deal(1)
            new_street = 2
        elif street == 2:
            # Deal river
            new_board = board + deck.deal(1)
            new_street = 3
        else:
            return self._get_terminal_value(player, hands, board, pot, action_history, deck)

        # Continue CFR with new board, reset action history for new street
        return self._cfr(
            player, hands, new_board, pot, stacks,
            new_street, (), reach_probs, deck
        )

    def _is_terminal(self, action_history: tuple, stacks: tuple) -> bool:
        """Check if we've reached a terminal node."""
        if not action_history:
            return False

        last_action = action_history[-1]

        # Fold ends the hand
        if last_action == ActionBucket.FOLD:
            return True

        # All-in called
        if stacks[0] <= 0 or stacks[1] <= 0:
            if len(action_history) >= 2 and action_history[-1] == ActionBucket.CALL:
                return True

        # Check-check or bet-call sequence
        if len(action_history) >= 2:
            if action_history[-1] == ActionBucket.CHECK and action_history[-2] == ActionBucket.CHECK:
                return True
            if action_history[-1] == ActionBucket.CALL:
                return True

        return False

    def _should_deal(self, action_history: tuple, street: int) -> bool:
        """Check if we should deal community cards."""
        if street >= 3:  # Already on river
            return False

        if not action_history:
            return False

        # Deal after check-check or bet-call
        if len(action_history) >= 2:
            if action_history[-1] == ActionBucket.CHECK and action_history[-2] == ActionBucket.CHECK:
                return True
            if action_history[-1] == ActionBucket.CALL:
                return True

        return False

    def _get_terminal_value(
        self,
        player: int,
        hands: dict,
        board: list,
        pot: float,
        action_history: tuple,
        deck: Deck,
    ) -> float:
        """Get value at terminal node."""
        if not action_history:
            return 0

        # Check for fold
        if action_history[-1] == ActionBucket.FOLD:
            # Last actor folded
            acting = self._get_acting_player(action_history[:-1]) if len(action_history) > 1 else 0
            if acting == player:
                return -pot / 2  # We folded
            else:
                return pot / 2  # Opponent folded

        # Showdown - evaluate hands
        from pokerbot.core.evaluator import HandEvaluator

        board = list(board)  # Make a copy

        if len(board) < 5:
            # Need to complete the board
            cards_needed = 5 - len(board)
            board.extend(deck.deal(cards_needed))

        result0 = HandEvaluator.evaluate(hands[0], board)
        result1 = HandEvaluator.evaluate(hands[1], board)

        rank0 = result0.rank
        rank1 = result1.rank

        if rank0 > rank1:
            winner = 0
        elif rank1 > rank0:
            winner = 1
        else:
            # Tie
            return 0

        if winner == player:
            return pot / 2
        else:
            return -pot / 2

    def _get_acting_player(self, action_history: tuple) -> int:
        """Determine which player acts next."""
        # Simple alternating, starting with player 0 (SB/button in heads-up)
        return len(action_history) % 2

    def _get_available_actions(
        self,
        pot: float,
        stacks: tuple,
        action_history: tuple,
    ) -> list[ActionBucket]:
        """Get available action buckets."""
        acting = self._get_acting_player(action_history)
        stack = stacks[acting]

        if not action_history or action_history[-1] == ActionBucket.CHECK:
            # Can check or bet
            actions = [ActionBucket.CHECK]
            if stack > 0:
                actions.extend([
                    ActionBucket.BET_SMALL,
                    ActionBucket.BET_MEDIUM,
                    ActionBucket.BET_LARGE,
                ])
                if stack > pot:
                    actions.append(ActionBucket.BET_OVERBET)
                actions.append(ActionBucket.ALL_IN)
            return actions
        else:
            # Facing bet - can fold, call, raise
            actions = [ActionBucket.FOLD, ActionBucket.CALL]
            if stack > 0:
                actions.extend([
                    ActionBucket.BET_SMALL,
                    ActionBucket.BET_MEDIUM,
                    ActionBucket.BET_LARGE,
                    ActionBucket.ALL_IN,
                ])
            return actions

    def _apply_action(
        self,
        action: ActionBucket,
        pot: float,
        stacks: tuple,
        action_history: tuple,
        street: int,
    ) -> tuple:
        """
        Apply action and return new game state.

        Returns: (new_pot, new_stacks, new_history, new_street)
        """
        acting = self._get_acting_player(action_history)
        stack = stacks[acting]

        new_stacks = list(stacks)
        new_pot = pot
        new_history = action_history + (action,)
        new_street = street

        if action == ActionBucket.FOLD:
            pass  # No money movement
        elif action == ActionBucket.CHECK:
            pass
        elif action == ActionBucket.CALL:
            # Match opponent's bet (simplified)
            call_amount = min(stack, pot * 0.5)  # Approximate
            new_stacks[acting] -= call_amount
            new_pot += call_amount
        elif action == ActionBucket.ALL_IN:
            new_pot += stack
            new_stacks[acting] = 0
        else:
            # Betting action
            bet_size = ActionAbstraction.get_bet_size(action, pot, stack)
            bet_size = min(bet_size, stack)
            new_stacks[acting] -= bet_size
            new_pot += bet_size

        return new_pot, tuple(new_stacks), new_history, new_street

    def get_action(
        self,
        hand: Hand,
        board: list[Card],
        pot: float,
        stack: float,
        action_history: tuple = (),
        street: int = 0,
    ) -> ActionBucket:
        """
        Get recommended action from trained strategy.

        Args:
            hand: Player's hole cards
            board: Community cards
            pot: Current pot
            stack: Player's stack
            action_history: Actions taken this street
            street: Current street (0-3)

        Returns:
            Recommended action bucket
        """
        hand_bucket = self.hand_abstraction.get_bucket(hand, board)
        info_key = create_info_set_key(
            hand_bucket, street, action_history, pot / self.stack_size
        )

        available = self._get_available_actions(pot, (stack, stack), action_history)

        return self.strategy.get_strategy(0).get_action(info_key, available)


class CFRTrainer:
    """
    High-level interface for training CFR strategies.
    """

    def __init__(
        self,
        stack_size: float = 100.0,
        big_blind: float = 1.0,
        save_path: Optional[str] = None,
    ):
        self.solver = CFRSolver(stack_size, big_blind)
        self.save_path = save_path
        self.iterations_trained = 0

    def train(
        self,
        iterations: int = 10000,
        save_every: int = 1000,
        verbose: bool = True,
    ):
        """
        Train the CFR strategy.

        Args:
            iterations: Number of iterations
            save_every: Save checkpoint every N iterations
            verbose: Print progress
        """
        for i in range(iterations):
            self.solver.train(iterations=1, verbose=False)
            self.iterations_trained += 1

            if verbose and (self.iterations_trained % 1000) == 0:
                exploit = self.solver.strategy.get_strategy(0).get_exploitability()
                info_sets = len(self.solver.strategy.get_strategy(0))
                print(
                    f"Iteration {self.iterations_trained}: "
                    f"info_sets={info_sets}, "
                    f"exploitability={exploit:.6f}"
                )

            if self.save_path and (self.iterations_trained % save_every) == 0:
                self.save()

    def save(self):
        """Save current strategy."""
        if self.save_path:
            self.solver.strategy.save(self.save_path)

    def load(self):
        """Load saved strategy."""
        if self.save_path:
            self.solver.strategy.load(self.save_path)

    def get_action(self, hand: Hand, board: list[Card], pot: float,
                   stack: float, **kwargs) -> ActionBucket:
        """Get action from trained strategy."""
        return self.solver.get_action(hand, board, pot, stack, **kwargs)
