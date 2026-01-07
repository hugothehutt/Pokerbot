"""Decision engine combining equity, GTO concepts, and opponent modeling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
import random

from pokerbot.core.hand import Hand
from pokerbot.core.card import Card
from pokerbot.core.evaluator import HandEvaluator, HandRank
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType, AvailableActions
from pokerbot.game.player import Position
from pokerbot.equity.calculator import EquityCalculator
from pokerbot.equity.range import HandRange
from pokerbot.strategy.gto.ranges import PreflopRanges, get_opening_range
from pokerbot.strategy.gto.sizing import BetSizer, SizingStrategy
from pokerbot.strategy.gto.solver import SimpleSolver
from pokerbot.strategy.opponent import OpponentModel, OpponentProfile
from pokerbot.strategy.cfr.abstraction import HandAbstraction, ActionAbstraction, HandBucket, ActionBucket
from pokerbot.strategy.cfr.precomputed import get_precomputed_strategy


class Confidence(Enum):
    """Confidence level in a decision."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass
class Decision:
    """
    A recommended poker decision with reasoning.

    Contains the recommended action, alternatives, and explanation.
    """
    action: Action
    confidence: Confidence
    reasoning: str
    ev_estimate: Optional[float] = None

    # Alternative actions considered
    alternatives: list[tuple[Action, float]] = None  # (action, EV)

    # Analysis details
    equity: Optional[float] = None
    pot_odds: Optional[float] = None
    required_equity: Optional[float] = None

    def __str__(self) -> str:
        conf = self.confidence.name.lower()
        ev_str = f" (EV: {self.ev_estimate:.2f})" if self.ev_estimate else ""
        return f"[{conf}] {self.action}{ev_str}: {self.reasoning}"


@dataclass
class DecisionContext:
    """Context for making a decision."""
    game_state: GameState
    hero_hand: Hand
    hero_position: Position
    villain_range: Optional[HandRange] = None
    opponent_profile: Optional[OpponentProfile] = None


class DecisionEngine:
    """
    Main decision-making engine.

    Combines:
    - Equity calculations (Monte Carlo)
    - GTO concepts (ranges, sizing, frequencies)
    - Opponent modeling (exploitative adjustments)
    """

    def __init__(
        self,
        simulations: int = 5000,
        use_opponent_modeling: bool = True,
    ):
        """
        Initialize decision engine.

        Args:
            simulations: Monte Carlo simulations for equity
            use_opponent_modeling: Whether to use exploitative adjustments
        """
        self.equity_calc = EquityCalculator(default_simulations=simulations)
        self.solver = SimpleSolver()
        self.opponent_model = OpponentModel() if use_opponent_modeling else None
        self._rng = random.Random()

        # CFR-based precomputed strategy
        self.cfr_strategy = get_precomputed_strategy()
        self.hand_abstraction = HandAbstraction()

        # GTO bet sizing options with pot percentages
        # These represent the sizes GTO solvers typically use
        self.BET_SIZES = {
            "quarter": 0.25,    # 25% pot - probe/block bets
            "third": 0.33,      # 33% pot - small sizing
            "half": 0.50,       # 50% pot - medium sizing
            "two_thirds": 0.67, # 67% pot - standard
            "three_quarters": 0.75,  # 75% pot - large
            "pot": 1.00,        # 100% pot - pot-sized
            "overbet_small": 1.25,   # 125% pot - small overbet
            "overbet_large": 1.50,   # 150% pot - large overbet
            "overbet_massive": 2.00, # 200% pot - massive overbet
        }

    def make_decision(self, context: DecisionContext) -> Decision:
        """
        Make a decision for the current spot.

        Args:
            context: Decision context with game state and hand

        Returns:
            Decision with recommended action
        """
        state = context.game_state
        street = state.street

        if street == Street.PREFLOP:
            return self._preflop_decision(context)
        else:
            return self._postflop_decision(context)

    def _select_bet_size(
        self,
        context: DecisionContext,
        equity: float,
        is_value: bool = True,
    ) -> tuple[float, str]:
        """
        Select appropriate bet size based on GTO principles.

        GTO sizing selection considers:
        - Hand strength (polarized vs merged)
        - Board texture (wet vs dry)
        - SPR (stack-to-pot ratio)
        - Street (earlier streets = smaller sizes)

        Args:
            context: Decision context
            equity: Current equity vs opponent range
            is_value: True for value betting, False for bluffs

        Returns:
            Tuple of (pot_percentage, reasoning)
        """
        state = context.game_state
        pot = state.pot.total
        spr = state.get_stack_to_pot_ratio()
        street = state.street
        board = state.board

        # Analyze board texture
        is_wet_board = self._is_wet_board(board) if board else False
        is_paired_board = self._is_paired_board(board) if board else False

        # Determine if we have a polarized or merged range situation
        # Polarized: very strong or bluffs (equity > 75% or < 30%)
        # Merged: medium strength hands (equity 30-75%)
        is_polarized = equity > 0.75 or equity < 0.30

        # Select sizing based on multiple factors
        sizing_options = []

        # SPR-based constraints
        if spr < 2:
            # Very shallow - consider all-in or check
            if is_value and equity > 0.6:
                return (1.0, "Low SPR - committing size")
            else:
                return (0.33, "Low SPR - small probe")

        elif spr < 4:
            # Short SPR - larger sizes to set up all-in
            sizing_options = [
                (0.75, "Short SPR - 75% pot"),
                (1.00, "Short SPR - pot-sized"),
            ]

        elif spr > 12:
            # Deep stacked - smaller sizes maintain flexibility
            sizing_options = [
                (0.25, "Deep SPR - small probe"),
                (0.33, "Deep SPR - 33% pot"),
                (0.50, "Deep SPR - half pot"),
            ]

        else:
            # Medium SPR - full range of sizes
            if is_polarized:
                # Use larger sizes with polarized range
                if is_value:
                    sizing_options = [
                        (0.75, "Polarized value - 75% pot"),
                        (1.00, "Polarized value - pot-sized"),
                        (1.25, "Polarized value - 125% overbet"),
                    ]
                else:
                    # Bluffs should use same sizes as value for balance
                    sizing_options = [
                        (0.75, "Bluff - 75% pot"),
                        (1.00, "Bluff - pot-sized"),
                        (1.25, "Bluff - overbet"),
                    ]
            else:
                # Merged range - smaller sizes
                sizing_options = [
                    (0.33, "Merged - small bet"),
                    (0.50, "Merged - half pot"),
                    (0.67, "Merged - 67% pot"),
                ]

        # Adjust for board texture
        if is_wet_board:
            # On wet boards, use larger sizes to charge draws
            sizing_options = [(min(s[0] * 1.2, 1.5), s[1] + " (wet board)") for s in sizing_options]
        elif is_paired_board:
            # On paired boards, often use smaller sizes
            sizing_options = [(s[0] * 0.8, s[1] + " (paired)") for s in sizing_options]

        # Adjust for street
        if street == Street.FLOP:
            # Flop: generally smaller to set up later streets
            sizing_options = [(min(s[0], 0.75), s[1]) for s in sizing_options]
        elif street == Street.RIVER:
            # River: can use larger sizes, including overbets
            if is_value and equity > 0.8:
                sizing_options.append((1.50, "River overbet for value"))

        # Pick from options with some randomization for balance
        if sizing_options:
            # Weight towards middle options
            weights = [1.0] * len(sizing_options)
            if len(sizing_options) >= 3:
                weights[len(weights) // 2] = 2.0  # Middle option weighted higher

            total = sum(weights)
            r = self._rng.random() * total
            cumulative = 0
            for (size, reason), weight in zip(sizing_options, weights):
                cumulative += weight
                if r <= cumulative:
                    return (size, reason)

            return sizing_options[-1]

        return (0.67, "Default sizing")

    def _is_wet_board(self, board: list) -> bool:
        """Check if board is wet (many draws possible)."""
        if len(board) < 3:
            return False

        # Check for flush draws (2+ of same suit)
        suits = {}
        for card in board:
            suits[card.suit] = suits.get(card.suit, 0) + 1
        has_flush_draw = max(suits.values()) >= 2

        # Check for straight draws (connected cards)
        ranks = sorted([c.rank.value for c in board])
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks)-1)]
        has_straight_draw = min(gaps) <= 2 if gaps else False

        return has_flush_draw and has_straight_draw

    def _is_paired_board(self, board: list) -> bool:
        """Check if board is paired."""
        if len(board) < 2:
            return False
        ranks = [c.rank for c in board]
        return len(ranks) != len(set(ranks))

    def _preflop_decision(self, context: DecisionContext) -> Decision:
        """Make preflop decision."""
        state = context.game_state
        hand = context.hero_hand
        position = context.hero_position
        available = state.get_available_actions()

        # Get GTO ranges
        opening_range = get_opening_range(position)
        hand_in_range = hand in opening_range

        # Check if we're facing a raise
        facing_raise = state.current_bet > state.big_blind

        if not facing_raise:
            # First to act or limped to us
            if hand_in_range:
                # Open raise
                sizer = BetSizer(state)
                sizing = sizer._preflop_sizing()

                return Decision(
                    action=Action.raise_to(sizing.amount),
                    confidence=Confidence.HIGH,
                    reasoning=f"{hand.notation()} is in opening range for {position.name}",
                    ev_estimate=None,
                )
            else:
                # Fold or check (if BB)
                if available.can_check:
                    return Decision(
                        action=Action.check(),
                        confidence=Confidence.HIGH,
                        reasoning=f"{hand.notation()} not in opening range - check",
                    )
                else:
                    return Decision(
                        action=Action.fold(),
                        confidence=Confidence.HIGH,
                        reasoning=f"{hand.notation()} not in opening range - fold",
                    )
        else:
            # Facing a raise - decide 3bet/call/fold
            return self._facing_raise_decision(context, available)

    def _facing_raise_decision(
        self,
        context: DecisionContext,
        available: AvailableActions,
    ) -> Decision:
        """Decision when facing a raise preflop."""
        state = context.game_state
        hand = context.hero_hand
        position = context.hero_position

        # Estimate opener's position (simplified)
        opener_position = Position.CO  # Default assumption

        # Get 3bet and calling ranges
        three_bet_range = PreflopRanges.get_3bet_range(position, opener_position)
        calling_range = PreflopRanges.get_calling_range(position, opener_position)

        in_3bet = hand in three_bet_range
        in_call = hand in calling_range

        if in_3bet:
            # 3-bet
            sizer = BetSizer(state)
            raise_size = sizer.get_3bet_size(
                state.current_bet,
                in_position=(position.value > opener_position.value),
            )

            return Decision(
                action=Action.raise_to(min(raise_size, state.players[state.action_on].stack)),
                confidence=Confidence.HIGH,
                reasoning=f"{hand.notation()} in 3-bet range vs {opener_position.name}",
            )
        elif in_call:
            return Decision(
                action=Action.call(available.call_amount),
                confidence=Confidence.MEDIUM,
                reasoning=f"{hand.notation()} in calling range vs {opener_position.name}",
            )
        else:
            return Decision(
                action=Action.fold(),
                confidence=Confidence.HIGH,
                reasoning=f"{hand.notation()} not in defending range vs {opener_position.name}",
            )

    def _postflop_decision(self, context: DecisionContext) -> Decision:
        """Make postflop decision using CFR-based mixed strategies."""
        state = context.game_state
        hand = context.hero_hand
        available = state.get_available_actions()
        pot = state.pot.total
        player = state.players[state.action_on]
        stack = player.stack

        # Get hand bucket for CFR strategy
        hand_bucket = self.hand_abstraction.get_bucket(hand, state.board)

        # Calculate equity for additional context
        villain_range = context.villain_range or HandRange("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,A9o+,KTo+,QTo+,JTo")
        equity_result = self.equity_calc.hand_vs_range(
            hand,
            villain_range,
            state.board,
        )
        equity = equity_result.equity

        to_call = state.current_bet - player.bet_this_round
        facing_bet = to_call > 0

        # Get street number for CFR
        street_num = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}.get(state.street, 1)

        # Get action from CFR precomputed strategy (mixed strategy)
        cfr_action = self.cfr_strategy.get_action(hand_bucket, street_num, facing_bet)

        # Convert CFR action to game action with proper sizing
        return self._cfr_action_to_decision(
            context, cfr_action, hand_bucket, equity, available, pot, stack, to_call
        )

    def _cfr_action_to_decision(
        self,
        context: DecisionContext,
        cfr_action: ActionBucket,
        hand_bucket: HandBucket,
        equity: float,
        available: AvailableActions,
        pot: float,
        stack: float,
        to_call: float,
    ) -> Decision:
        """Convert CFR action bucket to actual game action."""
        state = context.game_state
        player_idx = state.action_on

        # Calculate required equity for context
        required = self.solver.required_equity_to_call(to_call, pot) if to_call > 0 else 0
        pot_odds = pot / to_call if to_call > 0 else float('inf')

        # Map CFR action to game action
        if cfr_action == ActionBucket.FOLD:
            if available.can_fold:
                return Decision(
                    action=Action.fold(player_idx),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"CFR: Fold with {hand_bucket.name}",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )
            elif available.can_check:
                return Decision(
                    action=Action.check(player_idx),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"CFR: Check (can't fold) with {hand_bucket.name}",
                    equity=equity,
                )

        elif cfr_action == ActionBucket.CHECK:
            if available.can_check:
                return Decision(
                    action=Action.check(player_idx),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"CFR: Check with {hand_bucket.name}",
                    equity=equity,
                )
            elif available.can_call:
                return Decision(
                    action=Action.call(available.call_amount, player_idx),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"CFR: Call (can't check) with {hand_bucket.name}",
                    equity=equity,
                )

        elif cfr_action == ActionBucket.CALL:
            if available.can_call:
                return Decision(
                    action=Action.call(available.call_amount, player_idx),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"CFR: Call with {hand_bucket.name} ({equity*100:.0f}% equity)",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )

        elif cfr_action == ActionBucket.ALL_IN:
            return Decision(
                action=Action.all_in(stack, player_idx),
                confidence=Confidence.HIGH,
                reasoning=f"CFR: All-in with {hand_bucket.name}",
                equity=equity,
            )

        elif cfr_action in [ActionBucket.BET_SMALL, ActionBucket.BET_MEDIUM,
                            ActionBucket.BET_LARGE, ActionBucket.BET_OVERBET]:
            # Get bet size from action abstraction
            bet_pct = ActionAbstraction.DEFAULT_SIZING.get(cfr_action, 0.67)
            bet_size = pot * bet_pct
            bet_size = min(bet_size, stack)

            if to_call > 0:
                # Raising
                if available.can_raise:
                    raise_amount = state.current_bet + pot * bet_pct
                    raise_amount = max(available.min_raise, min(raise_amount, available.max_raise))
                    return Decision(
                        action=Action.raise_to(raise_amount, player_idx),
                        confidence=Confidence.MEDIUM,
                        reasoning=f"CFR: Raise {bet_pct*100:.0f}% with {hand_bucket.name}",
                        equity=equity,
                    )
                elif available.can_call:
                    return Decision(
                        action=Action.call(available.call_amount, player_idx),
                        confidence=Confidence.LOW,
                        reasoning=f"CFR: Call (can't raise) with {hand_bucket.name}",
                        equity=equity,
                    )
            else:
                # Betting
                if available.can_bet:
                    if available.min_bet:
                        bet_size = max(available.min_bet, bet_size)
                    return Decision(
                        action=Action.bet(bet_size, player_idx),
                        confidence=Confidence.MEDIUM,
                        reasoning=f"CFR: Bet {bet_pct*100:.0f}% pot with {hand_bucket.name}",
                        equity=equity,
                    )

        # Fallback
        if available.can_check:
            return Decision(
                action=Action.check(player_idx),
                confidence=Confidence.LOW,
                reasoning=f"Fallback: Check with {hand_bucket.name}",
                equity=equity,
            )
        elif available.can_call:
            return Decision(
                action=Action.call(available.call_amount, player_idx),
                confidence=Confidence.LOW,
                reasoning=f"Fallback: Call with {hand_bucket.name}",
                equity=equity,
            )
        else:
            return Decision(
                action=Action.fold(player_idx),
                confidence=Confidence.LOW,
                reasoning=f"Fallback: Fold",
                equity=equity,
            )

    def _facing_bet_decision(
        self,
        context: DecisionContext,
        equity: float,
        available: AvailableActions,
    ) -> Decision:
        """Decision when facing a bet postflop."""
        state = context.game_state
        pot = state.pot.total
        to_call = available.call_amount
        player = state.players[state.action_on]
        stack = player.stack

        # Calculate required equity
        required = self.solver.required_equity_to_call(to_call, pot)
        pot_odds = pot / to_call if to_call > 0 else float('inf')

        # Get opponent bluff frequency estimate
        bluff_freq = 0.33  # Default
        if self.opponent_model and context.opponent_profile:
            bluff_freq = context.opponent_profile.bluff_frequency

        if equity >= required:
            # Profitable call - consider raising with strong hands
            if equity > 0.70 and available.can_raise:
                # Use dynamic sizing for raises
                size_pct, size_reason = self._select_bet_size(context, equity, is_value=True)

                # Calculate raise amount: current bet + (pot * sizing percentage)
                new_pot = pot + to_call
                raise_amount = state.current_bet + new_pot * size_pct
                raise_amount = max(available.min_raise, min(raise_amount, available.max_raise))

                # Consider all-in with very strong hands or when raise is > 60% of stack
                if equity > 0.85 or raise_amount > stack * 0.6:
                    return Decision(
                        action=Action.all_in(stack),
                        confidence=Confidence.HIGH,
                        reasoning=f"Very strong ({equity*100:.1f}%) - all-in for value",
                        equity=equity,
                        pot_odds=pot_odds,
                        required_equity=required,
                    )

                return Decision(
                    action=Action.raise_to(raise_amount),
                    confidence=Confidence.HIGH,
                    reasoning=f"Strong equity ({equity*100:.1f}%) - raise {size_pct*100:.0f}% ({size_reason})",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )

            # Semi-bluff raise with draws (30-50% equity)
            elif 0.30 <= equity < 0.50 and available.can_raise and self._rng.random() < 0.25:
                size_pct, size_reason = self._select_bet_size(context, equity, is_value=False)
                new_pot = pot + to_call
                raise_amount = state.current_bet + new_pot * size_pct
                raise_amount = max(available.min_raise, min(raise_amount, available.max_raise))

                return Decision(
                    action=Action.raise_to(raise_amount),
                    confidence=Confidence.LOW,
                    reasoning=f"Semi-bluff raise ({equity*100:.1f}% equity) - {size_pct*100:.0f}%",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )

            else:
                return Decision(
                    action=Action.call(to_call),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"Equity ({equity*100:.1f}%) > required ({required*100:.1f}%)",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )
        else:
            # Not enough equity
            # Check MDF for occasional bluff-catches
            mdf = self.solver.minimum_defense_frequency(to_call, pot)

            # Sometimes defend with marginal hands to stay unexploitable
            if equity > required * 0.7 and self._rng.random() < mdf * 0.3:
                return Decision(
                    action=Action.call(to_call),
                    confidence=Confidence.LOW,
                    reasoning=f"Defending at MDF ({mdf*100:.1f}%) with marginal hand",
                    equity=equity,
                    pot_odds=pot_odds,
                    required_equity=required,
                )

            return Decision(
                action=Action.fold(),
                confidence=Confidence.MEDIUM,
                reasoning=f"Equity ({equity*100:.1f}%) < required ({required*100:.1f}%)",
                equity=equity,
                pot_odds=pot_odds,
                required_equity=required,
            )

    def _checked_to_decision(
        self,
        context: DecisionContext,
        equity: float,
        available: AvailableActions,
    ) -> Decision:
        """Decision when checked to us with dynamic bet sizing."""
        state = context.game_state
        pot = state.pot.total
        player = state.players[state.action_on]
        stack = player.stack

        # Get recommended strategy from solver
        strategy = self.solver.suggest_strategy(pot, 0, equity)

        if equity > 0.65 and available.can_bet:
            # Value bet with dynamic sizing
            size_pct, size_reason = self._select_bet_size(context, equity, is_value=True)
            bet_size = pot * size_pct

            # Ensure bet is within legal bounds
            if available.min_bet:
                bet_size = max(available.min_bet, bet_size)
            bet_size = min(bet_size, stack)

            # All-in with very strong hands when bet would be large portion of stack
            if equity > 0.80 and bet_size > stack * 0.5:
                return Decision(
                    action=Action.all_in(stack),
                    confidence=Confidence.HIGH,
                    reasoning=f"Very strong ({equity*100:.1f}%) - all-in for value",
                    equity=equity,
                    ev_estimate=strategy.action_evs.get(ActionType.BET, 0),
                )

            return Decision(
                action=Action.bet(bet_size),
                confidence=Confidence.HIGH,
                reasoning=f"Value bet {size_pct*100:.0f}% pot ({size_reason})",
                equity=equity,
                ev_estimate=strategy.action_evs.get(ActionType.BET, 0),
            )

        elif equity > 0.45:
            # Medium strength - pot control or thin value
            # Sometimes bet small for thin value (merged range)
            if available.can_bet and self._rng.random() < 0.35:
                size_pct, size_reason = self._select_bet_size(context, equity, is_value=True)
                # Cap at 50% for medium strength
                size_pct = min(size_pct, 0.50)
                bet_size = pot * size_pct
                if available.min_bet:
                    bet_size = max(available.min_bet, bet_size)
                bet_size = min(bet_size, stack)

                return Decision(
                    action=Action.bet(bet_size),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"Thin value {size_pct*100:.0f}% pot ({size_reason})",
                    equity=equity,
                )

            if available.can_check:
                return Decision(
                    action=Action.check(),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"Medium equity ({equity*100:.1f}%) - pot control",
                    equity=equity,
                )
            else:
                # Forced to act - small bet
                bet_size = pot * 0.33
                if available.min_bet:
                    bet_size = max(available.min_bet, bet_size)
                return Decision(
                    action=Action.bet(min(bet_size, stack)),
                    confidence=Confidence.LOW,
                    reasoning=f"Medium equity ({equity*100:.1f}%) - small bet",
                    equity=equity,
                )

        else:
            # Weak - check or occasional bluff
            # GTO bluffs at optimal frequency
            bluff_freq = self.solver.optimal_bluff_frequency(pot * 0.67, pot)

            if available.can_bet and self._rng.random() < bluff_freq * 0.5:
                # Bluff with sizing matching our value bets
                size_pct, size_reason = self._select_bet_size(context, equity, is_value=False)
                bet_size = pot * size_pct
                if available.min_bet:
                    bet_size = max(available.min_bet, bet_size)
                bet_size = min(bet_size, stack)

                return Decision(
                    action=Action.bet(bet_size),
                    confidence=Confidence.LOW,
                    reasoning=f"Bluff {size_pct*100:.0f}% pot (balanced)",
                    equity=equity,
                )

            if available.can_check:
                return Decision(
                    action=Action.check(),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"Weak equity ({equity*100:.1f}%) - check",
                    equity=equity,
                )
            else:
                return Decision(
                    action=Action.fold(),
                    confidence=Confidence.HIGH,
                    reasoning=f"Weak equity ({equity*100:.1f}%) - fold",
                    equity=equity,
                )

    def analyze_spot(
        self,
        hand: Hand | str,
        board: list[Card] | str,
        villain_range: HandRange | str,
        pot: float,
        to_call: float = 0,
    ) -> dict:
        """
        Analyze a specific spot.

        Args:
            hand: Hero's hand
            board: Community cards
            villain_range: Opponent's range
            pot: Current pot
            to_call: Amount to call

        Returns:
            Analysis dict with equity, recommendations, etc.
        """
        if isinstance(hand, str):
            hand = Hand(hand)
        if isinstance(board, str):
            board = self.equity_calc._parse_board(board)
        if isinstance(villain_range, str):
            villain_range = HandRange(villain_range)

        # Calculate equity
        equity_result = self.equity_calc.hand_vs_range(hand, villain_range, board)

        # Calculate pot odds
        required_equity = self.solver.required_equity_to_call(to_call, pot) if to_call > 0 else 0
        pot_odds = pot / to_call if to_call > 0 else float('inf')

        # Get strategy suggestion
        strategy = self.solver.suggest_strategy(pot, to_call, equity_result.equity)

        return {
            "hand": str(hand),
            "board": " ".join(str(c) for c in board),
            "equity": equity_result.equity,
            "equity_pct": f"{equity_result.equity * 100:.1f}%",
            "confidence_interval": equity_result.confidence_interval,
            "pot": pot,
            "to_call": to_call,
            "pot_odds": pot_odds,
            "required_equity": required_equity,
            "required_equity_pct": f"{required_equity * 100:.1f}%",
            "profitable_call": equity_result.equity >= required_equity,
            "strategy": {
                "fold": f"{strategy.fold_freq * 100:.1f}%",
                "check": f"{strategy.check_freq * 100:.1f}%",
                "call": f"{strategy.call_freq * 100:.1f}%",
                "bet": f"{strategy.bet_freq * 100:.1f}%",
                "raise": f"{strategy.raise_freq * 100:.1f}%",
            },
            "evs": strategy.action_evs,
        }
