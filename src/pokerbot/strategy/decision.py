"""Decision engine combining equity, GTO concepts, and opponent modeling."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

from pokerbot.core.hand import Hand
from pokerbot.core.card import Card
from pokerbot.game.state import GameState, Street
from pokerbot.game.action import Action, ActionType, AvailableActions
from pokerbot.game.player import Position
from pokerbot.equity.calculator import EquityCalculator
from pokerbot.equity.range import HandRange
from pokerbot.strategy.gto.ranges import PreflopRanges, get_opening_range
from pokerbot.strategy.gto.sizing import BetSizer, SizingStrategy
from pokerbot.strategy.gto.solver import SimpleSolver
from pokerbot.strategy.opponent import OpponentModel, OpponentProfile


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
        """Make postflop decision."""
        state = context.game_state
        hand = context.hero_hand
        available = state.get_available_actions()

        # Calculate equity
        villain_range = context.villain_range or HandRange("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,A9o+,KTo+,QTo+,JTo")
        equity_result = self.equity_calc.hand_vs_range(
            hand,
            villain_range,
            state.board,
        )
        equity = equity_result.equity

        pot = state.pot.total
        to_call = state.current_bet - state.players[state.action_on].bet_this_round

        if to_call > 0:
            # Facing a bet
            return self._facing_bet_decision(context, equity, available)
        else:
            # Checked to us
            return self._checked_to_decision(context, equity, available)

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

        # Calculate required equity
        required = self.solver.required_equity_to_call(to_call, pot)
        pot_odds = pot / to_call if to_call > 0 else float('inf')

        # Get opponent bluff frequency estimate
        bluff_freq = 0.33  # Default
        if self.opponent_model and context.opponent_profile:
            bluff_freq = context.opponent_profile.bluff_frequency

        # Adjust equity for opponent tendencies
        # If they bluff less, we need more real equity to call

        if equity >= required:
            # Profitable call
            # Consider raising with strong hands
            if equity > 0.7 and available.can_raise:
                sizer = BetSizer(state)
                raise_to = sizer.get_raise_size(state.current_bet, SizingStrategy.LARGE)

                return Decision(
                    action=Action.raise_to(min(raise_to, available.max_raise)),
                    confidence=Confidence.HIGH,
                    reasoning=f"Strong equity ({equity*100:.1f}%) - raise for value",
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
            # Not enough equity - fold
            # Unless MDF suggests we need to defend
            mdf = self.solver.minimum_defense_frequency(to_call, pot)

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
        """Decision when checked to us."""
        state = context.game_state
        pot = state.pot.total

        # Get recommended strategy from solver
        strategy = self.solver.suggest_strategy(pot, 0, equity)

        if equity > 0.65 and available.can_bet:
            # Value bet
            sizer = BetSizer(state)
            bet_size = sizer.get_bet_size(SizingStrategy.MEDIUM)

            return Decision(
                action=Action.bet(bet_size),
                confidence=Confidence.HIGH,
                reasoning=f"Strong equity ({equity*100:.1f}%) - bet for value",
                equity=equity,
                ev_estimate=strategy.action_evs.get(ActionType.BET, 0),
            )
        elif equity > 0.45:
            # Medium strength - check for pot control or thin value
            if available.can_check:
                return Decision(
                    action=Action.check(),
                    confidence=Confidence.MEDIUM,
                    reasoning=f"Medium equity ({equity*100:.1f}%) - pot control",
                    equity=equity,
                )
            else:
                sizer = BetSizer(state)
                bet_size = sizer.get_bet_size(SizingStrategy.SMALL)
                return Decision(
                    action=Action.bet(bet_size),
                    confidence=Confidence.LOW,
                    reasoning=f"Medium equity ({equity*100:.1f}%) - small bet",
                    equity=equity,
                )
        else:
            # Weak - check/fold or bluff occasionally
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
