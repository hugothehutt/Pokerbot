"""Simple GTO solver concepts for poker strategy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum, auto

from pokerbot.game.action import ActionType
from pokerbot.equity.range import HandRange


class NodeType(Enum):
    """Types of nodes in a game tree."""
    DECISION = auto()    # Player decision point
    CHANCE = auto()      # Card dealt
    TERMINAL = auto()    # Hand ends


@dataclass
class StrategyNode:
    """
    Represents a node in the game tree with strategy information.

    This is a simplified representation for understanding GTO concepts,
    not a full solver implementation.
    """
    node_type: NodeType
    player: int = 0  # 0 = hero, 1 = villain
    pot: float = 0.0
    stack: float = 0.0

    # Strategy frequencies (sum to 1.0)
    fold_freq: float = 0.0
    check_freq: float = 0.0
    call_freq: float = 0.0
    bet_freq: float = 0.0
    raise_freq: float = 0.0

    # Ranges for each action
    fold_range: Optional[HandRange] = None
    check_range: Optional[HandRange] = None
    call_range: Optional[HandRange] = None
    bet_range: Optional[HandRange] = None
    raise_range: Optional[HandRange] = None

    # EV of each action
    action_evs: Dict[ActionType, float] = field(default_factory=dict)

    # Child nodes
    children: Dict[ActionType, "StrategyNode"] = field(default_factory=dict)

    def get_strategy_string(self) -> str:
        """Get human-readable strategy summary."""
        parts = []
        if self.fold_freq > 0:
            parts.append(f"Fold: {self.fold_freq*100:.1f}%")
        if self.check_freq > 0:
            parts.append(f"Check: {self.check_freq*100:.1f}%")
        if self.call_freq > 0:
            parts.append(f"Call: {self.call_freq*100:.1f}%")
        if self.bet_freq > 0:
            parts.append(f"Bet: {self.bet_freq*100:.1f}%")
        if self.raise_freq > 0:
            parts.append(f"Raise: {self.raise_freq*100:.1f}%")
        return " | ".join(parts) if parts else "No strategy"


class SimpleSolver:
    """
    Simple GTO solver for basic poker situations.

    This implements basic GTO concepts like:
    - Minimum Defense Frequency (MDF)
    - Optimal bluffing frequency
    - Pot odds calculations

    Note: This is educational and not a full solver.
    Real solvers use CFR (Counterfactual Regret Minimization).
    """

    @staticmethod
    def minimum_defense_frequency(bet_size: float, pot: float) -> float:
        """
        Calculate Minimum Defense Frequency (MDF).

        MDF = Pot / (Pot + Bet)

        This is the minimum frequency you must defend to prevent
        villain from profiting with any two cards.

        Args:
            bet_size: Size of villain's bet
            pot: Pot before villain's bet

        Returns:
            MDF as decimal (0-1)
        """
        return pot / (pot + bet_size)

    @staticmethod
    def optimal_bluff_frequency(bet_size: float, pot: float) -> float:
        """
        Calculate optimal bluffing frequency for a bet.

        Bluff% = Bet / (Pot + 2*Bet)

        This makes villain indifferent to calling with bluff-catchers.

        Args:
            bet_size: Size of bet
            pot: Pot before bet

        Returns:
            Optimal bluff frequency (0-1)
        """
        return bet_size / (pot + 2 * bet_size)

    @staticmethod
    def value_to_bluff_ratio(bet_size: float, pot: float) -> float:
        """
        Calculate optimal value-to-bluff ratio.

        For a balanced betting range:
        Value hands : Bluff hands = (Pot + Bet) : Bet

        Args:
            bet_size: Size of bet
            pot: Pot before bet

        Returns:
            Ratio of value bets to bluffs
        """
        return (pot + bet_size) / bet_size

    @staticmethod
    def required_equity_to_call(bet_size: float, pot: float) -> float:
        """
        Calculate equity needed to call profitably.

        Required equity = Bet / (Pot + Bet + Bet)

        Args:
            bet_size: Size of bet to call
            pot: Current pot

        Returns:
            Required equity (0-1)
        """
        return bet_size / (pot + 2 * bet_size)

    @staticmethod
    def breakeven_bluff_equity(bet_size: float, pot: float) -> float:
        """
        Calculate breakeven fold equity needed for a bluff.

        Breakeven = Bet / (Pot + Bet)

        Args:
            bet_size: Size of bluff bet
            pot: Current pot

        Returns:
            Required fold equity (0-1)
        """
        return bet_size / (pot + bet_size)

    def calculate_ev(
        self,
        pot: float,
        bet: float,
        equity: float,
        fold_equity: float = 0.0,
    ) -> dict[str, float]:
        """
        Calculate EV for different actions.

        Args:
            pot: Current pot
            bet: Bet size being considered
            equity: Our equity vs villain's range
            fold_equity: Probability villain folds

        Returns:
            Dict with EV for each action
        """
        evs = {}

        # EV of checking (if possible)
        evs["check"] = equity * pot

        # EV of calling (when facing bet)
        evs["call"] = equity * (pot + bet) - (1 - equity) * bet

        # EV of betting/raising
        ev_when_fold = pot
        ev_when_called = equity * (pot + 2 * bet) - (1 - equity) * bet
        evs["bet"] = fold_equity * ev_when_fold + (1 - fold_equity) * ev_when_called

        # EV of folding
        evs["fold"] = 0

        return evs

    def suggest_strategy(
        self,
        pot: float,
        facing_bet: float,
        our_equity: float,
        villain_bluff_freq: float = 0.33,
    ) -> StrategyNode:
        """
        Suggest a basic strategy for a spot.

        Args:
            pot: Current pot size
            facing_bet: Bet we're facing (0 if checked to us)
            our_equity: Our equity vs villain's range
            villain_bluff_freq: Estimated villain bluff frequency

        Returns:
            StrategyNode with suggested frequencies
        """
        node = StrategyNode(
            node_type=NodeType.DECISION,
            pot=pot,
        )

        if facing_bet > 0:
            # Facing a bet - decide call/fold/raise
            mdf = self.minimum_defense_frequency(facing_bet, pot)
            req_equity = self.required_equity_to_call(facing_bet, pot)

            if our_equity >= req_equity:
                # We have enough equity to call
                # Mix calls and raises with strong hands
                if our_equity > 0.7:
                    node.call_freq = 0.5
                    node.raise_freq = 0.5
                else:
                    node.call_freq = min(1.0, our_equity / req_equity * mdf)
                    node.fold_freq = 1.0 - node.call_freq
            else:
                # Not enough equity - mostly fold
                # But defend MDF with bluff-catchers
                node.fold_freq = 1.0 - mdf
                node.call_freq = mdf

        else:
            # Checked to us - decide bet/check
            # Bet with value hands and balanced bluffs
            if our_equity > 0.65:
                # Strong - bet for value
                node.bet_freq = 0.8
                node.check_freq = 0.2
            elif our_equity > 0.45:
                # Medium - mostly check
                node.bet_freq = 0.3
                node.check_freq = 0.7
            else:
                # Weak - check or bluff
                optimal_bluff = self.optimal_bluff_frequency(pot * 0.67, pot)
                node.bet_freq = min(optimal_bluff, 0.3)  # Cap bluffs
                node.check_freq = 1.0 - node.bet_freq

        # Calculate EVs
        node.action_evs = self.calculate_ev(pot, facing_bet or pot * 0.67, our_equity)

        return node


def get_gto_action_frequencies(
    pot: float,
    bet_size: float,
    equity: float,
) -> dict[ActionType, float]:
    """
    Get GTO-approximated action frequencies.

    Args:
        pot: Current pot
        bet_size: Bet facing (0 if none)
        equity: Our equity (0-1)

    Returns:
        Dict mapping action types to frequencies
    """
    solver = SimpleSolver()
    node = solver.suggest_strategy(pot, bet_size, equity)

    return {
        ActionType.FOLD: node.fold_freq,
        ActionType.CHECK: node.check_freq,
        ActionType.CALL: node.call_freq,
        ActionType.BET: node.bet_freq,
        ActionType.RAISE: node.raise_freq,
    }
