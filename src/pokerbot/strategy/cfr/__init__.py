"""
Counterfactual Regret Minimization (CFR) implementation.

This module provides:
- KuhnCFR: Simple 3-card poker for testing CFR correctness
- CFRSolver: Full poker CFR with hand/action abstraction
- CFRTrainer: High-level training interface with checkpointing
- PrecomputedStrategy: Pre-computed GTO-approximated strategies (instant)

Quick Start:
    # For testing CFR algorithm (fast)
    from pokerbot.strategy.cfr import train_kuhn_poker
    solver = train_kuhn_poker(10000)

    # For playing (use precomputed strategies)
    from pokerbot.strategy.cfr import get_precomputed_strategy
    strategy = get_precomputed_strategy()
    action = strategy.get_action(hand_bucket, street, facing_bet)

    # For training full poker (slow but learns)
    from pokerbot.strategy.cfr import CFRTrainer
    trainer = CFRTrainer(stack_size=100.0)
    trainer.train(iterations=10000)
"""

from pokerbot.strategy.cfr.algorithm import CFRSolver, CFRTrainer
from pokerbot.strategy.cfr.abstraction import (
    ActionAbstraction, HandAbstraction,
    ActionBucket, HandBucket
)
from pokerbot.strategy.cfr.strategy import CFRStrategy, StrategyProfile
from pokerbot.strategy.cfr.kuhn_poker import KuhnCFR, train_kuhn_poker
from pokerbot.strategy.cfr.precomputed import PrecomputedStrategy, get_precomputed_strategy

__all__ = [
    # Core algorithm
    "CFRSolver",
    "CFRTrainer",
    # Abstraction
    "ActionAbstraction",
    "HandAbstraction",
    "ActionBucket",
    "HandBucket",
    # Strategy
    "CFRStrategy",
    "StrategyProfile",
    # Kuhn Poker (for testing)
    "KuhnCFR",
    "train_kuhn_poker",
    # Precomputed (for playing)
    "PrecomputedStrategy",
    "get_precomputed_strategy",
]
