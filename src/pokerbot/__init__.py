"""
Pokerbot - A Texas Hold'em poker strategy analysis bot.

Features:
- Monte Carlo simulation for equity calculations
- GTO (Game Theory Optimal) concepts
- Hand range analysis
- Position-aware strategy recommendations
"""

__version__ = "0.1.0"

from pokerbot.core.card import Card
from pokerbot.core.deck import Deck
from pokerbot.core.hand import Hand
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.equity.monte_carlo import MonteCarloSimulator
from pokerbot.equity.range import HandRange

__all__ = [
    "Card",
    "Deck",
    "Hand",
    "HandEvaluator",
    "MonteCarloSimulator",
    "HandRange",
]
