"""
Strategy representation for CFR.

Stores the learned strategy (action probabilities) for each information set.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import random
import json
import os

from pokerbot.strategy.cfr.abstraction import ActionBucket, HandBucket


@dataclass
class InfoSetData:
    """
    Data stored for each information set.

    Contains:
    - Cumulative regrets for each action
    - Cumulative strategy for each action
    - Number of times visited
    """
    regret_sum: Dict[ActionBucket, float] = field(default_factory=dict)
    strategy_sum: Dict[ActionBucket, float] = field(default_factory=dict)
    visit_count: int = 0

    def get_strategy(self) -> Dict[ActionBucket, float]:
        """
        Get current strategy via regret matching.

        Strategy is proportional to positive regrets.
        """
        positive_regrets = {
            a: max(0, r) for a, r in self.regret_sum.items()
        }
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: r / total for a, r in positive_regrets.items()}
        else:
            # Uniform distribution when no positive regrets
            actions = list(self.regret_sum.keys())
            if not actions:
                return {}
            return {a: 1.0 / len(actions) for a in actions}

    def get_average_strategy(self) -> Dict[ActionBucket, float]:
        """
        Get average strategy over all iterations.

        This is the Nash equilibrium approximation.
        """
        total = sum(self.strategy_sum.values())

        if total > 0:
            return {a: s / total for a, s in self.strategy_sum.items()}
        else:
            return self.get_strategy()

    def update_regret(self, action: ActionBucket, regret: float):
        """Add regret for an action."""
        if action not in self.regret_sum:
            self.regret_sum[action] = 0
        self.regret_sum[action] += regret

    def update_strategy(self, action: ActionBucket, probability: float, reach_prob: float):
        """Update cumulative strategy."""
        if action not in self.strategy_sum:
            self.strategy_sum[action] = 0
        self.strategy_sum[action] += reach_prob * probability

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "regret_sum": {a.name: v for a, v in self.regret_sum.items()},
            "strategy_sum": {a.name: v for a, v in self.strategy_sum.items()},
            "visit_count": self.visit_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InfoSetData":
        """Create from dict."""
        obj = cls()
        obj.regret_sum = {
            ActionBucket[k]: v for k, v in data.get("regret_sum", {}).items()
        }
        obj.strategy_sum = {
            ActionBucket[k]: v for k, v in data.get("strategy_sum", {}).items()
        }
        obj.visit_count = data.get("visit_count", 0)
        return obj


class CFRStrategy:
    """
    Manages strategy storage and lookup.
    """

    def __init__(self):
        self.info_sets: Dict[str, InfoSetData] = {}
        self._rng = random.Random()

    def get_info_set(self, key: str) -> InfoSetData:
        """Get or create info set data."""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetData()
        return self.info_sets[key]

    def get_action(self, key: str, available_actions: list[ActionBucket]) -> ActionBucket:
        """
        Sample an action based on current strategy.

        Args:
            key: Information set key
            available_actions: Actions available in this state

        Returns:
            Sampled action
        """
        info_set = self.get_info_set(key)

        # Initialize regrets for new actions
        for action in available_actions:
            if action not in info_set.regret_sum:
                info_set.regret_sum[action] = 0

        strategy = info_set.get_strategy()

        # Filter to available actions and renormalize
        filtered = {a: strategy.get(a, 0) for a in available_actions}
        total = sum(filtered.values())

        if total == 0:
            # Uniform over available
            return self._rng.choice(available_actions)

        # Sample according to strategy
        r = self._rng.random() * total
        cumulative = 0
        for action, prob in filtered.items():
            cumulative += prob
            if r <= cumulative:
                return action

        return available_actions[-1]

    def get_strategy_for_info_set(self, key: str) -> Dict[ActionBucket, float]:
        """Get average (Nash) strategy for an info set."""
        if key not in self.info_sets:
            return {}
        return self.info_sets[key].get_average_strategy()

    def get_exploitability(self) -> float:
        """
        Estimate exploitability of current strategy.

        Lower is better. Zero means Nash equilibrium.
        """
        # This is a simplified estimate
        total_regret = 0
        count = 0

        for info_set in self.info_sets.values():
            for regret in info_set.regret_sum.values():
                total_regret += max(0, regret)
                count += 1

        return total_regret / max(count, 1)

    def save(self, filepath: str):
        """Save strategy to file."""
        data = {
            key: info_set.to_dict()
            for key, info_set in self.info_sets.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load strategy from file."""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.info_sets = {
            key: InfoSetData.from_dict(info_data)
            for key, info_data in data.items()
        }

    def __len__(self):
        return len(self.info_sets)


class StrategyProfile:
    """
    Holds strategies for multiple players.
    """

    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.strategies = [CFRStrategy() for _ in range(num_players)]

    def get_strategy(self, player: int) -> CFRStrategy:
        """Get strategy for a player."""
        return self.strategies[player]

    def get_action(self, player: int, key: str,
                   available_actions: list[ActionBucket]) -> ActionBucket:
        """Get action for player at info set."""
        return self.strategies[player].get_action(key, available_actions)

    def save(self, directory: str):
        """Save all strategies."""
        os.makedirs(directory, exist_ok=True)
        for i, strategy in enumerate(self.strategies):
            strategy.save(os.path.join(directory, f"player_{i}.json"))

    def load(self, directory: str):
        """Load all strategies."""
        for i, strategy in enumerate(self.strategies):
            filepath = os.path.join(directory, f"player_{i}.json")
            if os.path.exists(filepath):
                strategy.load(filepath)
