"""Tests for Monte Carlo simulation and equity calculations."""

import pytest

from pokerbot.core.card import Card
from pokerbot.core.hand import Hand
from pokerbot.equity.monte_carlo import MonteCarloSimulator, EquityResult
from pokerbot.equity.range import HandRange
from pokerbot.equity.calculator import EquityCalculator


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def test_hand_vs_hand_equity(self):
        """Test basic hand vs hand equity calculation."""
        simulator = MonteCarloSimulator(num_simulations=5000, seed=42)

        hand1 = Hand("AsKs")
        hand2 = Hand("QhQd")

        result = simulator.calculate_equity(hand1, hand2)

        # AKs vs QQ is roughly 43% vs 57%
        assert 0.35 < result.equity < 0.55
        assert result.num_simulations > 0

    def test_hand_vs_hand_with_board(self):
        """Test equity calculation with board cards."""
        simulator = MonteCarloSimulator(num_simulations=5000, seed=42)

        hand1 = Hand("AsKs")
        hand2 = Hand("QhQd")
        board = [Card("Ah"), Card("7c"), Card("2d")]  # Ace on board

        result = simulator.calculate_equity(hand1, hand2, board)

        # AK hit top pair, should be ahead
        assert result.equity > 0.7

    def test_hand_vs_range_equity(self):
        """Test hand vs range equity."""
        simulator = MonteCarloSimulator(num_simulations=5000, seed=42)

        hand = Hand("AsKs")
        villain_range = HandRange("QQ,JJ,TT")

        result = simulator.calculate_equity(hand, villain_range)

        # AKs vs QQ-TT is roughly 45%
        assert 0.35 < result.equity < 0.55

    def test_range_vs_range_equity(self):
        """Test range vs range equity."""
        simulator = MonteCarloSimulator(num_simulations=3000, seed=42)

        range1 = HandRange("AA,KK")
        range2 = HandRange("QQ,JJ")

        result = simulator.calculate_equity(range1, range2)

        # AA,KK vs QQ,JJ - premium pairs should be ahead
        assert result.equity > 0.7

    def test_equity_confidence_interval(self):
        """Test confidence interval is reasonable."""
        simulator = MonteCarloSimulator(num_simulations=10000, seed=42)

        hand1 = Hand("AsKs")
        hand2 = Hand("QhQd")

        result = simulator.calculate_equity(hand1, hand2)

        # CI should contain the equity
        assert result.confidence_interval[0] <= result.equity
        assert result.equity <= result.confidence_interval[1]

        # CI width should be reasonable for 10k sims
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        assert ci_width < 0.05  # Less than 5% width

    def test_multiway_equity(self):
        """Test multiway equity calculation."""
        simulator = MonteCarloSimulator(num_simulations=3000, seed=42)

        hands = [
            Hand("AsKs"),
            Hand("QhQd"),
            Hand("JcJd"),
        ]

        result = simulator.calculate_multiway_equity(hands)

        assert len(result.equities) == 3
        assert sum(result.equities) == pytest.approx(1.0, abs=0.05)

    def test_dead_cards_excluded(self):
        """Test that dead cards are properly excluded."""
        simulator = MonteCarloSimulator(num_simulations=1000, seed=42)

        hand1 = Hand("AsKs")
        hand2 = Hand("QhQd")
        dead = [Card("Ah")]  # Dead ace

        result = simulator.calculate_equity(hand1, hand2, dead_cards=dead)

        # Should complete without error
        assert result.num_simulations > 0


class TestEquityCalculator:
    """Tests for EquityCalculator class."""

    def test_hand_vs_hand(self):
        """Test high-level hand vs hand calculation."""
        calc = EquityCalculator(default_simulations=5000)

        result = calc.hand_vs_hand("AsKs", "QhQd")

        assert 0.35 < result.equity < 0.55

    def test_hand_vs_hand_string_board(self):
        """Test with board as string."""
        calc = EquityCalculator(default_simulations=5000)

        result = calc.hand_vs_hand("AsKs", "QhQd", "Ah 7c 2d")

        assert result.equity > 0.7  # Hit ace

    def test_hand_vs_range(self):
        """Test hand vs range calculation."""
        calc = EquityCalculator(default_simulations=3000)

        result = calc.hand_vs_range("AsKs", "TT+,AQs+")

        assert 0.3 < result.equity < 0.6

    def test_range_vs_range(self):
        """Test range vs range calculation."""
        calc = EquityCalculator(default_simulations=3000)

        result = calc.range_vs_range("AA,KK", "QQ,JJ,TT")

        assert result.equity > 0.7

    def test_multiway(self):
        """Test multiway calculation."""
        calc = EquityCalculator(default_simulations=3000)

        result = calc.multiway(["AsKs", "QhQd", "JcJd"])

        assert len(result.equities) == 3

    def test_preflop_equity_fast(self):
        """Test fast preflop equity lookup."""
        calc = EquityCalculator()

        equity_aa = calc.preflop_equity("AsAh")
        equity_72 = calc.preflop_equity("7s2h")

        assert equity_aa > 0.8
        assert equity_72 < 0.4

    def test_required_equity_to_call(self):
        """Test pot odds calculation."""
        calc = EquityCalculator()

        # Pot 100, call 50 -> need 33%
        required = calc.required_equity_to_call(100, 50)
        assert required == pytest.approx(0.333, abs=0.01)

        # Pot 100, call 100 -> need 50%
        required = calc.required_equity_to_call(100, 100)
        assert required == pytest.approx(0.5, abs=0.01)

    def test_ev_of_call(self):
        """Test EV calculation for calling."""
        calc = EquityCalculator()

        # Pot 100, call 50, 50% equity -> EV positive
        ev = calc.ev_of_call(100, 50, 0.5)
        assert ev > 0

        # Pot 100, call 50, 20% equity -> EV negative
        ev = calc.ev_of_call(100, 50, 0.2)
        assert ev < 0


class TestHandRange:
    """Tests for HandRange class."""

    def test_parse_single_hand(self):
        """Test parsing single hand notation."""
        range_ = HandRange("AKs")

        assert range_.num_notations == 1
        assert range_.num_combinations == 4

    def test_parse_pair(self):
        """Test parsing pocket pair."""
        range_ = HandRange("AA")

        assert range_.num_notations == 1
        assert range_.num_combinations == 6

    def test_parse_plus_notation(self):
        """Test parsing plus notation."""
        range_ = HandRange("TT+")

        # TT, JJ, QQ, KK, AA = 5 pairs * 6 combos = 30
        assert range_.num_combinations == 30

    def test_parse_range_notation(self):
        """Test parsing range notation."""
        range_ = HandRange("AA-TT")

        # Same as TT+
        assert range_.num_combinations == 30

    def test_parse_combined(self):
        """Test parsing combined notation."""
        range_ = HandRange("AA,KK,AKs")

        # 6 + 6 + 4 = 16
        assert range_.num_combinations == 16

    def test_hand_in_range(self):
        """Test checking if hand is in range."""
        range_ = HandRange("AA,KK,AKs")

        assert Hand("AsAh") in range_
        assert Hand("AsKs") in range_
        assert Hand("AsKh") not in range_  # AKo not in range

    def test_range_percentage(self):
        """Test range percentage calculation."""
        range_ = HandRange("AA")

        # 6/1326 ≈ 0.45%
        assert range_.percentage < 1.0

    def test_range_union(self):
        """Test range union."""
        range1 = HandRange("AA")
        range2 = HandRange("KK")
        combined = range1 | range2

        assert combined.num_combinations == 12

    def test_range_intersection(self):
        """Test range intersection."""
        range1 = HandRange("AA,KK,QQ")
        range2 = HandRange("QQ,JJ,TT")
        intersection = range1 & range2

        assert intersection.num_combinations == 6  # Only QQ
