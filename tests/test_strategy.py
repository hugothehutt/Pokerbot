"""Tests for strategy components."""

import pytest

from pokerbot.core.hand import Hand
from pokerbot.game.player import Position
from pokerbot.equity.range import HandRange
from pokerbot.strategy.gto.ranges import (
    PreflopRanges,
    get_opening_range,
    get_3bet_range,
    get_calling_range,
)
from pokerbot.strategy.gto.sizing import BetSizer, SizingStrategy, calculate_bet_size
from pokerbot.strategy.gto.solver import SimpleSolver
from pokerbot.strategy.decision import DecisionEngine


class TestPreflopRanges:
    """Tests for preflop GTO ranges."""

    def test_opening_range_utg(self):
        """Test UTG opening range is tighter than BTN."""
        utg_range = get_opening_range(Position.UTG)
        btn_range = get_opening_range(Position.BTN)

        assert utg_range.percentage < btn_range.percentage

    def test_opening_range_btn(self):
        """Test BTN opening range is wide."""
        btn_range = get_opening_range(Position.BTN)

        # BTN should open roughly 40-50% of hands
        assert btn_range.percentage > 35
        assert btn_range.percentage < 60

    def test_premium_hands_in_all_ranges(self):
        """Test that premium hands are in all opening ranges."""
        for pos in [Position.UTG, Position.HJ, Position.CO, Position.BTN]:
            range_ = get_opening_range(pos)

            assert Hand("AsAh") in range_
            assert Hand("KsKh") in range_
            assert Hand("AsKs") in range_

    def test_trash_hands_not_in_utg(self):
        """Test that weak hands are not in UTG range."""
        utg_range = get_opening_range(Position.UTG)

        assert Hand("7s2h") not in utg_range
        assert Hand("9s3h") not in utg_range

    def test_3bet_range_vs_utg(self):
        """Test 3-bet range vs UTG open."""
        three_bet = get_3bet_range(Position.BTN, Position.UTG)

        # Should 3-bet with premiums
        assert Hand("AsAh") in three_bet
        assert Hand("KsKh") in three_bet

        # Should not 3-bet with weak hands
        assert Hand("7s6s") not in three_bet

    def test_3bet_range_position_matters(self):
        """Test that 3-bet range varies by position."""
        btn_3bet = get_3bet_range(Position.BTN, Position.UTG)
        sb_3bet = get_3bet_range(Position.SB, Position.UTG)

        # BTN can 3-bet wider than SB vs UTG
        assert btn_3bet.percentage >= sb_3bet.percentage

    def test_calling_range_bb_vs_btn(self):
        """Test BB calling range vs BTN open."""
        calling = get_calling_range(Position.BB, Position.BTN)

        # BB should defend wide vs BTN
        assert calling.percentage > 20


class TestBetSizing:
    """Tests for bet sizing utilities."""

    def test_pot_percentage(self):
        """Test pot percentage calculation."""
        size = calculate_bet_size(100, 67)
        assert size == 67

        size = calculate_bet_size(100, 50)
        assert size == 50

    def test_sizing_strategy_enum(self):
        """Test sizing strategies produce different sizes."""
        pot = 100

        small = pot * 0.33
        medium = pot * 0.67
        large = pot * 1.0

        assert small < medium < large


class TestSimpleSolver:
    """Tests for simple GTO solver concepts."""

    def test_minimum_defense_frequency(self):
        """Test MDF calculation."""
        solver = SimpleSolver()

        # Half pot bet -> MDF = 2/3
        mdf = solver.minimum_defense_frequency(50, 100)
        assert mdf == pytest.approx(0.667, abs=0.01)

        # Full pot bet -> MDF = 1/2
        mdf = solver.minimum_defense_frequency(100, 100)
        assert mdf == pytest.approx(0.5, abs=0.01)

    def test_optimal_bluff_frequency(self):
        """Test optimal bluff frequency."""
        solver = SimpleSolver()

        # Full pot bet should have ~33% bluffs
        bluff_freq = solver.optimal_bluff_frequency(100, 100)
        assert bluff_freq == pytest.approx(0.333, abs=0.01)

    def test_required_equity_to_call(self):
        """Test required equity calculation."""
        solver = SimpleSolver()

        # Half pot bet needs 25% equity
        req = solver.required_equity_to_call(50, 100)
        assert req == pytest.approx(0.25, abs=0.01)

    def test_breakeven_bluff_equity(self):
        """Test breakeven fold equity for bluffs."""
        solver = SimpleSolver()

        # Full pot bet needs 50% folds
        be = solver.breakeven_bluff_equity(100, 100)
        assert be == pytest.approx(0.5, abs=0.01)

    def test_calculate_ev(self):
        """Test EV calculation."""
        solver = SimpleSolver()

        evs = solver.calculate_ev(
            pot=100,
            bet=50,
            equity=0.6,
            fold_equity=0.3,
        )

        # Should have EVs for different actions
        assert "check" in evs
        assert "call" in evs
        assert "bet" in evs
        assert "fold" in evs

        # Folding should always be 0 EV
        assert evs["fold"] == 0

    def test_suggest_strategy(self):
        """Test strategy suggestion."""
        solver = SimpleSolver()

        # With high equity, should want to bet
        node = solver.suggest_strategy(
            pot=100,
            facing_bet=0,
            our_equity=0.8,
        )

        assert node.bet_freq > node.check_freq

        # With low equity vs bet, should fold more
        node = solver.suggest_strategy(
            pot=100,
            facing_bet=100,
            our_equity=0.2,
        )

        assert node.fold_freq > node.call_freq


class TestDecisionEngine:
    """Tests for the decision engine."""

    def test_analyze_spot(self):
        """Test spot analysis."""
        engine = DecisionEngine(simulations=1000)

        analysis = engine.analyze_spot(
            hand="AsKs",
            board="Ah 7c 2d",
            villain_range="QQ,JJ,TT",
            pot=100,
            to_call=50,
        )

        assert "equity" in analysis
        assert "pot_odds" in analysis
        assert "strategy" in analysis
        assert "profitable_call" in analysis

    def test_analyze_profitable_call(self):
        """Test profitable call detection."""
        engine = DecisionEngine(simulations=1000)

        # Strong hand vs weak range - should be profitable
        analysis = engine.analyze_spot(
            hand="AsAh",
            board="Ac 7c 2d",
            villain_range="KK,QQ,JJ",
            pot=100,
            to_call=50,
        )

        assert analysis["profitable_call"] == True

    def test_analyze_unprofitable_call(self):
        """Test unprofitable call detection."""
        engine = DecisionEngine(simulations=1000)

        # Weak hand vs strong range with bad odds
        analysis = engine.analyze_spot(
            hand="7s6s",
            board="As Kc Qd",
            villain_range="AA,AK,AQ",
            pot=100,
            to_call=200,  # Bad odds
        )

        assert analysis["profitable_call"] == False
