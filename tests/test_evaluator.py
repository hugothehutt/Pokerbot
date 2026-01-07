"""Tests for hand evaluation."""

import pytest

from pokerbot.core.card import Card
from pokerbot.core.hand import Hand
from pokerbot.core.evaluator import HandEvaluator, HandRank, evaluate, compare


class TestHandEvaluator:
    """Tests for HandEvaluator class."""

    def test_evaluate_high_card(self):
        """Test high card evaluation."""
        hand = Hand("As2h")
        board = [Card("7c"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.HIGH_CARD

    def test_evaluate_pair(self):
        """Test pair evaluation."""
        hand = Hand("AsAh")
        board = [Card("7c"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.ONE_PAIR

    def test_evaluate_two_pair(self):
        """Test two pair evaluation."""
        hand = Hand("AsAh")
        board = [Card("7c"), Card("7d"), Card("Tc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.TWO_PAIR

    def test_evaluate_three_of_a_kind(self):
        """Test three of a kind evaluation."""
        hand = Hand("AsAh")
        board = [Card("Ac"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.THREE_OF_A_KIND

    def test_evaluate_straight(self):
        """Test straight evaluation."""
        hand = Hand("9s8h")
        board = [Card("7c"), Card("6d"), Card("5c"), Card("3s"), Card("2h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.STRAIGHT

    def test_evaluate_flush(self):
        """Test flush evaluation."""
        hand = Hand("As2s")
        board = [Card("7s"), Card("8s"), Card("Ts"), Card("3d"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.FLUSH

    def test_evaluate_full_house(self):
        """Test full house evaluation."""
        hand = Hand("AsAh")
        board = [Card("Ac"), Card("Kd"), Card("Kc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.FULL_HOUSE

    def test_evaluate_four_of_a_kind(self):
        """Test four of a kind evaluation."""
        hand = Hand("AsAh")
        board = [Card("Ac"), Card("Ad"), Card("Kc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.FOUR_OF_A_KIND

    def test_evaluate_straight_flush(self):
        """Test straight flush evaluation."""
        hand = Hand("9s8s")
        board = [Card("7s"), Card("6s"), Card("5s"), Card("3d"), Card("2h")]

        result = HandEvaluator.evaluate(hand, board)

        assert result.rank == HandRank.STRAIGHT_FLUSH

    def test_compare_hands(self):
        """Test comparing two hands."""
        hand1 = Hand("AsAh")  # Pair of aces
        hand2 = Hand("KsKh")  # Pair of kings
        board = [Card("7c"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        result = compare(hand1, hand2, board)

        assert result == 1  # Aces win

    def test_compare_tie(self):
        """Test comparing hands that tie."""
        hand1 = Hand("As2h")
        hand2 = Hand("Ac2d")  # Same strength
        board = [Card("Ks"), Card("Qd"), Card("Jc"), Card("Ts"), Card("9h")]

        result = compare(hand1, hand2, board)

        assert result == 0  # Tie - both have straight

    def test_winners_multiple(self):
        """Test finding winners among multiple hands."""
        hands = [
            Hand("AsAh"),  # Pair of aces
            Hand("KsKh"),  # Pair of kings
            Hand("QsQh"),  # Pair of queens
        ]
        board = [Card("7c"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        winners = HandEvaluator.winners(hands, board)

        assert winners == [0]  # Only aces win

    def test_evaluation_result_percentile(self):
        """Test percentile calculation."""
        hand = Hand("AsAh")
        board = [Card("Ac"), Card("Ad"), Card("Kc"), Card("3s"), Card("5h")]

        result = HandEvaluator.evaluate(hand, board)

        # Four of a kind should be very high percentile
        assert result.percentile > 99

    def test_evaluation_result_beats(self):
        """Test beats() method."""
        hand1 = Hand("AsAh")
        hand2 = Hand("KsKh")
        board = [Card("7c"), Card("8d"), Card("Tc"), Card("3s"), Card("5h")]

        result1 = HandEvaluator.evaluate(hand1, board)
        result2 = HandEvaluator.evaluate(hand2, board)

        assert result1.beats(result2)
        assert not result2.beats(result1)
