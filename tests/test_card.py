"""Tests for card, deck, and hand modules."""

import pytest

from pokerbot.core.card import Card, Rank, Suit, all_cards
from pokerbot.core.deck import Deck
from pokerbot.core.hand import Hand, hand_combinations


class TestCard:
    """Tests for Card class."""

    def test_create_card_from_string(self):
        """Test creating cards from string notation."""
        card = Card("As")
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

        card = Card("2c")
        assert card.rank == Rank.TWO
        assert card.suit == Suit.CLUBS

        card = Card("Th")
        assert card.rank == Rank.TEN
        assert card.suit == Suit.HEARTS

    def test_create_card_from_rank_suit(self):
        """Test creating cards from rank and suit."""
        card = Card(Rank.KING, Suit.DIAMONDS)
        assert card.rank == Rank.KING
        assert card.suit == Suit.DIAMONDS

    def test_card_equality(self):
        """Test card equality comparison."""
        card1 = Card("As")
        card2 = Card("As")
        card3 = Card("Ah")

        assert card1 == card2
        assert card1 != card3

    def test_card_hash(self):
        """Test card hashing for use in sets/dicts."""
        card1 = Card("As")
        card2 = Card("As")

        assert hash(card1) == hash(card2)
        assert len({card1, card2}) == 1

    def test_card_comparison(self):
        """Test card ordering."""
        ace = Card("As")
        king = Card("Ks")
        two = Card("2s")

        assert ace > king > two
        assert two < king < ace

    def test_card_string_representation(self):
        """Test card string output."""
        card = Card("As")
        assert str(card) == "As"

        card = Card("Th")
        assert str(card) == "Th"

    def test_card_index(self):
        """Test card index (0-51)."""
        # First card (2c) should be index 0
        card = Card("2c")
        assert card.index == 0

        # Ace of spades should be index 51
        card = Card("As")
        assert card.index == 51

    def test_card_from_index(self):
        """Test creating card from index."""
        card = Card.from_index(0)
        assert card.rank == Rank.TWO
        assert card.suit == Suit.CLUBS

        card = Card.from_index(51)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_all_cards(self):
        """Test iterating all 52 cards."""
        cards = list(all_cards())
        assert len(cards) == 52
        assert len(set(cards)) == 52  # All unique


class TestDeck:
    """Tests for Deck class."""

    def test_deck_creation(self):
        """Test deck has 52 cards."""
        deck = Deck()
        assert len(deck) == 52

    def test_deck_deal(self):
        """Test dealing cards from deck."""
        deck = Deck()
        cards = deck.deal(5)

        assert len(cards) == 5
        assert len(deck) == 47

    def test_deck_deal_one(self):
        """Test dealing single card."""
        deck = Deck()
        card = deck.deal_one()

        assert isinstance(card, Card)
        assert len(deck) == 51

    def test_deck_remove(self):
        """Test removing specific cards."""
        deck = Deck()
        card = Card("As")
        deck.remove(card)

        assert not deck.is_available(card)
        assert len(deck) == 51

    def test_deck_shuffle(self):
        """Test deck shuffling."""
        deck1 = Deck(seed=42)
        deck2 = Deck(seed=42)

        # Same seed should produce same order
        cards1 = deck1.deal(5)
        cards2 = deck2.deal(5)
        assert cards1 == cards2

    def test_deck_copy(self):
        """Test deck copying."""
        deck = Deck()
        deck.deal(5)

        copy = deck.copy()
        assert len(copy) == len(deck)

    def test_deck_with_dead_cards(self):
        """Test creating deck with dead cards."""
        dead = [Card("As"), Card("Ks")]
        deck = Deck.with_dead_cards(dead)

        assert len(deck) == 50
        assert not deck.is_available(Card("As"))


class TestHand:
    """Tests for Hand class."""

    def test_hand_creation_from_string(self):
        """Test creating hand from string."""
        hand = Hand("AsKs")
        assert hand.high_card.rank == Rank.ACE
        assert hand.low_card.rank == Rank.KING

    def test_hand_creation_from_cards(self):
        """Test creating hand from card objects."""
        cards = [Card("As"), Card("Ks")]
        hand = Hand(cards)

        assert hand.high_card == Card("As")
        assert hand.low_card == Card("Ks")

    def test_hand_pocket_pair(self):
        """Test pocket pair detection."""
        pair = Hand("AsAh")
        non_pair = Hand("AsKs")

        assert pair.is_pocket_pair
        assert not non_pair.is_pocket_pair

    def test_hand_suited(self):
        """Test suited detection."""
        suited = Hand("AsKs")
        offsuit = Hand("AsKh")

        assert suited.is_suited
        assert not offsuit.is_suited

    def test_hand_connected(self):
        """Test connected cards detection."""
        connected = Hand("9s8s")
        gapper = Hand("9s7s")
        big_gap = Hand("As2s")

        assert connected.is_connected
        assert connected.gap == 0
        assert not gapper.is_connected
        assert gapper.gap == 1
        assert big_gap.gap == 11

    def test_hand_notation(self):
        """Test hand notation output."""
        pair = Hand("AsAh")
        suited = Hand("AsKs")
        offsuit = Hand("AsKh")

        assert pair.notation() == "AA"
        assert suited.notation() == "AKs"
        assert offsuit.notation() == "AKo"

    def test_hand_combinations(self):
        """Test getting specific hand combinations."""
        # 6 combinations of AA
        aa_combos = hand_combinations("AA")
        assert len(aa_combos) == 6

        # 4 combinations of AKs
        aks_combos = hand_combinations("AKs")
        assert len(aks_combos) == 4

        # 12 combinations of AKo
        ako_combos = hand_combinations("AKo")
        assert len(ako_combos) == 12

    def test_hand_equality(self):
        """Test hand equality."""
        hand1 = Hand("AsKs")
        hand2 = Hand("KsAs")  # Same cards, different order

        assert hand1 == hand2
