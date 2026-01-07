"""Hand representation for Texas Hold'em poker."""

from __future__ import annotations
from typing import Iterator

from pokerbot.core.card import Card, Rank


class Hand:
    """
    Represents a poker hand (hole cards).

    In Texas Hold'em, each player has exactly 2 hole cards.
    This class provides utilities for working with hole card combinations.
    """

    __slots__ = ("_cards",)

    def __init__(self, cards: list[Card] | tuple[Card, Card] | str):
        """
        Create a hand from cards.

        Args:
            cards: Two cards as list, tuple, or string like "AsKh"
        """
        if isinstance(cards, str):
            cards = self._parse_hand_string(cards)

        if len(cards) != 2:
            raise ValueError(f"Hand must have exactly 2 cards, got {len(cards)}")

        # Store cards sorted by rank (high card first)
        card_list = list(cards)
        if card_list[0].rank < card_list[1].rank:
            card_list = [card_list[1], card_list[0]]

        self._cards: tuple[Card, Card] = (card_list[0], card_list[1])

    @staticmethod
    def _parse_hand_string(s: str) -> list[Card]:
        """Parse hand string like 'AsKh' or 'As Kh'."""
        s = s.replace(" ", "")
        if len(s) == 4:
            return [Card(s[:2]), Card(s[2:])]
        elif len(s) == 5:  # Handle 10
            if s.startswith("10"):
                return [Card(s[:3]), Card(s[3:])]
            elif "10" in s:
                idx = s.index("10")
                if idx == 0:
                    return [Card(s[:3]), Card(s[3:])]
                else:
                    return [Card(s[:2]), Card(s[2:])]
        raise ValueError(f"Cannot parse hand string: {s}")

    @property
    def cards(self) -> tuple[Card, Card]:
        """Get the two hole cards."""
        return self._cards

    @property
    def high_card(self) -> Card:
        """Get the higher ranked card."""
        return self._cards[0]

    @property
    def low_card(self) -> Card:
        """Get the lower ranked card."""
        return self._cards[1]

    @property
    def is_pocket_pair(self) -> bool:
        """Check if this is a pocket pair."""
        return self._cards[0].rank == self._cards[1].rank

    @property
    def is_suited(self) -> bool:
        """Check if both cards share the same suit."""
        return self._cards[0].suit == self._cards[1].suit

    @property
    def gap(self) -> int:
        """
        Return the gap between card ranks.

        0 = connected (e.g., 87)
        1 = one-gapper (e.g., 97)
        etc.
        """
        return abs(self._cards[0].rank - self._cards[1].rank) - 1

    @property
    def is_connected(self) -> bool:
        """Check if cards are connected (no gap)."""
        return self.gap == 0

    @property
    def is_broadway(self) -> bool:
        """Check if both cards are broadway (T+)."""
        return self._cards[0].rank >= Rank.TEN and self._cards[1].rank >= Rank.TEN

    def notation(self) -> str:
        """
        Return standard hand notation like 'AKs', 'QQ', '72o'.

        Format:
        - Pairs: 'AA', 'KK', etc.
        - Suited: 'AKs', 'JTs', etc.
        - Offsuit: 'AKo', 'Q9o', etc.
        """
        r1 = self._cards[0].rank.symbol
        r2 = self._cards[1].rank.symbol
        if r1 == "10":
            r1 = "T"
        if r2 == "10":
            r2 = "T"

        if self.is_pocket_pair:
            return f"{r1}{r2}"
        elif self.is_suited:
            return f"{r1}{r2}s"
        else:
            return f"{r1}{r2}o"

    def to_treys(self) -> list[int]:
        """Convert to treys library format."""
        return [card.to_treys() for card in self._cards]

    def __repr__(self) -> str:
        return f"Hand('{self}')"

    def __str__(self) -> str:
        """Return string like 'As Kh'."""
        return f"{self._cards[0]} {self._cards[1]}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            return NotImplemented
        return set(self._cards) == set(other._cards)

    def __hash__(self) -> int:
        return hash(frozenset(c.index for c in self._cards))

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    def __contains__(self, card: Card) -> bool:
        return card in self._cards

    @property
    def pretty(self) -> str:
        """Return pretty-printed hand with Unicode suits."""
        return f"{self._cards[0].pretty} {self._cards[1].pretty}"


def all_hands() -> Iterator[Hand]:
    """
    Yield all 1326 possible Hold'em starting hands.

    This is useful for building preflop charts and range analysis.
    """
    from pokerbot.core.card import all_cards

    cards = list(all_cards())
    for i, card1 in enumerate(cards):
        for card2 in cards[i + 1:]:
            yield Hand([card1, card2])


def hand_combinations(hand_notation: str) -> list[Hand]:
    """
    Get all specific hand combinations for a notation.

    Args:
        hand_notation: Notation like 'AKs', 'QQ', '72o'

    Returns:
        List of all specific hands matching the notation

    Examples:
        hand_combinations('AA') -> 6 hands (AsAh, AsAd, AsAc, AhAd, AhAc, AdAc)
        hand_combinations('AKs') -> 4 hands (AsKs, AhKh, AdKd, AcKc)
        hand_combinations('AKo') -> 12 hands (all unsuited AK combinations)
    """
    from pokerbot.core.card import Rank, Suit, Card

    notation = hand_notation.upper().replace("10", "T")

    # Parse the notation
    if len(notation) == 2:
        # Pocket pair like "AA"
        rank = Rank.from_char(notation[0])
        suits = list(Suit)
        hands = []
        for i, s1 in enumerate(suits):
            for s2 in suits[i + 1:]:
                hands.append(Hand([Card(rank, s1), Card(rank, s2)]))
        return hands

    elif len(notation) == 3:
        r1 = Rank.from_char(notation[0])
        r2 = Rank.from_char(notation[1])
        suited = notation[2].lower()

        suits = list(Suit)
        hands = []

        if suited == "s":
            # Suited hands
            for s in suits:
                hands.append(Hand([Card(r1, s), Card(r2, s)]))
        elif suited == "o":
            # Offsuit hands
            for s1 in suits:
                for s2 in suits:
                    if s1 != s2:
                        hands.append(Hand([Card(r1, s1), Card(r2, s2)]))
        else:
            raise ValueError(f"Invalid notation: {hand_notation}")

        return hands

    else:
        raise ValueError(f"Invalid hand notation: {hand_notation}")
