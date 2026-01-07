"""Card representation for Texas Hold'em poker."""

from __future__ import annotations
from enum import IntEnum
from functools import total_ordering
from typing import Iterator


class Suit(IntEnum):
    """Card suits with integer values for efficient comparison."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        return self.symbol

    @property
    def symbol(self) -> str:
        """Return the Unicode symbol for this suit."""
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠",
        }
        return symbols[self]

    @property
    def letter(self) -> str:
        """Return single-letter representation."""
        return "cdhs"[self.value]

    @classmethod
    def from_char(cls, char: str) -> Suit:
        """Create Suit from character representation."""
        char = char.lower()
        mapping = {"c": cls.CLUBS, "d": cls.DIAMONDS, "h": cls.HEARTS, "s": cls.SPADES}
        if char not in mapping:
            raise ValueError(f"Invalid suit character: {char}")
        return mapping[char]


class Rank(IntEnum):
    """Card ranks from 2 to Ace (14)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        return self.symbol

    @property
    def symbol(self) -> str:
        """Return the symbol for this rank."""
        if self.value <= 10:
            return str(self.value)
        return {11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]

    @classmethod
    def from_char(cls, char: str) -> Rank:
        """Create Rank from character representation."""
        char = char.upper()
        mapping = {
            "2": cls.TWO, "3": cls.THREE, "4": cls.FOUR, "5": cls.FIVE,
            "6": cls.SIX, "7": cls.SEVEN, "8": cls.EIGHT, "9": cls.NINE,
            "T": cls.TEN, "10": cls.TEN, "J": cls.JACK, "Q": cls.QUEEN,
            "K": cls.KING, "A": cls.ACE,
        }
        if char not in mapping:
            raise ValueError(f"Invalid rank character: {char}")
        return mapping[char]


@total_ordering
class Card:
    """
    Represents a single playing card.

    Cards are immutable and can be efficiently compared and hashed.
    Internal representation uses an integer (0-51) for fast operations.
    """

    __slots__ = ("_index",)

    def __init__(self, rank: Rank | int | str, suit: Suit | int | str | None = None):
        """
        Create a card.

        Args:
            rank: Rank as Rank enum, int (2-14), or string ("A", "K", etc.)
                  Can also be a full card string like "As" or "Kh"
            suit: Suit as Suit enum, int (0-3), or string ("s", "h", etc.)
                  Not needed if rank is a full card string
        """
        if isinstance(rank, str) and suit is None:
            # Parse full card string like "As", "Kh", "2c"
            if len(rank) < 2:
                raise ValueError(f"Invalid card string: {rank}")
            rank_str = rank[:-1]
            suit_str = rank[-1]
            rank = Rank.from_char(rank_str)
            suit = Suit.from_char(suit_str)
        elif isinstance(rank, str):
            rank = Rank.from_char(rank)
        elif isinstance(rank, int) and not isinstance(rank, Rank):
            rank = Rank(rank)

        if isinstance(suit, str):
            suit = Suit.from_char(suit)
        elif isinstance(suit, int) and not isinstance(suit, Suit):
            suit = Suit(suit)

        if suit is None:
            raise ValueError("Suit must be provided")

        # Store as single integer: rank * 4 + suit
        self._index: int = (rank.value - 2) * 4 + suit.value

    @classmethod
    def from_index(cls, index: int) -> Card:
        """Create a card from its integer index (0-51)."""
        if not 0 <= index <= 51:
            raise ValueError(f"Card index must be 0-51, got {index}")
        card = object.__new__(cls)
        card._index = index
        return card

    @property
    def rank(self) -> Rank:
        """Get the card's rank."""
        return Rank(self._index // 4 + 2)

    @property
    def suit(self) -> Suit:
        """Get the card's suit."""
        return Suit(self._index % 4)

    @property
    def index(self) -> int:
        """Get the card's integer index (0-51)."""
        return self._index

    def __repr__(self) -> str:
        return f"Card('{self}')"

    def __str__(self) -> str:
        """Return string like 'As', 'Kh', '2c'."""
        rank_char = self.rank.symbol
        if rank_char == "10":
            rank_char = "T"
        return f"{rank_char}{self.suit.letter}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._index == other._index

    def __lt__(self, other: Card) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        # Compare by rank first, then by suit
        return (self.rank, self.suit) < (other.rank, other.suit)

    def __hash__(self) -> int:
        return hash(self._index)

    @property
    def pretty(self) -> str:
        """Return a pretty-printed card with Unicode suit symbol."""
        rank_char = self.rank.symbol
        return f"{rank_char}{self.suit.symbol}"

    def to_treys(self) -> int:
        """Convert to treys library format for hand evaluation."""
        from treys import Card as TreysCard
        return TreysCard.new(str(self))


def all_cards() -> Iterator[Card]:
    """Yield all 52 cards in a standard deck."""
    for i in range(52):
        yield Card.from_index(i)
