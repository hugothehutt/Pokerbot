"""Deck management for Texas Hold'em poker."""

from __future__ import annotations
import random
from typing import Iterator

from pokerbot.core.card import Card


class Deck:
    """
    A standard 52-card deck with shuffling and dealing operations.

    The deck maintains a list of available cards and tracks dealt cards
    to prevent dealing duplicates.
    """

    __slots__ = ("_cards", "_dealt", "_rng")

    def __init__(self, seed: int | None = None):
        """
        Create a new shuffled deck.

        Args:
            seed: Optional random seed for reproducible shuffles
        """
        self._rng = random.Random(seed)
        self._cards: list[int] = list(range(52))  # Card indices
        self._dealt: set[int] = set()
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle all cards back into the deck."""
        self._cards = list(range(52))
        self._dealt.clear()
        self._rng.shuffle(self._cards)

    def deal(self, count: int = 1) -> list[Card]:
        """
        Deal cards from the deck.

        Args:
            count: Number of cards to deal

        Returns:
            List of dealt cards

        Raises:
            ValueError: If not enough cards remain
        """
        if count > len(self._cards):
            raise ValueError(f"Cannot deal {count} cards, only {len(self._cards)} remain")

        dealt = []
        for _ in range(count):
            index = self._cards.pop()
            self._dealt.add(index)
            dealt.append(Card.from_index(index))
        return dealt

    def deal_one(self) -> Card:
        """Deal a single card from the deck."""
        return self.deal(1)[0]

    def remove(self, cards: list[Card] | Card) -> None:
        """
        Remove specific cards from the deck (mark as dealt).

        Args:
            cards: Card or list of cards to remove
        """
        if isinstance(cards, Card):
            cards = [cards]

        for card in cards:
            idx = card.index
            if idx in self._dealt:
                continue  # Already dealt
            if idx in self._cards:
                self._cards.remove(idx)
            self._dealt.add(idx)

    def remaining(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self._cards)

    def is_available(self, card: Card) -> bool:
        """Check if a card is still available in the deck."""
        return card.index in self._cards

    def available_cards(self) -> list[Card]:
        """Return list of all available cards."""
        return [Card.from_index(i) for i in self._cards]

    def dealt_cards(self) -> list[Card]:
        """Return list of all dealt cards."""
        return [Card.from_index(i) for i in sorted(self._dealt)]

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        """Iterate over remaining cards without dealing them."""
        for idx in self._cards:
            yield Card.from_index(idx)

    def __repr__(self) -> str:
        return f"Deck({len(self._cards)} cards remaining)"

    def copy(self) -> Deck:
        """Create a copy of this deck in its current state."""
        new_deck = Deck.__new__(Deck)
        new_deck._rng = random.Random()
        new_deck._cards = self._cards.copy()
        new_deck._dealt = self._dealt.copy()
        return new_deck

    @classmethod
    def with_dead_cards(cls, dead_cards: list[Card], seed: int | None = None) -> Deck:
        """
        Create a deck with certain cards already removed.

        Useful for Monte Carlo simulations where some cards are known.

        Args:
            dead_cards: Cards to remove from the deck
            seed: Optional random seed

        Returns:
            A new deck with dead cards removed
        """
        deck = cls(seed=seed)
        deck.remove(dead_cards)
        return deck
