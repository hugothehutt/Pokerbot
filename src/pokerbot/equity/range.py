"""Hand range representation and parsing for poker strategy analysis."""

from __future__ import annotations
import re
from typing import Iterator, Optional
from dataclasses import dataclass, field

from pokerbot.core.card import Card, Rank, Suit
from pokerbot.core.hand import Hand, hand_combinations


# All 169 distinct starting hand types
ALL_HAND_NOTATIONS = []
for r1 in range(14, 1, -1):  # A to 2
    for r2 in range(r1, 1, -1):  # Same rank down to 2
        rank1 = Rank(r1).symbol.replace("10", "T")
        rank2 = Rank(r2).symbol.replace("10", "T")
        if r1 == r2:
            ALL_HAND_NOTATIONS.append(f"{rank1}{rank2}")  # Pairs
        else:
            ALL_HAND_NOTATIONS.append(f"{rank1}{rank2}s")  # Suited
            ALL_HAND_NOTATIONS.append(f"{rank1}{rank2}o")  # Offsuit


@dataclass
class RangeWeight:
    """A hand notation with an associated weight (frequency)."""
    notation: str
    weight: float = 1.0

    def __str__(self) -> str:
        if self.weight == 1.0:
            return self.notation
        return f"{self.notation}:{self.weight:.2f}"


class HandRange:
    """
    Represents a range of poker hands.

    Supports parsing of standard range notation:
    - Specific hands: "AKs", "QQ", "72o"
    - Ranges: "AA-TT", "AKs-ATs"
    - Combined: "AA,KK,QQ,AKs,AQs"
    - Weighted: "AKs:0.5,QQ:1.0"
    - Plus notation: "TT+", "ATs+"

    Examples:
        range = HandRange("AA,KK,QQ,AKs")
        range = HandRange("TT+,AQs+,AKo")
        range = HandRange("AA:0.5,KK:1.0")  # 50% of AA, 100% of KK
    """

    def __init__(self, notation: str = "", hands: Optional[list[Hand]] = None):
        """
        Create a hand range from notation or list of hands.

        Args:
            notation: Range notation string
            hands: Optional list of specific hands
        """
        self._hands: dict[str, RangeWeight] = {}
        self._combo_cache: Optional[list[Hand]] = None

        if hands:
            for hand in hands:
                self.add_hand(hand)
        elif notation:
            self._parse_notation(notation)

    def _parse_notation(self, notation: str) -> None:
        """Parse range notation string."""
        # Remove whitespace and split by comma
        notation = notation.replace(" ", "")
        parts = notation.split(",")

        for part in parts:
            if not part:
                continue

            # Check for weight
            weight = 1.0
            if ":" in part:
                part, weight_str = part.split(":")
                weight = float(weight_str)

            # Parse the hand notation
            self._parse_hand_part(part, weight)

    def _parse_hand_part(self, part: str, weight: float) -> None:
        """Parse a single part of range notation."""
        part = part.upper().replace("10", "T")

        # Plus notation: TT+, ATs+
        if part.endswith("+"):
            self._parse_plus_notation(part[:-1], weight)
            return

        # Range notation: AA-TT, AKs-ATs
        if "-" in part:
            self._parse_range_notation(part, weight)
            return

        # Single hand
        self._add_notation(part, weight)

    def _parse_plus_notation(self, base: str, weight: float) -> None:
        """Parse plus notation like TT+ or ATs+."""
        if len(base) == 2 and base[0] == base[1]:
            # Pair plus: TT+ means TT, JJ, QQ, KK, AA
            base_rank = Rank.from_char(base[0])
            for rank_val in range(base_rank.value, 15):  # Up to Ace
                rank = Rank(rank_val)
                symbol = rank.symbol.replace("10", "T")
                self._add_notation(f"{symbol}{symbol}", weight)
        elif len(base) == 3:
            # Suited/offsuit plus: ATs+ means ATs, AJs, AQs, AKs
            r1 = Rank.from_char(base[0])
            r2 = Rank.from_char(base[1])
            suited = base[2].lower()

            for rank_val in range(r2.value, r1.value):
                rank = Rank(rank_val)
                s1 = r1.symbol.replace("10", "T")
                s2 = rank.symbol.replace("10", "T")
                self._add_notation(f"{s1}{s2}{suited}", weight)

    def _parse_range_notation(self, notation: str, weight: float) -> None:
        """Parse range notation like AA-TT or AKs-ATs."""
        parts = notation.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range notation: {notation}")

        start, end = parts

        if len(start) == 2 and start[0] == start[1]:
            # Pair range: AA-TT
            start_rank = Rank.from_char(start[0])
            end_rank = Rank.from_char(end[0])

            low = min(start_rank.value, end_rank.value)
            high = max(start_rank.value, end_rank.value)

            for rank_val in range(low, high + 1):
                rank = Rank(rank_val)
                symbol = rank.symbol.replace("10", "T")
                self._add_notation(f"{symbol}{symbol}", weight)
        else:
            # Suited/offsuit range: AKs-ATs
            r1 = Rank.from_char(start[0])
            r2_start = Rank.from_char(start[1])
            r2_end = Rank.from_char(end[1])
            suited = start[2].lower() if len(start) > 2 else ""

            low = min(r2_start.value, r2_end.value)
            high = max(r2_start.value, r2_end.value)

            for rank_val in range(low, high + 1):
                rank = Rank(rank_val)
                s1 = r1.symbol.replace("10", "T")
                s2 = rank.symbol.replace("10", "T")
                self._add_notation(f"{s1}{s2}{suited}", weight)

    def _add_notation(self, notation: str, weight: float) -> None:
        """Add a hand notation with weight."""
        notation = notation.upper().replace("10", "T")
        self._hands[notation] = RangeWeight(notation, weight)
        self._combo_cache = None

    def add_hand(self, hand: Hand, weight: float = 1.0) -> None:
        """Add a specific hand to the range."""
        notation = hand.notation()
        self._add_notation(notation, weight)

    def remove_hand(self, notation: str) -> None:
        """Remove a hand notation from the range."""
        notation = notation.upper().replace("10", "T")
        if notation in self._hands:
            del self._hands[notation]
            self._combo_cache = None

    def set_weight(self, notation: str, weight: float) -> None:
        """Set the weight for a hand notation."""
        notation = notation.upper().replace("10", "T")
        if notation in self._hands:
            self._hands[notation].weight = weight

    def get_weight(self, notation: str) -> float:
        """Get the weight for a hand notation."""
        notation = notation.upper().replace("10", "T")
        if notation in self._hands:
            return self._hands[notation].weight
        return 0.0

    def contains_notation(self, notation: str) -> bool:
        """Check if a hand notation is in the range."""
        notation = notation.upper().replace("10", "T")
        return notation in self._hands

    def contains_hand(self, hand: Hand) -> bool:
        """Check if a specific hand is in the range."""
        return self.contains_notation(hand.notation())

    def get_combinations(self, dead_cards: Optional[list[Card]] = None) -> list[Hand]:
        """
        Get all specific hand combinations in the range.

        Args:
            dead_cards: Cards to exclude (already dealt)

        Returns:
            List of all hands in the range
        """
        if self._combo_cache is None or dead_cards:
            dead_set = set(c.index for c in (dead_cards or []))
            combos = []

            for rw in self._hands.values():
                for hand in hand_combinations(rw.notation):
                    # Check if any card is dead
                    if not any(c.index in dead_set for c in hand.cards):
                        combos.append(hand)

            if not dead_cards:
                self._combo_cache = combos
            return combos

        return self._combo_cache

    def get_weighted_combinations(
        self,
        dead_cards: Optional[list[Card]] = None
    ) -> list[tuple[Hand, float]]:
        """
        Get all combinations with their weights.

        Returns:
            List of (hand, weight) tuples
        """
        dead_set = set(c.index for c in (dead_cards or []))
        result = []

        for rw in self._hands.values():
            for hand in hand_combinations(rw.notation):
                if not any(c.index in dead_set for c in hand.cards):
                    result.append((hand, rw.weight))

        return result

    @property
    def num_notations(self) -> int:
        """Number of distinct hand notations (max 169)."""
        return len(self._hands)

    @property
    def num_combinations(self) -> int:
        """Number of specific hand combinations (max 1326)."""
        return len(self.get_combinations())

    @property
    def percentage(self) -> float:
        """Percentage of all hands this range represents."""
        return self.num_combinations / 1326 * 100

    def __contains__(self, item) -> bool:
        if isinstance(item, Hand):
            return self.contains_hand(item)
        elif isinstance(item, str):
            return self.contains_notation(item)
        return False

    def __len__(self) -> int:
        return self.num_combinations

    def __iter__(self) -> Iterator[Hand]:
        return iter(self.get_combinations())

    def __str__(self) -> str:
        """Return the range notation string."""
        parts = []
        for rw in sorted(self._hands.values(), key=lambda x: x.notation):
            parts.append(str(rw))
        return ",".join(parts)

    def __repr__(self) -> str:
        return f"HandRange('{self}')"

    def __or__(self, other: HandRange) -> HandRange:
        """Union of two ranges."""
        new_range = HandRange()
        new_range._hands = self._hands.copy()
        for notation, rw in other._hands.items():
            if notation not in new_range._hands:
                new_range._hands[notation] = rw
            else:
                # Take maximum weight
                new_range._hands[notation].weight = max(
                    new_range._hands[notation].weight,
                    rw.weight
                )
        return new_range

    def __and__(self, other: HandRange) -> HandRange:
        """Intersection of two ranges."""
        new_range = HandRange()
        for notation in self._hands:
            if notation in other._hands:
                # Take minimum weight
                weight = min(
                    self._hands[notation].weight,
                    other._hands[notation].weight
                )
                new_range._hands[notation] = RangeWeight(notation, weight)
        return new_range


# Common preflop ranges as constants
PREMIUM_PAIRS = HandRange("AA,KK,QQ")
STRONG_PAIRS = HandRange("AA,KK,QQ,JJ,TT")
MEDIUM_PAIRS = HandRange("99,88,77")
SMALL_PAIRS = HandRange("66,55,44,33,22")

PREMIUM_BROADWAY = HandRange("AKs,AKo,AQs")
STRONG_BROADWAY = HandRange("AKs,AKo,AQs,AQo,AJs,KQs")

SUITED_CONNECTORS = HandRange("JTs,T9s,98s,87s,76s,65s,54s")
SUITED_ACES = HandRange("AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s")


def parse_range(notation: str) -> HandRange:
    """Parse a range notation string. Convenience function."""
    return HandRange(notation)
