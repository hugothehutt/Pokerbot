"""Hand range representation and parsing for poker strategy analysis."""

from __future__ import annotations
import re
from enum import Enum, auto
from typing import Iterator, Optional
from dataclasses import dataclass, field

from pokerbot.core.card import Card, Rank, Suit
from pokerbot.core.hand import Hand, hand_combinations


class RangeType(Enum):
    """Types of poker ranges based on hand selection strategy."""
    LINEAR = auto()      # Top X% of hands in order (e.g., value-heavy)
    POLARIZED = auto()   # Strong hands + bluffs, skipping medium strength
    CAPPED = auto()      # Range missing the strongest hands
    MERGED = auto()      # Mix of value and medium-strength hands
    CONDENSED = auto()   # Mostly medium-strength hands


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


# Hand strength rankings for percentage-based ranges
# Based on preflop all-in equity vs random hand
# Ordered from strongest to weakest (index 0 = strongest)
HAND_STRENGTH_RANKING = [
    # Tier 1: Premium (top ~2.5%)
    "AA", "KK", "QQ", "JJ", "AKs",
    # Tier 2: Strong (top ~5%)
    "AKo", "AQs", "TT", "AQo", "AJs",
    # Tier 3: Good (top ~10%)
    "KQs", "99", "ATs", "AJo", "KJs", "KQo", "88", "A9s", "KTs", "ATo",
    # Tier 4: Playable (top ~15%)
    "QJs", "A8s", "K9s", "77", "A5s", "QTs", "A7s", "KJo", "A4s", "A6s",
    "A3s", "QJo", "66", "A9o", "KTo", "A2s", "Q9s", "JTs", "K8s", "A8o",
    # Tier 5: Marginal (top ~25%)
    "55", "K7s", "QTo", "A5o", "J9s", "A7o", "K6s", "T9s", "K5s", "A4o",
    "Q8s", "K9o", "A6o", "K4s", "44", "JTo", "A3o", "J8s", "Q9o", "K3s",
    "T8s", "K2s", "Q7s", "A2o", "98s", "K8o", "33", "Q6s", "J9o", "87s",
    # Tier 6: Speculative (top ~40%)
    "K7o", "Q5s", "T9o", "22", "J7s", "Q4s", "97s", "K6o", "Q8o", "T7s",
    "J8o", "Q3s", "76s", "K5o", "86s", "Q2s", "J6s", "98o", "65s", "K4o",
    "T8o", "J5s", "Q7o", "96s", "K3o", "54s", "75s", "J4s", "87o", "K2o",
    # Tier 7: Weak (top ~60%)
    "T6s", "Q6o", "J3s", "85s", "97o", "64s", "J7o", "T7o", "Q5o", "J2s",
    "76o", "53s", "T5s", "86o", "Q4o", "74s", "95s", "T4s", "65o", "Q3o",
    "43s", "96o", "J6o", "T3s", "84s", "54o", "Q2o", "63s", "75o", "T2s",
    # Tier 8: Very Weak (top ~80%)
    "J5o", "94s", "73s", "52s", "85o", "93s", "J4o", "64o", "T6o", "42s",
    "92s", "83s", "53o", "J3o", "74o", "T5o", "82s", "62s", "43o", "95o",
    "J2o", "72s", "84o", "T4o", "63o", "32s", "94o", "T3o", "52o", "73o",
    # Tier 9: Trash (bottom ~20%)
    "42o", "93o", "T2o", "83o", "92o", "62o", "82o", "72o", "32o",
]

# Number of combos for each hand type
HAND_COMBO_COUNT = {}
for notation in HAND_STRENGTH_RANKING:
    if len(notation) == 2:  # Pair
        HAND_COMBO_COUNT[notation] = 6
    elif notation.endswith('s'):  # Suited
        HAND_COMBO_COUNT[notation] = 4
    else:  # Offsuit
        HAND_COMBO_COUNT[notation] = 12

# Cumulative combo count for percentage calculations
CUMULATIVE_COMBOS = []
total = 0
for notation in HAND_STRENGTH_RANKING:
    total += HAND_COMBO_COUNT[notation]
    CUMULATIVE_COMBOS.append(total)
TOTAL_COMBOS = 1326  # 52 choose 2


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


# =============================================================================
# Percentage-based Range Functions
# =============================================================================

def top_percent_range(percent: float) -> HandRange:
    """
    Create a range containing the top X% of hands by strength.

    This creates a LINEAR range - the strongest hands down to a cutoff.

    Args:
        percent: Percentage of hands to include (0-100)

    Returns:
        HandRange containing the top X% of hands

    Examples:
        top_percent_range(10)  -> Top 10% of hands
        top_percent_range(35)  -> Top 35% of hands (typical open range)
    """
    if percent <= 0:
        return HandRange()
    if percent >= 100:
        return HandRange(",".join(HAND_STRENGTH_RANKING))

    target_combos = int(TOTAL_COMBOS * (percent / 100))

    hands_to_include = []
    for i, notation in enumerate(HAND_STRENGTH_RANKING):
        if CUMULATIVE_COMBOS[i] <= target_combos:
            hands_to_include.append(notation)
        else:
            # Partial inclusion with weight for the boundary hand
            if i > 0:
                prev_combos = CUMULATIVE_COMBOS[i - 1]
            else:
                prev_combos = 0
            remaining = target_combos - prev_combos
            if remaining > 0:
                weight = remaining / HAND_COMBO_COUNT[notation]
                hands_to_include.append(f"{notation}:{weight:.2f}")
            break

    return HandRange(",".join(hands_to_include))


def linear_range(percent: float) -> HandRange:
    """
    Create a linear (value-heavy) range.

    Linear ranges contain the top X% of hands in order of strength.
    Used when you want to play straightforward poker with your best hands.

    Alias for top_percent_range().

    Args:
        percent: Percentage of hands (0-100)

    Returns:
        HandRange with linear construction
    """
    return top_percent_range(percent)


def polarized_range(
    value_percent: float,
    bluff_percent: float,
    gap_start_percent: float = 15,
    gap_end_percent: float = 50,
) -> HandRange:
    """
    Create a polarized range with strong hands and bluffs, skipping medium hands.

    Polarized ranges are used when you want to bet/raise with:
    - Very strong hands (for value)
    - Weak hands with potential (as bluffs)
    - While checking medium-strength hands

    Args:
        value_percent: Top X% of hands for value (e.g., 10 = top 10%)
        bluff_percent: Percentage of bluffs to add from the gap
        gap_start_percent: Where the gap starts (end of value hands)
        gap_end_percent: Where the gap ends (start of give-up hands)

    Returns:
        HandRange with polarized construction

    Example:
        polarized_range(10, 5)  -> Top 10% for value + some bluffs from weaker range
    """
    # Value portion (top hands)
    value_range = top_percent_range(value_percent)

    # Bluff portion (selected from the gap between gap_start and gap_end)
    # Bluffs typically come from hands at the bottom of the gap
    # that have some equity (suited connectors, suited aces, etc.)
    bluff_start = max(gap_start_percent, gap_end_percent - bluff_percent)
    bluff_end = gap_end_percent

    bluff_hands = []
    start_combos = int(TOTAL_COMBOS * (bluff_start / 100))
    end_combos = int(TOTAL_COMBOS * (bluff_end / 100))

    for i, notation in enumerate(HAND_STRENGTH_RANKING):
        combo_start = CUMULATIVE_COMBOS[i - 1] if i > 0 else 0
        combo_end = CUMULATIVE_COMBOS[i]

        if combo_start >= start_combos and combo_end <= end_combos:
            bluff_hands.append(notation)
        elif combo_start < end_combos and combo_end > start_combos:
            # Partial overlap
            bluff_hands.append(notation)

    bluff_range = HandRange(",".join(bluff_hands)) if bluff_hands else HandRange()

    return value_range | bluff_range


def capped_range(
    min_percent: float,
    max_percent: float,
) -> HandRange:
    """
    Create a capped range that excludes the strongest hands.

    Capped ranges are missing premium hands, typically because:
    - You would have 3-bet with those hands
    - You're in a spot where you can't have the nuts

    Args:
        min_percent: Start of range (hands weaker than top X%)
        max_percent: End of range (hands stronger than bottom X%)

    Returns:
        HandRange excluding the top hands

    Example:
        capped_range(5, 40)  -> Hands ranked 5%-40% (excludes AA, KK, etc.)
    """
    min_combos = int(TOTAL_COMBOS * (min_percent / 100))
    max_combos = int(TOTAL_COMBOS * (max_percent / 100))

    hands_to_include = []
    for i, notation in enumerate(HAND_STRENGTH_RANKING):
        combo_start = CUMULATIVE_COMBOS[i - 1] if i > 0 else 0
        combo_end = CUMULATIVE_COMBOS[i]

        # Include if within range
        if combo_start >= min_combos and combo_end <= max_combos:
            hands_to_include.append(notation)
        elif combo_start < max_combos and combo_end > min_combos:
            # Partial overlap - include with weight
            overlap_start = max(combo_start, min_combos)
            overlap_end = min(combo_end, max_combos)
            overlap = overlap_end - overlap_start
            weight = overlap / HAND_COMBO_COUNT[notation]
            if weight > 0.1:  # Only include if significant
                hands_to_include.append(f"{notation}:{weight:.2f}")

    return HandRange(",".join(hands_to_include))


def merged_range(percent: float, medium_weight: float = 0.5) -> HandRange:
    """
    Create a merged range with value hands and some medium-strength hands.

    Merged ranges mix strong and medium hands, used when:
    - Stack-to-pot ratio is low
    - You want to protect checking range
    - Board favors condensed ranges

    Args:
        percent: Total percentage of hands
        medium_weight: Weight for medium-strength hands (0-1)

    Returns:
        HandRange with merged construction
    """
    # Top portion at full weight
    value_cutoff = percent * 0.6
    value_range = top_percent_range(value_cutoff)

    # Medium portion at reduced weight
    medium_start = value_cutoff
    medium_end = percent

    medium_hands = []
    start_combos = int(TOTAL_COMBOS * (medium_start / 100))
    end_combos = int(TOTAL_COMBOS * (medium_end / 100))

    for i, notation in enumerate(HAND_STRENGTH_RANKING):
        combo_start = CUMULATIVE_COMBOS[i - 1] if i > 0 else 0
        combo_end = CUMULATIVE_COMBOS[i]

        if combo_start >= start_combos and combo_end <= end_combos:
            medium_hands.append(f"{notation}:{medium_weight:.2f}")

    medium_range = HandRange(",".join(medium_hands)) if medium_hands else HandRange()

    return value_range | medium_range


def condensed_range(start_percent: float, end_percent: float) -> HandRange:
    """
    Create a condensed range of medium-strength hands.

    Condensed ranges contain mostly medium-strength hands, used when:
    - You've capped your range by not raising earlier
    - Board favors medium-strength hands
    - Opponent has polarized range

    Args:
        start_percent: Start of range (e.g., 10 = start at top 10%)
        end_percent: End of range (e.g., 40 = end at top 40%)

    Returns:
        HandRange with condensed construction

    Example:
        condensed_range(10, 40)  -> Hands ranked 10%-40%
    """
    return capped_range(start_percent, end_percent)


def get_hand_percentile(hand: Hand | str) -> float:
    """
    Get the percentile ranking of a hand (0-100, lower = stronger).

    Args:
        hand: Hand object or notation string

    Returns:
        Percentile (0 = AA, 100 = worst hand)

    Example:
        get_hand_percentile("AA")  -> ~0.45 (top 0.45%)
        get_hand_percentile("72o") -> ~99.5 (bottom 0.5%)
    """
    if isinstance(hand, Hand):
        notation = hand.notation()
    else:
        notation = hand.upper().replace("10", "T")

    # Normalize notation format to match HAND_STRENGTH_RANKING
    # The ranking uses uppercase with 's' and 'o' suffixes
    if len(notation) == 3:
        notation = notation[:2].upper() + notation[2].lower()

    try:
        idx = HAND_STRENGTH_RANKING.index(notation)
        return (CUMULATIVE_COMBOS[idx] / TOTAL_COMBOS) * 100
    except ValueError:
        # Try without suited/offsuit marker for pairs
        if len(notation) == 2:
            try:
                idx = HAND_STRENGTH_RANKING.index(notation.upper())
                return (CUMULATIVE_COMBOS[idx] / TOTAL_COMBOS) * 100
            except ValueError:
                pass
        return 50.0  # Default to middle if not found


def get_hands_in_percentile(start: float, end: float) -> list[str]:
    """
    Get list of hand notations within a percentile range.

    Args:
        start: Start percentile (0-100)
        end: End percentile (0-100)

    Returns:
        List of hand notations in that range
    """
    start_combos = int(TOTAL_COMBOS * (start / 100))
    end_combos = int(TOTAL_COMBOS * (end / 100))

    hands = []
    for i, notation in enumerate(HAND_STRENGTH_RANKING):
        combo_start = CUMULATIVE_COMBOS[i - 1] if i > 0 else 0
        combo_end = CUMULATIVE_COMBOS[i]

        if combo_end > start_combos and combo_start < end_combos:
            hands.append(notation)

    return hands


def describe_range(hand_range: HandRange) -> dict:
    """
    Analyze and describe a range's characteristics.

    Returns:
        Dict with range analysis including type classification
    """
    if hand_range.num_combinations == 0:
        return {
            "type": RangeType.LINEAR,
            "percentage": 0,
            "description": "Empty range",
            "is_capped": False,
            "is_polarized": False,
        }

    # Get percentiles of hands in range
    percentiles = []
    for notation in hand_range._hands.keys():
        percentiles.append(get_hand_percentile(notation))

    percentiles.sort()
    min_pct = min(percentiles)
    max_pct = max(percentiles)
    pct = hand_range.percentage

    # Check for gaps (polarization)
    gaps = []
    for i in range(1, len(percentiles)):
        gap = percentiles[i] - percentiles[i-1]
        if gap > 5:  # Significant gap
            gaps.append(gap)

    # Classify range type
    is_capped = min_pct > 3  # Missing top hands
    has_large_gap = any(g > 15 for g in gaps)

    if is_capped and not has_large_gap:
        range_type = RangeType.CAPPED
        description = f"Capped range (missing top {min_pct:.1f}%)"
    elif has_large_gap:
        range_type = RangeType.POLARIZED
        description = f"Polarized range with gaps"
    elif max_pct - min_pct < pct * 1.5:
        range_type = RangeType.CONDENSED
        description = f"Condensed range ({min_pct:.1f}%-{max_pct:.1f}%)"
    else:
        range_type = RangeType.LINEAR
        description = f"Linear range (top {pct:.1f}%)"

    return {
        "type": range_type,
        "percentage": pct,
        "percentile_range": (min_pct, max_pct),
        "description": description,
        "is_capped": is_capped,
        "is_polarized": has_large_gap,
        "num_hands": hand_range.num_notations,
        "num_combos": hand_range.num_combinations,
    }
