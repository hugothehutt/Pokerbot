"""GTO preflop ranges by position for 6-max Texas Hold'em."""

from __future__ import annotations
from typing import Optional

from pokerbot.equity.range import HandRange
from pokerbot.game.player import Position


class PreflopRanges:
    """
    GTO-approximated preflop ranges for 6-max No-Limit Hold'em.

    These ranges are based on solver outputs and represent
    balanced opening, calling, and 3-betting strategies.
    """

    # Opening ranges (RFI - Raise First In) by position
    # Format: position -> range notation
    OPENING_RANGES = {
        Position.UTG: (
            "AA,KK,QQ,JJ,TT,99,88,77,"
            "AKs,AQs,AJs,ATs,A5s,A4s,"
            "AKo,AQo,AJo,"
            "KQs,KJs,KTs,"
            "KQo,"
            "QJs,QTs,"
            "JTs,"
            "T9s,"
            "98s,"
            "87s"
        ),
        Position.HJ: (
            "AA,KK,QQ,JJ,TT,99,88,77,66,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,"
            "KQs,KJs,KTs,K9s,"
            "KQo,KJo,"
            "QJs,QTs,Q9s,"
            "QJo,"
            "JTs,J9s,"
            "T9s,T8s,"
            "98s,"
            "87s,"
            "76s,"
            "65s"
        ),
        Position.CO: (
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,"
            "KQo,KJo,KTo,"
            "QJs,QTs,Q9s,Q8s,"
            "QJo,QTo,"
            "JTs,J9s,J8s,"
            "JTo,"
            "T9s,T8s,"
            "T9o,"
            "98s,97s,"
            "87s,86s,"
            "76s,75s,"
            "65s,64s,"
            "54s"
        ),
        Position.BTN: (
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,"
            "KQo,KJo,KTo,K9o,K8o,"
            "QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,Q4s,"
            "QJo,QTo,Q9o,"
            "JTs,J9s,J8s,J7s,J6s,"
            "JTo,J9o,"
            "T9s,T8s,T7s,T6s,"
            "T9o,T8o,"
            "98s,97s,96s,"
            "98o,"
            "87s,86s,85s,"
            "87o,"
            "76s,75s,74s,"
            "76o,"
            "65s,64s,63s,"
            "54s,53s,"
            "43s"
        ),
        Position.SB: (
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,"
            "KQo,KJo,KTo,K9o,K8o,K7o,"
            "QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,Q4s,Q3s,Q2s,"
            "QJo,QTo,Q9o,Q8o,"
            "JTs,J9s,J8s,J7s,J6s,J5s,J4s,"
            "JTo,J9o,J8o,"
            "T9s,T8s,T7s,T6s,T5s,"
            "T9o,T8o,"
            "98s,97s,96s,95s,"
            "98o,97o,"
            "87s,86s,85s,84s,"
            "87o,"
            "76s,75s,74s,"
            "76o,"
            "65s,64s,63s,"
            "54s,53s,52s,"
            "43s,42s,"
            "32s"
        ),
    }

    # 3-bet ranges vs different positions
    THREE_BET_RANGES = {
        # vs UTG open
        Position.UTG: {
            Position.HJ: "AA,KK,QQ,AKs,AKo",
            Position.CO: "AA,KK,QQ,JJ,AKs,AQs,AKo",
            Position.BTN: "AA,KK,QQ,JJ,TT,AKs,AQs,AJs,AKo,AQo",
            Position.SB: "AA,KK,QQ,JJ,AKs,AQs,AKo",
            Position.BB: "AA,KK,QQ,JJ,TT,AKs,AQs,AJs,A5s,AKo",
        },
        # vs HJ open
        Position.HJ: {
            Position.CO: "AA,KK,QQ,JJ,TT,AKs,AQs,AJs,AKo,AQo",
            Position.BTN: "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,ATs,A5s,AKo,AQo",
            Position.SB: "AA,KK,QQ,JJ,TT,AKs,AQs,AJs,AKo,AQo",
            Position.BB: "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,ATs,A5s,A4s,AKo,AQo",
        },
        # vs CO open
        Position.CO: {
            Position.BTN: "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,A5s,A4s,KQs,AKo,AQo,AJo",
            Position.SB: "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,ATs,A5s,KQs,AKo,AQo",
            Position.BB: "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,A9s,A5s,A4s,KQs,KJs,AKo,AQo,AJo",
        },
        # vs BTN open
        Position.BTN: {
            Position.SB: "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,A9s,A5s,A4s,A3s,KQs,KJs,KTs,QJs,AKo,AQo,AJo,ATo,KQo",
            Position.BB: "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,ATs,A9s,A8s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,AKo,AQo,AJo,ATo,A9o,KQo,KJo",
        },
        # vs SB open
        Position.SB: {
            Position.BB: "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s,65s,AKo,AQo,AJo,ATo,A9o,A8o,KQo,KJo,KTo,QJo,QTo,JTo",
        },
    }

    # Calling ranges (vs open)
    CALLING_RANGES = {
        # Position calling: {opener_position: range}
        Position.HJ: {
            Position.UTG: "TT,99,88,77,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s",
        },
        Position.CO: {
            Position.UTG: "TT,99,88,77,66,AQs,AJs,ATs,A9s,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s",
            Position.HJ: "TT,99,88,77,66,55,AJs,ATs,A9s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s",
        },
        Position.BTN: {
            Position.UTG: "99,88,77,66,55,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,65s,54s",
            Position.HJ: "99,88,77,66,55,44,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,KQs,KJs,KTs,K9s,K8s,QJs,QTs,Q9s,Q8s,JTs,J9s,J8s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s",
            Position.CO: "88,77,66,55,44,33,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KJs,KTs,K9s,K8s,K7s,QTs,Q9s,Q8s,JTs,J9s,J8s,T9s,T8s,T7s,98s,97s,96s,87s,86s,85s,76s,75s,65s,64s,54s,53s,43s",
        },
        Position.SB: {},  # SB usually 3-bets or folds
        Position.BB: {
            Position.UTG: "TT,99,88,77,66,55,44,33,22,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,AJo,ATo,A9o,KQs,KJs,KTs,K9s,K8s,K7s,KQo,KJo,KTo,QJs,QTs,Q9s,Q8s,QJo,QTo,JTs,J9s,J8s,JTo,T9s,T8s,T9o,98s,97s,87s,86s,76s,75s,65s,64s,54s,53s,43s",
            Position.HJ: "99,88,77,66,55,44,33,22,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,ATo,A9o,A8o,KJs,KTs,K9s,K8s,K7s,K6s,KJo,KTo,K9o,QJs,QTs,Q9s,Q8s,Q7s,QJo,QTo,Q9o,JTs,J9s,J8s,J7s,JTo,J9o,T9s,T8s,T7s,T9o,98s,97s,96s,98o,87s,86s,85s,76s,75s,74s,65s,64s,63s,54s,53s,52s,43s,42s,32s",
            Position.CO: "88,77,66,55,44,33,22,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,A9o,A8o,A7o,A6o,A5o,KTs,K9s,K8s,K7s,K6s,K5s,K4s,KTo,K9o,K8o,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,QTo,Q9o,Q8o,JTs,J9s,J8s,J7s,J6s,JTo,J9o,T9s,T8s,T7s,T6s,T9o,T8o,98s,97s,96s,95s,98o,97o,87s,86s,85s,84s,87o,76s,75s,74s,73s,65s,64s,63s,62s,54s,53s,52s,43s,42s,32s",
            Position.BTN: "77,66,55,44,33,22,A8s,A7s,A6s,A5s,A4s,A3s,A2s,A8o,A7o,A6o,A5o,A4o,A3o,A2o,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,K9o,K8o,K7o,K6o,Q9s,Q8s,Q7s,Q6s,Q5s,Q4s,Q3s,Q2s,Q9o,Q8o,Q7o,J9s,J8s,J7s,J6s,J5s,J4s,J9o,J8o,J7o,T8s,T7s,T6s,T5s,T4s,T8o,T7o,98s,97s,96s,95s,94s,97o,96o,87s,86s,85s,84s,83s,86o,85o,76s,75s,74s,73s,72s,75o,74o,65s,64s,63s,62s,64o,63o,54s,53s,52s,53o,52o,43s,42s,32s",
            Position.SB: "77,66,55,44,33,22,A7s,A6s,A5s,A4s,A3s,A2s,A7o,A6o,A5o,A4o,A3o,A2o,K8s,K7s,K6s,K5s,K4s,K3s,K2s,K8o,K7o,K6o,K5o,K4o,Q8s,Q7s,Q6s,Q5s,Q4s,Q3s,Q2s,Q8o,Q7o,Q6o,Q5o,J8s,J7s,J6s,J5s,J4s,J3s,J8o,J7o,J6o,T7s,T6s,T5s,T4s,T3s,T7o,T6o,97s,96s,95s,94s,93s,96o,95o,86s,85s,84s,83s,82s,85o,84o,75s,74s,73s,72s,74o,73o,64s,63s,62s,63o,62o,53s,52s,52o,42s,32s",
        },
    }

    @classmethod
    def get_opening_range(cls, position: Position) -> HandRange:
        """
        Get the opening (RFI) range for a position.

        Args:
            position: Table position

        Returns:
            HandRange for opening from that position
        """
        notation = cls.OPENING_RANGES.get(position, "")
        return HandRange(notation) if notation else HandRange()

    @classmethod
    def get_3bet_range(
        cls,
        hero_position: Position,
        villain_position: Position,
    ) -> HandRange:
        """
        Get the 3-bet range vs an open.

        Args:
            hero_position: Your position
            villain_position: Opener's position

        Returns:
            HandRange for 3-betting
        """
        if villain_position not in cls.THREE_BET_RANGES:
            return HandRange()

        ranges = cls.THREE_BET_RANGES[villain_position]
        notation = ranges.get(hero_position, "")
        return HandRange(notation) if notation else HandRange()

    @classmethod
    def get_calling_range(
        cls,
        hero_position: Position,
        villain_position: Position,
    ) -> HandRange:
        """
        Get the calling range vs an open.

        Args:
            hero_position: Your position
            villain_position: Opener's position

        Returns:
            HandRange for calling
        """
        if hero_position not in cls.CALLING_RANGES:
            return HandRange()

        ranges = cls.CALLING_RANGES[hero_position]
        notation = ranges.get(villain_position, "")
        return HandRange(notation) if notation else HandRange()


def get_opening_range(position: Position | str) -> HandRange:
    """
    Get opening range for position.

    Args:
        position: Position enum or string like "BTN", "CO"

    Returns:
        HandRange for that position
    """
    if isinstance(position, str):
        position = Position[position.upper()]
    return PreflopRanges.get_opening_range(position)


def get_3bet_range(hero_position: Position | str, villain_position: Position | str) -> HandRange:
    """
    Get 3-bet range vs opener.

    Args:
        hero_position: Your position
        villain_position: Opener position

    Returns:
        HandRange for 3-betting
    """
    if isinstance(hero_position, str):
        hero_position = Position[hero_position.upper()]
    if isinstance(villain_position, str):
        villain_position = Position[villain_position.upper()]
    return PreflopRanges.get_3bet_range(hero_position, villain_position)


def get_calling_range(hero_position: Position | str, villain_position: Position | str) -> HandRange:
    """
    Get calling range vs opener.

    Args:
        hero_position: Your position
        villain_position: Opener position

    Returns:
        HandRange for calling
    """
    if isinstance(hero_position, str):
        hero_position = Position[hero_position.upper()]
    if isinstance(villain_position, str):
        villain_position = Position[villain_position.upper()]
    return PreflopRanges.get_calling_range(hero_position, villain_position)
