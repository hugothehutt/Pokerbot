#!/usr/bin/env python3
"""
Pokerbot CLI - Interactive poker analysis tool.

Commands:
- equity: Calculate hand/range equity
- analyze: Analyze a specific spot
- preflop: Get preflop recommendations
- range: Display hand ranges
- simulate: Run simulations
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional

from pokerbot.core.hand import Hand
from pokerbot.core.card import Card
from pokerbot.equity.calculator import EquityCalculator
from pokerbot.equity.range import HandRange
from pokerbot.strategy.gto.ranges import get_opening_range, get_3bet_range, get_calling_range
from pokerbot.strategy.decision import DecisionEngine
from pokerbot.bot.gto_bot import GTOBot, create_bot
from pokerbot.game.player import Position


def print_banner():
    """Print the pokerbot banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                      POKERBOT v0.1.0                       ║
║         Texas Hold'em Strategy Analysis Tool              ║
║                                                           ║
║  Monte Carlo Simulation • GTO Concepts • Range Analysis   ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def cmd_equity(args: argparse.Namespace) -> None:
    """Calculate equity between hands/ranges."""
    calc = EquityCalculator(default_simulations=args.simulations)

    hand1 = args.hand1
    hand2 = args.hand2
    board = args.board

    print(f"\nCalculating equity ({args.simulations} simulations)...\n")

    # Determine if inputs are hands or ranges
    is_range1 = "," in hand1 or "+" in hand1 or "-" in hand1
    is_range2 = "," in hand2 or "+" in hand2 or "-" in hand2

    if is_range1 and is_range2:
        result = calc.range_vs_range(hand1, hand2, board)
        print(f"Range 1: {hand1}")
        print(f"Range 2: {hand2}")
    elif is_range1 or is_range2:
        if is_range2:
            result = calc.hand_vs_range(hand1, hand2, board)
            print(f"Hand: {hand1}")
            print(f"Range: {hand2}")
        else:
            result = calc.hand_vs_range(hand2, hand1, board)
            print(f"Hand: {hand2}")
            print(f"Range: {hand1}")
    else:
        result = calc.hand_vs_hand(hand1, hand2, board)
        print(f"Hand 1: {hand1}")
        print(f"Hand 2: {hand2}")

    if board:
        print(f"Board: {board}")

    print(f"\n{result}")
    print(f"95% CI: [{result.confidence_interval[0]*100:.2f}%, {result.confidence_interval[1]*100:.2f}%]")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a specific poker spot."""
    engine = DecisionEngine(simulations=args.simulations)

    hand = args.hand
    board = args.board or ""
    villain_range = args.villain or "22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,A9o+,KTo+,QTo+"
    pot = args.pot
    to_call = args.call

    print(f"\nAnalyzing spot...")
    print(f"Hand: {hand}")
    if board:
        print(f"Board: {board}")
    print(f"Villain Range: {villain_range}")
    print(f"Pot: {pot}")
    if to_call > 0:
        print(f"To Call: {to_call}")

    analysis = engine.analyze_spot(
        hand=hand,
        board=board,
        villain_range=villain_range,
        pot=pot,
        to_call=to_call,
    )

    print(f"\n{'='*50}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"\nEquity: {analysis['equity_pct']}")
    print(f"Pot Odds: {analysis['pot_odds']:.2f}:1")
    print(f"Required Equity: {analysis['required_equity_pct']}")
    print(f"Profitable Call: {'Yes ✓' if analysis['profitable_call'] else 'No ✗'}")

    print(f"\nGTO Strategy Frequencies:")
    for action, freq in analysis['strategy'].items():
        if float(freq.rstrip('%')) > 0:
            print(f"  {action.capitalize()}: {freq}")


def cmd_preflop(args: argparse.Namespace) -> None:
    """Get preflop recommendations."""
    hand = args.hand
    position = args.position.upper()

    try:
        pos = Position[position]
    except KeyError:
        print(f"Invalid position: {position}")
        print("Valid positions: UTG, UTG1, UTG2, LJ, HJ, CO, BTN, SB, BB")
        return

    hand_obj = Hand(hand)
    print(f"\nHand: {hand_obj.pretty} ({hand_obj.notation()})")
    print(f"Position: {position}")

    # Check ranges
    opening_range = get_opening_range(pos)
    in_opening = hand_obj in opening_range

    print(f"\n{'='*50}")
    print("PREFLOP ANALYSIS")
    print(f"{'='*50}")

    print(f"\nOpening Range ({opening_range.percentage:.1f}% of hands):")
    print(f"  {'✓ IN RANGE' if in_opening else '✗ NOT IN RANGE'}")

    if args.vs_raise:
        print(f"\nVs Open from {args.vs_raise}:")
        try:
            opener_pos = Position[args.vs_raise.upper()]
            three_bet = get_3bet_range(pos, opener_pos)
            calling = get_calling_range(pos, opener_pos)

            in_3bet = hand_obj in three_bet
            in_call = hand_obj in calling

            print(f"  3-Bet Range: {'✓ 3-BET' if in_3bet else '✗ Not in 3-bet range'}")
            print(f"  Call Range: {'✓ CALL' if in_call else '✗ Not in calling range'}")

            if not in_3bet and not in_call:
                print(f"  Recommendation: FOLD")
            elif in_3bet:
                print(f"  Recommendation: 3-BET")
            else:
                print(f"  Recommendation: CALL")
        except KeyError:
            print(f"  Invalid opener position: {args.vs_raise}")


def cmd_range(args: argparse.Namespace) -> None:
    """Display hand ranges."""
    position = args.position.upper() if args.position else None

    if position:
        try:
            pos = Position[position]
            opening = get_opening_range(pos)

            print(f"\n{'='*50}")
            print(f"OPENING RANGE: {position}")
            print(f"{'='*50}")
            print(f"Hands: {opening.num_notations}")
            print(f"Combos: {opening.num_combinations}")
            print(f"Percentage: {opening.percentage:.1f}%")
            print(f"\nRange: {opening}")

        except KeyError:
            print(f"Invalid position: {position}")
    else:
        # Show all positions
        print(f"\n{'='*50}")
        print("OPENING RANGES BY POSITION (6-max)")
        print(f"{'='*50}")

        for pos in [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB]:
            opening = get_opening_range(pos)
            print(f"\n{pos.name}: {opening.percentage:.1f}% ({opening.num_combinations} combos)")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a poker simulation."""
    bot = create_bot(
        name="SimBot",
        simulations=args.simulations,
    )

    hand = args.hand
    board = args.board or ""
    villain = args.villain or "random"

    print(f"\nRunning simulation...")
    print(f"Hand: {hand}")
    if board:
        print(f"Board: {board}")
    print(f"Villain: {villain}")

    analysis = bot.analyze_hand(
        hole_cards=hand,
        board=board,
        villain_range=villain if villain != "random" else "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,87s,76s,65s,54s,A2o+,K7o+,Q8o+,J8o+,T8o+,98o",
        pot=10.0,
        to_call=0.0,
    )

    print(f"\n{analysis}")


def cmd_interactive(args: argparse.Namespace) -> None:
    """Run interactive mode."""
    print_banner()
    print("Type 'help' for commands, 'quit' to exit.\n")

    calc = EquityCalculator()
    bot = create_bot(simulations=5000)

    while True:
        try:
            line = input("pokerbot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        elif cmd == 'help':
            print("""
Commands:
  equity <hand1> <hand2> [board]  - Calculate equity
  analyze <hand> [board]          - Analyze a spot
  preflop <hand> <position>       - Preflop recommendation
  range [position]                - Show opening ranges
  quit                            - Exit

Examples:
  equity AsKs QhQd
  equity AsKs QhQd "Ks Qc 2d"
  equity "TT+,AQs+" "JJ+,AKs"
  analyze AsKs "Ks Qh 2d"
  preflop AsKs CO
  preflop 77 BTN --vs UTG
  range CO
""")

        elif cmd == 'equity' and len(parts) >= 3:
            hand1 = parts[1]
            hand2 = parts[2]
            board = " ".join(parts[3:]) if len(parts) > 3 else None

            try:
                result = calc.hand_vs_hand(hand1, hand2, board)
                print(f"Equity: {result}")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == 'analyze' and len(parts) >= 2:
            hand = parts[1]
            board = " ".join(parts[2:]) if len(parts) > 2 else ""

            try:
                analysis = bot.analyze_hand(hand, board)
                print(f"\nEquity: {analysis['equity_pct']}")
                print(f"Strategy: {analysis['strategy']}")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == 'preflop' and len(parts) >= 3:
            hand = parts[1]
            position = parts[2]

            try:
                result = bot.get_preflop_action(hand, position)
                print(f"\n{result}")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == 'range':
            position = parts[1] if len(parts) > 1 else None
            if position:
                try:
                    pos = Position[position.upper()]
                    opening = get_opening_range(pos)
                    print(f"\n{position.upper()}: {opening.percentage:.1f}%")
                    print(f"Range: {opening}")
                except KeyError:
                    print(f"Invalid position: {position}")
            else:
                for pos in [Position.UTG, Position.HJ, Position.CO, Position.BTN, Position.SB]:
                    opening = get_opening_range(pos)
                    print(f"{pos.name}: {opening.percentage:.1f}%")

        else:
            print(f"Unknown command: {cmd}. Type 'help' for commands.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Pokerbot - Texas Hold'em Strategy Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pokerbot equity AsKs QhQd
  pokerbot equity AsKs "TT+,AQs+" --board "Ks Qh 2d"
  pokerbot analyze AsKs --board "Ks Qh 2d" --pot 100 --call 50
  pokerbot preflop AsKs CO
  pokerbot preflop 77 BTN --vs UTG
  pokerbot range CO
  pokerbot  (starts interactive mode)
        """,
    )

    parser.add_argument(
        '--simulations', '-s',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)',
    )

    subparsers = parser.add_subparsers(dest='command')

    # Equity command
    equity_parser = subparsers.add_parser('equity', help='Calculate hand/range equity')
    equity_parser.add_argument('hand1', help='First hand or range (e.g., "AsKs" or "TT+,AQs+")')
    equity_parser.add_argument('hand2', help='Second hand or range')
    equity_parser.add_argument('--board', '-b', help='Board cards (e.g., "Ks Qh 2d")')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a poker spot')
    analyze_parser.add_argument('hand', help='Your hand (e.g., "AsKs")')
    analyze_parser.add_argument('--board', '-b', help='Board cards')
    analyze_parser.add_argument('--villain', '-v', help='Villain range')
    analyze_parser.add_argument('--pot', '-p', type=float, default=10.0, help='Pot size')
    analyze_parser.add_argument('--call', '-c', type=float, default=0.0, help='Amount to call')

    # Preflop command
    preflop_parser = subparsers.add_parser('preflop', help='Get preflop recommendations')
    preflop_parser.add_argument('hand', help='Your hand (e.g., "AsKs")')
    preflop_parser.add_argument('position', help='Your position (UTG, HJ, CO, BTN, SB, BB)')
    preflop_parser.add_argument('--vs', dest='vs_raise', help='Position of raiser to face')

    # Range command
    range_parser = subparsers.add_parser('range', help='Display hand ranges')
    range_parser.add_argument('position', nargs='?', help='Position (optional, shows all if not specified)')

    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('hand', help='Your hand')
    sim_parser.add_argument('--board', '-b', help='Board cards')
    sim_parser.add_argument('--villain', '-v', help='Villain range')

    args = parser.parse_args()

    if args.command == 'equity':
        cmd_equity(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'preflop':
        cmd_preflop(args)
    elif args.command == 'range':
        cmd_range(args)
    elif args.command == 'simulate':
        cmd_simulate(args)
    else:
        # No command - run interactive mode
        cmd_interactive(args)


if __name__ == "__main__":
    main()
