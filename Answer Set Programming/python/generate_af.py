"""Utility: generate random argumentation frameworks as ASP facts.

Example:
    python generate_af.py --arguments 10 --attacks 20 --seed 1 --output ..\\examples\\random_n10_a20_seed1.lp
"""

from __future__ import annotations

import argparse
import random
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def build_argument_names(count: int) -> List[str]:
    return [f"a{i+1}" for i in range(count)]


def pick_attacks(args: Sequence[str], attack_count: int, allow_self: bool, seed: int) -> List[Tuple[str, str]]:
    candidates = [(src, dst) for src, dst in product(args, repeat=2) if allow_self or src != dst]
    if attack_count > len(candidates):
        raise ValueError("Requested more attacks than available unique pairs.")
    rng = random.Random(seed)
    return rng.sample(candidates, attack_count)


def format_facts(args: Iterable[str], attacks: Iterable[Tuple[str, str]]) -> str:
    parts: List[str] = []
    for arg in args:
        parts.append(f"arg({arg}).\n")
    for src, dst in attacks:
        parts.append(f"att({src},{dst}).\n")
    return "".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random AFs as ASP facts.")
    parser.add_argument("--arguments", type=int, required=True, help="Number of arguments to create.")
    parser.add_argument("--attacks", type=int, required=True, help="Number of attacks to sample (ordered pairs).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .lp file path.")
    parser.add_argument("--no-self", action="store_true", help="Disallow self-attacks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.arguments <= 0:
        raise SystemExit("--arguments must be positive")
    if args.attacks < 0:
        raise SystemExit("--attacks must be non-negative")
    names = build_argument_names(args.arguments)
    attacks = pick_attacks(names, args.attacks, allow_self=not args.no_self, seed=args.seed)
    content = format_facts(names, attacks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="ascii")
    print(f"Wrote {args.output} with {args.arguments} args and {len(attacks)} attacks (seed={args.seed}).")


if __name__ == "__main__":
    main()
