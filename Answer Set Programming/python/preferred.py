"""Program 5: Preferred semantics in Python.

The script parses argumentation frameworks expressed as ASP facts (`arg/1` and
`att/2`), enumerates all admissible sets via a conflict-free backtracking search,
and filters the inclusion-maximal ones (preferred extensions). A tiny random
instance generator plus a benchmarking switch are provided to help with the
report's empirical comparison section.

Example usage:
    python preferred.py --input ..\\examples\\example5.lp
    python preferred.py --random 20 --density 0.25 --seed 1 --benchmark
"""

from __future__ import annotations

import argparse
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet, Union

Atom = Union[str, int]

TOKEN = r"([A-Za-z][A-Za-z0-9_]*|-?\d+)"
ARG_RE = re.compile(rf"arg\s*\(\s*{TOKEN}\s*\)\s*\.\s*$")
ATT_RE = re.compile(rf"att\s*\(\s*{TOKEN}\s*,\s*{TOKEN}\s*\)\s*\.\s*$")


def _coerce_token(raw: str):
    raw = raw.strip()
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


@dataclass
class ArgumentationFramework:
    arguments: Set[Atom]
    attacks: Set[Tuple[Atom, Atom]]
    attackers: Dict[Atom, Set[Atom]] = field(init=False)
    targets: Dict[Atom, Set[Atom]] = field(init=False)

    def __post_init__(self) -> None:
        self.attackers = {arg: set() for arg in self.arguments}
        self.targets = {arg: set() for arg in self.arguments}
        for src, dst in self.attacks:
            if src not in self.arguments or dst not in self.arguments:
                raise ValueError(f"Attack ({src},{dst}) references unknown argument")
            self.targets[src].add(dst)
            self.attackers[dst].add(src)

    @property
    def size(self) -> Tuple[int, int]:
        return len(self.arguments), len(self.attacks)

    def degree(self, arg: Atom) -> int:
        return len(self.attackers[arg]) + len(self.targets[arg])


def parse_asp_facts(path: Path) -> ArgumentationFramework:
    arguments: Set[Atom] = set()
    attacks: Set[Tuple[Atom, Atom]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.split("%", 1)[0].strip()
        if not payload:
            continue
        if match := ARG_RE.match(payload):
            arguments.add(_coerce_token(match.group(1)))
            continue
        if match := ATT_RE.match(payload):
            src = _coerce_token(match.group(1))
            dst = _coerce_token(match.group(2))
            attacks.add((src, dst))
            continue
        raise ValueError(f"Cannot parse line: {line!r}")
    if not arguments:
        raise ValueError("No arguments declared.")
    return ArgumentationFramework(arguments, attacks)


def random_framework(size: int, density: float, seed: Optional[int]) -> ArgumentationFramework:
    rng = random.Random(seed)
    arguments: Set[Atom] = {f"a{i+1}" for i in range(size)}
    attacks: Set[Tuple[Atom, Atom]] = set()
    ordered = sorted(arguments)
    for src in ordered:
        for dst in ordered:
            if rng.random() <= density:
                attacks.add((src, dst))
    return ArgumentationFramework(arguments, attacks)


def is_conflict_free_with(arg: Atom, current: Set[Atom], af: ArgumentationFramework) -> bool:
    return all(arg not in af.targets[other] and other not in af.targets[arg] for other in current)


def is_admissible(candidate: Set[Atom], af: ArgumentationFramework) -> bool:
    for arg in candidate:
        for attacker in af.attackers[arg]:
            if attacker not in candidate:
                if not any(attacker in af.targets[defender] for defender in candidate):
                    return False
            elif attacker in candidate:
                # Guard: conflict-free generation should prevent this.
                return False
    return True


def enumerate_admissible_sets(af: ArgumentationFramework) -> List[FrozenSet[Atom]]:
    order = sorted(af.arguments, key=lambda a: (-af.degree(a), str(a)))
    results: List[FrozenSet[Atom]] = []
    chosen: Set[Atom] = set()

    def dfs(index: int) -> None:
        if index == len(order):
            if is_admissible(chosen, af):
                results.append(frozenset(chosen))
            return
        arg = order[index]
        dfs(index + 1)
        if is_conflict_free_with(arg, chosen, af):
            chosen.add(arg)
            dfs(index + 1)
            chosen.remove(arg)

    dfs(0)
    return results


def maximal_extensions(candidates: Iterable[FrozenSet[Atom]]) -> List[FrozenSet[Atom]]:
    unique: List[FrozenSet[Atom]] = []
    for cand in candidates:
        if cand not in unique:
            unique.append(cand)
    maximal = [cand for cand in unique if not any(cand < other for other in unique)]
    maximal.sort(key=lambda ext: tuple(str(arg) for arg in sorted(ext, key=str)))
    return maximal


def preferred_extensions(af: ArgumentationFramework) -> List[FrozenSet[Atom]]:
    return maximal_extensions(enumerate_admissible_sets(af))


def format_extension(extension: FrozenSet[Atom]) -> str:
    if not extension:
        return "(empty)"
    ordered = sorted(extension, key=str)
    return " ".join(f"in({arg})" for arg in ordered)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute preferred extensions imperatively.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to a file with arg/att facts.")
    source.add_argument("--random", type=int, metavar="N", help="Generate a random AF with N arguments.")
    parser.add_argument("--density", type=float, default=0.2, help="Attack probability for random AF generation.")
    parser.add_argument("--seed", type=int, help="Seed for the random generator.")
    parser.add_argument("--show-admissible", action="store_true", help="Display all admissible sets before filtering.")
    parser.add_argument("--benchmark", action="store_true", help="Print runtime information.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)

    if args.input:
        af = parse_asp_facts(args.input)
    else:
        if args.random <= 0:
            parser.error("--random must be positive")
        if not 0.0 <= args.density <= 1.0:
            parser.error("--density must be between 0 and 1")
        af = random_framework(args.random, args.density, args.seed)

    print(f"Loaded AF with {af.size[0]} arguments and {af.size[1]} attacks.")

    started = time.perf_counter()
    admissible_sets = enumerate_admissible_sets(af)
    preferred = maximal_extensions(admissible_sets)
    elapsed = time.perf_counter() - started

    if args.show_admissible:
        print("Admissible sets:")
        for idx, ext in enumerate(sorted(admissible_sets, key=lambda e: tuple(str(a) for a in sorted(e, key=str))), start=1):
            print(f"  A{idx}: {format_extension(ext)}")

    print("Preferred extensions:")
    if preferred:
        for idx, ext in enumerate(preferred, start=1):
            print(f"  P{idx}: {format_extension(ext)}")
    else:
        print("  (none)")

    if args.benchmark:
        print(f"Computation time: {elapsed * 1000:.2f} ms")


if __name__ == "__main__":
    main()
