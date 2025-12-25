"""Benchmark ASP vs. Python preferred-semantics solvers.

Runs clingo on asp/preferred.lp and the imperative solver on the same
instances, measuring wall-clock time (ms) and counting preferred extensions.
Produces a Markdown table that can be pasted into the assignment report.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Allow reusing parse_asp_facts to obtain (#arguments, #attacks).
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from preferred import parse_asp_facts  # type: ignore  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ASP and Python solvers.")
    parser.add_argument(
        "--instances",
        nargs="*",
        help="Paths to .lp files with arg/att facts (defaults to examples/*.lp).",
    )
    parser.add_argument(
        "--asp-program",
        type=Path,
        default=Path("asp") / "preferred.lp",
        help="Path to the preferred-semantics ASP encoding.",
    )
    parser.add_argument(
        "--python-script",
        type=Path,
        default=Path("python") / "preferred.py",
        help="Path to the imperative solver.",
    )
    parser.add_argument(
        "--clingo",
        default="clingo",
        help="Command used to invoke clingo (default: clingo).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repetitions per tool to average timings (default: 3).",
    )
    return parser.parse_args(argv)


def discover_instances(user_paths: Iterable[str] | None) -> List[Path]:
    if user_paths:
        return [Path(path).resolve() for path in user_paths]
    default_dir = THIS_DIR.parent / "examples"
    return sorted(default_dir.glob("*.lp"))


def run_command(cmd: Sequence[str]) -> Tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in {0, 10, 20, 30}:
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, proc.stdout


def average_time(cmd: Sequence[str], runs: int) -> Tuple[float, str]:
    outputs: List[str] = []
    total = 0.0
    for _ in range(runs):
        elapsed, stdout = run_command(cmd)
        total += elapsed
        outputs.append(stdout)
    return total / runs, outputs[-1]


def count_preferred_from_asp(output: str) -> int:
    """Return the number of answer sets emittted by clingo.

    Clingo prints each model prefixed with "Answer:"; counting those lines
    works regardless of the actual atoms that follow or the quiet settings.
    """

    count = 0
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Answer:"):
            count += 1
    return count


def count_preferred_from_python(output: str) -> int:
    pattern = re.compile(r"^P\d+:")
    count = 0
    for line in output.splitlines():
        stripped = line.strip()
        if pattern.match(stripped):
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    instances = discover_instances(args.instances)
    if not instances:
        raise SystemExit("No instances found. Provide paths via --instances.")

    header = "| Instance | #Args | #Attacks | ASP time (ms) | Python time (ms) | #Preferred |"
    divider = "| --- | --- | --- | --- | --- | --- |"
    print(header)
    print(divider)

    for instance in instances:
        af = parse_asp_facts(instance)
        asp_cmd = [args.clingo, str(args.asp_program), str(instance)]
        py_cmd = [sys.executable, str(args.python_script), "--input", str(instance), "--benchmark"]

        asp_time, asp_output = average_time(asp_cmd, args.runs)
        py_time, py_output = average_time(py_cmd, args.runs)

        pref_count = count_preferred_from_asp(asp_output)
        if pref_count == 0:
            pref_count = count_preferred_from_python(py_output)

        print(f"| {instance.name} | {len(af.arguments)} | {len(af.attacks)} | " f"{asp_time:.2f} | {py_time:.2f} | {pref_count} |")


if __name__ == "__main__":
    main()
