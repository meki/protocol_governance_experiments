from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Protocolization Frontier evidence package")
    parser.add_argument("command", choices=["run-all"])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Experiments directory (defaults to the installed source tree)",
    )
    args = parser.parse_args()
    if args.command == "run-all":
        outputs = run_all(args.root.resolve())
        for name, path in outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()

