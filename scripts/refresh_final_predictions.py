"""Refresh Grand Final predictions after final draw or closing odds updates.

Default mode is a dry run: validate the 2026 Grand Final running order and
print the commands that would refresh model artefacts. Use ``--run`` only after
the draw/odds data has been written to disk and reviewed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
EXPECTED_GF_COUNTRIES = 26
TARGET_YEAR = 2026


def validate_final_running_order(
    data_path: Path,
    *,
    target_year: int = TARGET_YEAR,
    expected_count: int = EXPECTED_GF_COUNTRIES,
) -> pd.DataFrame:
    """Return validated Grand Final rows or raise ValueError with a clear reason."""
    frame = pd.read_csv(data_path)
    required = {"Year", "Country", "Grand_Final_Ind", "Running_Order_Final"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{data_path} missing required columns: {sorted(missing)}")

    target = frame[frame["Year"].eq(target_year)].copy()
    finalists = target[pd.to_numeric(target["Grand_Final_Ind"], errors="coerce").eq(1)].copy()
    if len(finalists) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Grand Final rows for {target_year}; found {len(finalists)}. "
            "Update Grand_Final_Ind after semi-finals before refreshing final predictions."
        )

    order = pd.to_numeric(finalists["Running_Order_Final"], errors="coerce")
    missing_order = finalists.loc[order.isna(), "Country"].astype(str).tolist()
    if missing_order:
        raise ValueError(
            "Running_Order_Final is missing for finalists: " + ", ".join(sorted(missing_order))
        )

    order_int = order.astype(int)
    expected_order = set(range(1, expected_count + 1))
    actual_order = set(order_int.tolist())
    if actual_order != expected_order or order_int.duplicated().any():
        duplicates = sorted(order_int[order_int.duplicated()].unique().tolist())
        missing_positions = sorted(expected_order.difference(actual_order))
        extra_positions = sorted(actual_order.difference(expected_order))
        raise ValueError(
            "Running_Order_Final must be a unique 1..26 sequence. "
            f"duplicates={duplicates}; missing={missing_positions}; extra={extra_positions}"
        )

    finalists["Running_Order_Final"] = order_int
    return finalists.sort_values("Running_Order_Final").reset_index(drop=True)


def refresh_commands(data_path: Path, odds_client_file: Path | None) -> list[list[str]]:
    commands: list[list[str]] = []
    if odds_client_file is not None:
        commands.append([
            sys.executable,
            "-m",
            "src.data.process_odds",
            "--client-file",
            str(odds_client_file),
        ])
    commands.extend(
        [
            [sys.executable, "-m", "src.models.train", "--data", str(data_path)],
            [sys.executable, "-m", "src.models.confidence", "--data", str(data_path)],
            [sys.executable, "-m", "src.models.narratives", "--data", str(data_path)],
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_predictions_json.py")],
        ]
    )
    return commands


def format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Eurovision 2026 final prediction artefacts.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Enriched CSV path")
    parser.add_argument("--target-year", type=int, default=TARGET_YEAR, help="Target contest year")
    parser.add_argument("--expected-finalists", type=int, default=EXPECTED_GF_COUNTRIES)
    parser.add_argument("--odds-client-file", type=Path, default=None, help="Optional closing odds client CSV")
    parser.add_argument("--run", action="store_true", help="Execute commands after validation")
    args = parser.parse_args()

    finalists = validate_final_running_order(
        args.data,
        target_year=args.target_year,
        expected_count=args.expected_finalists,
    )
    print(
        f"Validated {len(finalists)} Grand Final running-order rows for {args.target_year}: "
        f"{int(finalists['Running_Order_Final'].min())}-{int(finalists['Running_Order_Final'].max())}"
    )

    commands = refresh_commands(args.data, args.odds_client_file)
    if not args.run:
        print("\nDry run. Re-run with --run to execute:")
        for command in commands:
            print(f"  {format_command(command)}")
        return

    for command in commands:
        print(f"\nRunning: {format_command(command)}")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
