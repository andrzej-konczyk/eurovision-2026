"""Sprint 11 final prediction refresh safeguards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.refresh_final_predictions import refresh_commands, validate_final_running_order


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _finalist_rows(count: int = 26) -> list[dict[str, object]]:
    return [
        {
            "Year": 2026,
            "Country": f"Country {index}",
            "Grand_Final_Ind": 1,
            "Running_Order_Final": index,
        }
        for index in range(1, count + 1)
    ]


def test_validate_final_running_order_accepts_unique_26_country_sequence(tmp_path):
    data_path = tmp_path / "enriched.csv"
    _write_csv(data_path, _finalist_rows())

    finalists = validate_final_running_order(data_path)

    assert len(finalists) == 26
    assert finalists["Running_Order_Final"].tolist() == list(range(1, 27))


def test_validate_final_running_order_rejects_incomplete_finalist_count(tmp_path):
    data_path = tmp_path / "enriched.csv"
    _write_csv(data_path, _finalist_rows(25))

    with pytest.raises(ValueError, match="Expected 26 Grand Final rows"):
        validate_final_running_order(data_path)


def test_validate_final_running_order_rejects_duplicate_or_missing_positions(tmp_path):
    rows = _finalist_rows()
    rows[-1]["Running_Order_Final"] = 25
    data_path = tmp_path / "enriched.csv"
    _write_csv(data_path, rows)

    with pytest.raises(ValueError, match="unique 1..26 sequence"):
        validate_final_running_order(data_path)


def test_refresh_commands_include_odds_ingest_before_model_refresh(tmp_path):
    data_path = tmp_path / "enriched.csv"
    odds_path = tmp_path / "closing_odds.csv"

    commands = refresh_commands(data_path, odds_path)

    assert commands[0][1:4] == ["-m", "src.data.process_odds", "--client-file"]
    assert any(command[1:3] == ["-m", "src.models.train"] for command in commands)
    assert any(command[1:3] == ["-m", "src.models.confidence"] for command in commands)
    assert any(command[1:3] == ["-m", "src.models.narratives"] for command in commands)
    assert commands[-1][-1].endswith("build_predictions_json.py")
