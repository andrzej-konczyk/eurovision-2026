"""Navigation checks for Sprint 10."""

from __future__ import annotations

import app


def test_navigation_has_distinct_voting_pages():
    assert app.NAVIGATION_PAGES.count("Voting Blocs") == 1
    assert app.NAVIGATION_PAGES.count("Voting Network") == 1
    assert "Backtest" not in app.NAVIGATION_PAGES
    assert "Data Health" not in app.NAVIGATION_PAGES
    assert len(app.NAVIGATION_PAGES) == len(set(app.NAVIGATION_PAGES))
    assert set(app.NAVIGATION_PAGES).issubset(app.PAGE_CAPTIONS)


def test_data_health_keeps_voting_blocs_and_network_artifacts_separate():
    data = app.load_dashboard_data()
    checks = app.data_health_checks(data, load_time_s=0.5)

    voting_checks = checks[checks["check"].str.startswith("Voting")]

    assert set(voting_checks["check"]) == {"Voting network JSON", "Voting bloc co-occurrence CSV"}
    assert voting_checks["path"].is_unique
    assert data["voting_network_path"] != data["bloc_cooccurrence_path"]


def test_artifact_rows_are_unique_and_include_runtime():
    data = app.load_dashboard_data()
    rows = app.dashboard_artifact_rows(data, load_time_s=0.5)

    paths = [row["path"] for row in rows]

    assert len(paths) == len(set(paths))
    assert paths[-1] == "runtime"
