"""Tests for US-S5-04 — src/features/voting_network.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.voting_network import (
    _jury_top_n_per_year,
    build_jury_cooccurrence,
    build_links,
    build_network,
    build_nodes,
    save_network,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small deterministic dataset: 3 countries, 4 Grand Final years."""
    rng = np.random.default_rng(42)
    years = [2018, 2019, 2021, 2022]
    countries = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    rows = []
    for year in years:
        for i, country in enumerate(countries):
            rows.append({
                "Year": year,
                "Country": country,
                "Country_Group": "Western",
                "Grand_Final_Ind": 1,
                "Big6_Ind": 1 if country == "Alpha" else 0,
                "jury_points": float(100 - i * 10 + (year - 2018)),
                "tele_points": rng.uniform(50, 150),
                "Final_Place": float(i + 1),
            })
    # add 2026 participants (target year, no Grand_Final history needed here)
    for i, country in enumerate(countries):
        rows.append({
            "Year": 2026,
            "Country": country,
            "Country_Group": "Western",
            "Grand_Final_Ind": 1,
            "Big6_Ind": 1 if country == "Alpha" else 0,
            "jury_points": np.nan,  # 2026 jury_points not yet known
            "tele_points": np.nan,
            "Final_Place": np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _jury_top_n_per_year
# ---------------------------------------------------------------------------


def test_jury_top_n_per_year_keys(synthetic_df):
    result = _jury_top_n_per_year(synthetic_df, top_n=3)
    assert set(result.keys()) == {2018, 2019, 2021, 2022}


def test_jury_top_n_per_year_size(synthetic_df):
    result = _jury_top_n_per_year(synthetic_df, top_n=3)
    for year, countries in result.items():
        assert len(countries) == 3, f"Year {year}: expected 3 countries, got {len(countries)}"


def test_jury_top_n_per_year_excludes_target_year(synthetic_df):
    result = _jury_top_n_per_year(synthetic_df, top_n=3)
    assert 2026 not in result


def test_jury_top_n_per_year_excludes_null_jury(synthetic_df):
    # inject a null jury row for one country in one year
    df = synthetic_df.copy()
    mask = (df["Year"] == 2018) & (df["Country"] == "Alpha")
    df.loc[mask, "jury_points"] = np.nan
    result = _jury_top_n_per_year(df, top_n=3)
    assert "Alpha" not in result[2018]


# ---------------------------------------------------------------------------
# build_jury_cooccurrence
# ---------------------------------------------------------------------------


def test_cooccurrence_keys_are_sorted_pairs(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    for a, b in cooc:
        assert a < b, f"Pair ({a}, {b}) not in lexicographic order"


def test_cooccurrence_weight_positive(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    for pair, data in cooc.items():
        assert data["weight"] > 0
        assert len(data["years"]) == data["weight"]


def test_cooccurrence_years_sorted(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    for pair, data in cooc.items():
        assert data["years"] == sorted(data["years"])


def test_cooccurrence_alpha_beta_always_together(synthetic_df):
    # Alpha has highest jury_points every year (100, 101, 102, 103)
    # Beta is second every year (90, 91, 92, 93)
    # With top_n=2 they should co-occur in all 4 years
    cooc = build_jury_cooccurrence(synthetic_df, top_n=2)
    pair = ("Alpha", "Beta")
    assert pair in cooc
    assert cooc[pair]["weight"] == 4


def test_cooccurrence_no_pairs_with_all_nan(synthetic_df):
    df = synthetic_df.copy()
    # wipe all jury_points for historical years
    df.loc[df["Year"] < 2026, "jury_points"] = np.nan
    cooc = build_jury_cooccurrence(df, top_n=3)
    assert len(cooc) == 0


# ---------------------------------------------------------------------------
# build_nodes
# ---------------------------------------------------------------------------


def test_build_nodes_count(synthetic_df, tmp_path):
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    assert len(nodes) == 5  # Alpha, Beta, Gamma, Delta, Epsilon


def test_build_nodes_required_keys(synthetic_df, tmp_path):
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    for node in nodes:
        for key in ("id", "group", "probability", "prediction", "big6", "appearances"):
            assert key in node, f"Missing key {key!r} in node {node}"


def test_build_nodes_probability_none_without_narratives(synthetic_df, tmp_path):
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    for node in nodes:
        assert node["probability"] is None


def test_build_nodes_probability_loaded_from_json(synthetic_df, tmp_path):
    narratives_path = tmp_path / "narratives_2026.json"
    payload = {
        "story": "US-S5-03",
        "target_year": 2026,
        "countries": [
            {"country": "Alpha", "probability": 0.85},
            {"country": "Beta",  "probability": 0.40},
        ],
    }
    narratives_path.write_text(json.dumps(payload), encoding="utf-8")
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["Alpha"]["probability"] == pytest.approx(0.85, abs=1e-4)
    assert by_id["Beta"]["probability"]  == pytest.approx(0.40, abs=1e-4)
    assert by_id["Gamma"]["probability"] is None


def test_build_nodes_big6_flag(synthetic_df, tmp_path):
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["Alpha"]["big6"] is True
    assert by_id["Beta"]["big6"]  is False


def test_build_nodes_appearances(synthetic_df, tmp_path):
    # Alpha appears in 4 historical Grand Finals (2018,2019,2021,2022)
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["Alpha"]["appearances"] == 4


def test_build_nodes_prediction_field(synthetic_df, tmp_path):
    narratives_path = tmp_path / "narratives_2026.json"
    payload = {
        "story": "US-S5-03",
        "target_year": 2026,
        "countries": [
            {"country": "Alpha", "probability": 0.85},
            {"country": "Beta",  "probability": 0.30},
        ],
    }
    narratives_path.write_text(json.dumps(payload), encoding="utf-8")
    nodes = build_nodes(synthetic_df, target_year=2026, reports_dir=tmp_path)
    by_id = {n["id"]: n for n in nodes}
    assert by_id["Alpha"]["prediction"] == "top10"
    assert by_id["Beta"]["prediction"]  == "outside_top10"
    # no probability → prediction defaults to outside_top10
    assert by_id["Gamma"]["prediction"] == "outside_top10"


# ---------------------------------------------------------------------------
# build_links
# ---------------------------------------------------------------------------


def test_build_links_min_weight_filter(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    node_ids = {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}
    links_2 = build_links(cooc, node_ids, min_weight=2)
    links_3 = build_links(cooc, node_ids, min_weight=3)
    assert all(e["weight"] >= 2 for e in links_2)
    assert all(e["weight"] >= 3 for e in links_3)
    assert len(links_2) >= len(links_3)


def test_build_links_node_filter(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    # exclude Delta and Epsilon
    subset = {"Alpha", "Beta", "Gamma"}
    links = build_links(cooc, subset, min_weight=1)
    for e in links:
        assert e["source"] in subset
        assert e["target"] in subset


def test_build_links_required_keys(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    node_ids = {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}
    links = build_links(cooc, node_ids, min_weight=1)
    for e in links:
        for key in ("source", "target", "weight", "years"):
            assert key in e


def test_build_links_sorted_by_weight_desc(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    node_ids = {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}
    links = build_links(cooc, node_ids, min_weight=1)
    weights = [e["weight"] for e in links]
    assert weights == sorted(weights, reverse=True)


def test_build_links_years_match_weight(synthetic_df):
    cooc = build_jury_cooccurrence(synthetic_df, top_n=3)
    node_ids = {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}
    links = build_links(cooc, node_ids, min_weight=1)
    for e in links:
        assert len(e["years"]) == e["weight"]


# ---------------------------------------------------------------------------
# build_network — integration
# ---------------------------------------------------------------------------


def test_build_network_meta_keys(tmp_path):
    """build_network on synthetic CSV returns correct meta structure."""
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df,
        target_year=2026,
        top_n=3,
        min_weight=1,
        reports_dir=tmp_path,
    )
    meta = network["meta"]
    for key in ("story", "generated_at", "target_year", "top_n_jury",
                "min_weight", "n_nodes", "n_edges", "gf_years_used"):
        assert key in meta


def test_build_network_node_count(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    assert network["meta"]["n_nodes"] == len(network["nodes"])
    assert network["meta"]["n_nodes"] == 5


def test_build_network_edge_count_consistent(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    assert network["meta"]["n_edges"] == len(network["links"])


def test_build_network_story_tag(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    assert network["meta"]["story"] == "US-S5-04"


def test_build_network_gf_years_excludes_target(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    assert 2026 not in network["meta"]["gf_years_used"]


# ---------------------------------------------------------------------------
# save_network
# ---------------------------------------------------------------------------


def test_save_network_writes_json(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    out = save_network(network, reports_dir=tmp_path)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "links" in data
    assert "meta" in data


def test_save_network_filename(tmp_path):
    df = _make_synthetic_csv(tmp_path)
    network = build_network(
        data_path=df, target_year=2026, top_n=3, min_weight=1, reports_dir=tmp_path
    )
    out = save_network(network, reports_dir=tmp_path)
    assert out.name == "voting_network_2026.json"


# ---------------------------------------------------------------------------
# Helper — write synthetic data CSV
# ---------------------------------------------------------------------------


def _make_synthetic_csv(tmp_path: Path) -> Path:
    """Write synthetic_df to a CSV and return the path."""
    rng = np.random.default_rng(0)
    years = [2018, 2019, 2021, 2022]
    countries = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    rows = []
    for year in years:
        for i, country in enumerate(countries):
            rows.append({
                "Year": year,
                "Country": country,
                "Country_Group": "Western",
                "Grand_Final_Ind": 1,
                "Big6_Ind": 1 if country == "Alpha" else 0,
                "jury_points": float(100 - i * 10 + (year - 2018)),
                "tele_points": rng.uniform(50, 150),
                "Final_Place": float(i + 1),
            })
    for i, country in enumerate(countries):
        rows.append({
            "Year": 2026,
            "Country": country,
            "Country_Group": "Western",
            "Grand_Final_Ind": 1,
            "Big6_Ind": 1 if country == "Alpha" else 0,
            "jury_points": np.nan,
            "tele_points": np.nan,
            "Final_Place": np.nan,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "enriched.csv"
    df.to_csv(path, index=False)
    return path
