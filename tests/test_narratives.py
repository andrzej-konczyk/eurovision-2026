"""Tests for US-S5-03 — src/models/narratives.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.narratives import (
    _phrase,
    _PHRASES,
    build_narrative,
    generate_narratives,
)


# ---------------------------------------------------------------------------
# _phrase
# ---------------------------------------------------------------------------


def test_phrase_known_positive():
    result = _phrase("implied_prob_close", 0.3)
    assert "betting" in result.lower()


def test_phrase_known_negative():
    result = _phrase("implied_prob_close", -0.3)
    assert "limited" in result.lower()


def test_phrase_unknown_feature_fallback():
    result = _phrase("unknown_feature_xyz", 0.1)
    assert isinstance(result, str)
    assert len(result) > 0


def test_phrase_zero_shap_is_positive():
    # SHAP = 0 counts as non-negative → positive phrase
    result = _phrase("avg_final_rank_3yr", 0.0)
    pos, _ = _PHRASES["avg_final_rank_3yr"]
    assert result == pos


def test_all_known_features_have_non_empty_phrases():
    for feat, (pos, neg) in _PHRASES.items():
        assert len(pos) > 0, f"Empty positive phrase for {feat}"
        assert len(neg) > 0, f"Empty negative phrase for {feat}"


# ---------------------------------------------------------------------------
# build_narrative
# ---------------------------------------------------------------------------


@pytest.fixture
def top5_mixed() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature": "implied_prob_close",  "shap_value":  0.31, "rank": 1},
        {"feature": "avg_final_rank_3yr",  "shap_value":  0.18, "rank": 2},
        {"feature": "avg_tele_3yr",        "shap_value":  0.07, "rank": 3},
        {"feature": "Running_Order_Final", "shap_value": -0.12, "rank": 4},
        {"feature": "zscore_ogae_points",  "shap_value": -0.04, "rank": 5},
    ])


@pytest.fixture
def top5_all_positive() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature": "implied_prob_close", "shap_value": 0.30, "rank": 1},
        {"feature": "avg_final_rank_3yr", "shap_value": 0.20, "rank": 2},
        {"feature": "avg_jury_3yr",       "shap_value": 0.15, "rank": 3},
        {"feature": "avg_tele_3yr",       "shap_value": 0.10, "rank": 4},
        {"feature": "zscore_ogae_points", "shap_value": 0.05, "rank": 5},
    ])


@pytest.fixture
def top5_all_negative() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature": "implied_prob_close",  "shap_value": -0.25, "rank": 1},
        {"feature": "avg_final_rank_3yr",  "shap_value": -0.18, "rank": 2},
        {"feature": "avg_tele_3yr",        "shap_value": -0.10, "rank": 3},
        {"feature": "avg_bloc_jury_3yr",   "shap_value": -0.06, "rank": 4},
        {"feature": "zscore_myesb_community", "shap_value": -0.03, "rank": 5},
    ])


def test_build_narrative_returns_string(top5_mixed):
    result = build_narrative("Sweden", 0.78, top5_mixed)
    assert isinstance(result, str)
    assert len(result) > 50


def test_build_narrative_mentions_country(top5_mixed):
    result = build_narrative("Sweden", 0.78, top5_mixed)
    assert "Sweden" in result


def test_build_narrative_high_prob_label(top5_mixed):
    result = build_narrative("Sweden", 0.78, top5_mixed)
    assert "strong" in result.lower()


def test_build_narrative_mid_prob_label(top5_mixed):
    result = build_narrative("Moldova", 0.48, top5_mixed)
    assert "borderline" in result.lower()


def test_build_narrative_low_prob_label(top5_all_negative):
    result = build_narrative("Estonia", 0.20, top5_all_negative)
    assert "unlikely" in result.lower()


def test_build_narrative_2_to_4_sentences(top5_mixed):
    result = build_narrative("Sweden", 0.78, top5_mixed)
    # Count sentence-ending punctuation
    n_sentences = result.count(".")
    assert 2 <= n_sentences <= 4


def test_build_narrative_all_positive_no_limiting_factor(top5_all_positive):
    result = build_narrative("Italy", 0.80, top5_all_positive)
    assert "limiting" not in result.lower()


def test_build_narrative_all_negative_no_positive_signal(top5_all_negative):
    result = build_narrative("Estonia", 0.15, top5_all_negative)
    assert "no strong positive" in result.lower()


def test_build_narrative_mixed_includes_both_directions(top5_mixed):
    result = build_narrative("Sweden", 0.78, top5_mixed)
    # Should mention at least one positive and one limiting factor
    assert "betting" in result.lower() or "track record" in result.lower()
    assert "limiting" in result.lower() or "challenging" in result.lower()


# ---------------------------------------------------------------------------
# generate_narratives — integration smoke test
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_shap_csv(tmp_path: Path) -> Path:
    countries = ["Alpha", "Beta", "Gamma"]
    features = [
        "implied_prob_close", "avg_final_rank_3yr", "avg_tele_3yr",
        "Running_Order_Final", "zscore_ogae_points"
    ]
    rows = []
    for country in countries:
        for rank, feat in enumerate(features, 1):
            rows.append({
                "country": country, "rank": rank,
                "feature": feat,
                "shap_value": (0.3 - rank * 0.05),
                "feature_value": 0.5,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "shap_top5_lgbm.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_enriched_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    years = [2021, 2022, 2023, 2026]
    countries = ["Alpha", "Beta", "Gamma"]
    rows = []
    for year in years:
        for i, country in enumerate(countries):
            finalist = True
            rows.append({
                "Year": year, "Country": country,
                "Country_Group": "Western",
                "Grand_Final_Ind": 1,
                "Final_Place": float(i + 1),
                "Final_Points": rng.uniform(100, 500),
                "jury_points": rng.uniform(50, 200),
                "tele_points": rng.uniform(50, 200),
                "Top 10": 1.0 if i == 0 else 0.0,
                "Big6_Ind": 0, "National_Final": 1, "Solo_Artist": 1,
                "Returning_Artist_Ind": 0, "Number of Members": 1,
                "Multiple_Language": 0, "EU": 1, "NATO": 1,
                "Qualification_Record": 0.7,
                "Semi_Final_Num": np.nan, "Running_Order_Semi": np.nan,
                "Running_Order_Final": float(i + 1),
                "MyESB_Community": rng.uniform(10, 100),
                "MyESB_Personal": rng.uniform(10, 100),
                "OGAE_Points": rng.uniform(10, 500),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "enriched.csv"
    df.to_csv(path, index=False)
    return path


_MOCK_PROBS = pd.DataFrame({
    "country": ["Alpha", "Beta", "Gamma"],
    "probability": [0.75, 0.45, 0.20],
})


def _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv, **kwargs):
    """Helper: run generate_narratives with mlflow and _get_probabilities mocked."""
    with patch("src.models.narratives.mlflow"), \
         patch("src.models.narratives._get_probabilities", return_value=_MOCK_PROBS.copy()):
        return generate_narratives(
            data_path=synthetic_enriched_csv,
            target_year=2026,
            model_name="lgbm",
            out_dir=tmp_path,
            reports_dir=tmp_path,
            **kwargs,
        )


def test_generate_narratives_returns_list(
    tmp_path: Path,
    synthetic_shap_csv: Path,
    synthetic_enriched_csv: Path,
) -> None:
    cards = _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv)
    assert isinstance(cards, list)
    assert len(cards) == 3  # Alpha, Beta, Gamma


def test_generate_narratives_sorted_by_probability(
    tmp_path: Path,
    synthetic_shap_csv: Path,
    synthetic_enriched_csv: Path,
) -> None:
    cards = _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv)
    probs = [c["probability"] for c in cards]
    assert probs == sorted(probs, reverse=True)


def test_generate_narratives_writes_json(
    tmp_path: Path,
    synthetic_shap_csv: Path,
    synthetic_enriched_csv: Path,
) -> None:
    _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv)
    json_path = tmp_path / "narratives_2026.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["story"] == "US-S5-03"
    assert data["target_year"] == 2026
    assert len(data["countries"]) == 3


def test_generate_narratives_writes_markdown(
    tmp_path: Path,
    synthetic_shap_csv: Path,
    synthetic_enriched_csv: Path,
) -> None:
    _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv)
    md_path = tmp_path / "narratives_2026.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Eurovision 2026" in content
    assert "Alpha" in content


def test_generate_narratives_card_keys(
    tmp_path: Path,
    synthetic_shap_csv: Path,
    synthetic_enriched_csv: Path,
) -> None:
    cards = _run_generate(tmp_path, synthetic_shap_csv, synthetic_enriched_csv)
    for card in cards:
        assert "country" in card
        assert "probability" in card
        assert "narrative" in card
        assert "prediction" in card
        assert card["prediction"] in ("top10", "outside_top10")


def test_generate_narratives_missing_shap_raises(
    tmp_path: Path,
    synthetic_enriched_csv: Path,
) -> None:
    """If SHAP CSV is missing, FileNotFoundError with helpful message."""
    with pytest.raises(FileNotFoundError, match="shap_pipeline"):
        generate_narratives(
            data_path=synthetic_enriched_csv,
            target_year=2026,
            model_name="lgbm",
            out_dir=tmp_path,   # no shap_top5_lgbm.csv here
            reports_dir=tmp_path,
        )
