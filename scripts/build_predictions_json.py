"""Build the Streamlit predictions JSON artifact from model confidence outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS_DIR = PROJECT_ROOT / "models" / "artefacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIDENCE_META = ARTEFACTS_DIR / "confidence_meta.json"
CONFIDENCE_XGB = ARTEFACTS_DIR / "confidence_xgb.csv"
CONFIDENCE_LGBM = ARTEFACTS_DIR / "confidence_lgbm.csv"
SURROGATE_JSON = REPORTS_DIR / "surrogate_2026.json"
NARRATIVES_JSON = REPORTS_DIR / "narratives_2026.json"
SEMI_PREDICTIONS_JSON = REPORTS_DIR / "semi_predictions_2026.json"
OUTPUT_JSON = REPORTS_DIR / "predictions_2026.json"
SF1_ACTUAL_QUALIFIERS = {
    "Greece",
    "Finland",
    "Belgium",
    "Sweden",
    "Moldova",
    "Israel",
    "Serbia",
    "Croatia",
    "Lithuania",
    "Poland",
}
SF1_ACTUAL_NON_QUALIFIERS = {
    "Portugal",
    "Georgia",
    "Montenegro",
    "Estonia",
    "San Marino",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def model_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path).sort_values("prob_mean", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)
    return frame.to_dict(orient="records")


def indexed_by_country(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["country"]: row for row in rows}


def qualification_context(country: str, semi_by_country: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if country in SF1_ACTUAL_QUALIFIERS:
        return {
            "qualification_status": "SF1 qualified",
            "sf_qualify_prob": 1.0,
            "live_multiplier": 1.0,
            "probability_basis": "known_qualified",
        }
    if country in SF1_ACTUAL_NON_QUALIFIERS:
        return {
            "qualification_status": "SF1 eliminated",
            "sf_qualify_prob": 0.0,
            "live_multiplier": 0.0,
            "probability_basis": "known_eliminated",
        }

    semi = semi_by_country.get(country)
    if not semi:
        return {
            "qualification_status": "Automatic finalist",
            "sf_qualify_prob": 1.0,
            "live_multiplier": 1.0,
            "probability_basis": "automatic_finalist",
        }

    sf_num = int(semi.get("semi_final") or 0)
    sf_prob = semi.get("prob_qualify")
    if sf_num == 2 and sf_prob is not None:
        return {
            "qualification_status": "SF2 predicted Q" if bool(semi.get("predicted_qualifier")) else "SF2 pending",
            "sf_qualify_prob": float(sf_prob),
            "live_multiplier": float(sf_prob),
            "probability_basis": "base_gf_prob_times_sf2_qualify_prob",
        }
    return {
        "qualification_status": "Pending",
        "sf_qualify_prob": sf_prob,
        "live_multiplier": 1.0,
        "probability_basis": "base_gf_prob",
    }


def build_predictions() -> dict[str, Any]:
    confidence_meta = read_json(CONFIDENCE_META)
    surrogate = read_json(SURROGATE_JSON)
    narratives = read_json(NARRATIVES_JSON)
    semi_predictions = read_json(SEMI_PREDICTIONS_JSON) if SEMI_PREDICTIONS_JSON.exists() else {}

    xgb_rows = model_rows(CONFIDENCE_XGB)
    lgbm_rows = model_rows(CONFIDENCE_LGBM)
    xgb_by_country = indexed_by_country(xgb_rows)
    lgbm_by_country = indexed_by_country(lgbm_rows)
    surrogate_by_country = indexed_by_country(surrogate.get("countries", []))
    narrative_by_country = indexed_by_country(narratives.get("countries", []))
    semi_by_country = indexed_by_country(semi_predictions.get("countries", []))

    countries: list[dict[str, Any]] = []
    for country in sorted(set(xgb_by_country) | set(lgbm_by_country)):
        xgb = xgb_by_country.get(country, {})
        lgbm = lgbm_by_country.get(country, {})
        surrogate_row = surrogate_by_country.get(country, {})
        narrative_row = narrative_by_country.get(country, {})
        xgb_prob = xgb.get("prob_mean")
        lgbm_prob = lgbm.get("prob_mean")
        probs = [prob for prob in [xgb_prob, lgbm_prob] if prob is not None]
        base_prob = sum(probs) / len(probs) if probs else None
        qual = qualification_context(country, semi_by_country)
        live_prob = None if base_prob is None else base_prob * float(qual["live_multiplier"])
        countries.append(
            {
                "country": country,
                "base_consensus_prob": base_prob,
                "consensus_prob": live_prob,
                "live_consensus_prob": live_prob,
                **qual,
                "xgb_rank": xgb.get("rank"),
                "xgb_prob": xgb_prob,
                "xgb_ci80_lo": xgb.get("ci80_lo"),
                "xgb_ci80_hi": xgb.get("ci80_hi"),
                "xgb_ci50_lo": xgb.get("ci50_lo"),
                "xgb_ci50_hi": xgb.get("ci50_hi"),
                "lgbm_rank": lgbm.get("rank"),
                "lgbm_prob": lgbm_prob,
                "lgbm_ci80_lo": lgbm.get("ci80_lo"),
                "lgbm_ci80_hi": lgbm.get("ci80_hi"),
                "lgbm_ci50_lo": lgbm.get("ci50_lo"),
                "lgbm_ci50_hi": lgbm.get("ci50_hi"),
                "in_xgb_top10": bool(xgb.get("rank") and xgb["rank"] <= 10),
                "in_lgbm_top10": bool(lgbm.get("rank") and lgbm["rank"] <= 10),
                "surrogate_rank": surrogate_row.get("surrogate_rank"),
                "surrogate_prob": surrogate_row.get("surrogate_prob"),
                "narrative_prob": narrative_row.get("probability"),
                "narrative_prediction": narrative_row.get("prediction"),
            }
        )

    countries = sorted(countries, key=lambda row: row["consensus_prob"] or 0, reverse=True)
    for index, row in enumerate(countries, start=1):
        row["consensus_rank"] = index
        row["in_consensus_top10"] = index <= 10

    consensus_top10 = [
        {
            "rank": row["consensus_rank"],
            "country": row["country"],
            "consensus_prob": row["consensus_prob"],
            "xgb_rank": row["xgb_rank"],
            "lgbm_rank": row["lgbm_rank"],
        }
        for row in countries[:10]
    ]

    return {
        "story": "US-S7-01",
        "generated_at": datetime.now(UTC).isoformat(),
        "target_year": confidence_meta.get("target_year", 2026),
        "live_adjustment": {
            "method": "post_sf1_unconditional_top10",
            "note": (
                "Base Grand Final Top-10 probabilities are adjusted for known qualification status: "
                "SF1 eliminated countries are set to 0, known finalists keep base probability, "
                "and unresolved SF2 countries are multiplied by calibrated semi-final qualification probability."
            ),
        },
        "source_files": {
            "confidence_meta": str(CONFIDENCE_META.relative_to(PROJECT_ROOT)),
            "xgb": str(CONFIDENCE_XGB.relative_to(PROJECT_ROOT)),
            "lgbm": str(CONFIDENCE_LGBM.relative_to(PROJECT_ROOT)),
            "surrogate": str(SURROGATE_JSON.relative_to(PROJECT_ROOT)),
            "narratives": str(NARRATIVES_JSON.relative_to(PROJECT_ROOT)),
            "semi_predictions": str(SEMI_PREDICTIONS_JSON.relative_to(PROJECT_ROOT)),
        },
        "n_countries": len(countries),
        "models": {
            "xgb": {
                "ranked_by": "prob_mean",
                "n_countries": len(xgb_rows),
                "countries": xgb_rows,
            },
            "lgbm": {
                "ranked_by": "prob_mean",
                "n_countries": len(lgbm_rows),
                "countries": lgbm_rows,
            },
        },
        "consensus_top10": consensus_top10,
        "countries": countries,
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_predictions()
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
    print(f"Wrote {OUTPUT_JSON.relative_to(PROJECT_ROOT)} with {payload['n_countries']} countries")


if __name__ == "__main__":
    main()
