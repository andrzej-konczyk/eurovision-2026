"""
US-S5-03 — SHAP narratives: plain-language prediction cards per country.

For each entry in *target_year*, converts SHAP top-5 feature drivers into a
2–4 sentence human-readable card explaining the model's prediction.

Artefacts produced:
    reports/narratives_YYYY.json   — structured JSON (one object per country,
                                     sorted by descending probability)
    reports/narratives_YYYY.md     — formatted Markdown summary

Prerequisites:
    models/artefacts/{model}_model.pkl     — run src.models.train first
    models/artefacts/shap_top5_{model}.csv — run src.models.shap_pipeline first

CLI:
    python -m src.models.narratives [--target-year YEAR] [--model lgbm|xgb]
                                     [--data PATH] [--reports-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.models.shap_pipeline import impute, load_pipeline
from src.models.train import (
    ARTEFACT_DIR,
    ENRICHED_CSV,
    EXPERIMENT_NAME,
    FEATURE_COLS,
    MLFLOW_URI,
    ROOT,
    build_feature_matrix,
)

load_dotenv(ROOT / ".env")

TARGET_YEAR = 2026
REPORTS_DIR = ROOT / "reports"
DEFAULT_MODEL = "lgbm"

# Features that are constant across all entries in a given year (year-era flags,
# semi-final group assignment).  Their SHAP values explain the era-level base
# rate, not what differentiates one country from another — skip in narratives.
_SKIP_IN_NARRATIVE: frozenset[str] = frozenset({
    "rule_2019_semifinal_reform",
    "rule_2023_jury_weight_reform",
    "Semi_Final_Num",
})

# ---------------------------------------------------------------------------
# Feature → human-readable phrase  (positive direction, negative direction)
# ---------------------------------------------------------------------------

_PHRASES: dict[str, tuple[str, str]] = {
    "implied_prob_close": (
        "strong backing from the betting markets",
        "limited backing from the betting markets",
    ),
    "avg_final_rank_3yr": (
        "a strong 3-year Grand Final track record",
        "a weak recent Grand Final history",
    ),
    "avg_jury_3yr": (
        "consistent jury-vote accumulation in recent contests",
        "limited jury support in recent contests",
    ),
    "avg_tele_3yr": (
        "strong public televote support in recent years",
        "limited public televote support in recent years",
    ),
    "avg_bloc_jury_3yr": (
        "solid jury-vote support from the regional voting bloc",
        "limited jury support from the regional voting bloc",
    ),
    "avg_bloc_tele_3yr": (
        "strong televote solidarity from the regional voting bloc",
        "limited televote support from the regional voting bloc",
    ),
    "Running_Order_Final": (
        "a favourable Grand Final running-order position",
        "a challenging Grand Final running-order position",
    ),
    "Running_Order_Semi": (
        "a favourable semi-final running-order position",
        "a challenging semi-final running-order position",
    ),
    "zscore_myesb_community": (
        "high pre-contest fan community enthusiasm (MyESB community score)",
        "modest pre-contest fan community enthusiasm (MyESB community score)",
    ),
    "zscore_myesb_personal": (
        "strong personal fan ratings on MyESB",
        "modest personal fan ratings on MyESB",
    ),
    "zscore_ogae_points": (
        "strong support from the OGAE fan-jury network",
        "limited support from the OGAE fan-jury network",
    ),
    "Big6_Ind": (
        "automatic Grand Final qualification (Big Five / host)",
        "a competitive semi-final route to the Grand Final",
    ),
    "National_Final": (
        "selection through a competitive national final",
        "internal broadcaster selection",
    ),
    "Solo_Artist": (
        "a solo performance format",
        "a group performance format",
    ),
    "Returning_Artist_Ind": (
        "prior Eurovision experience (returning artist)",
        "a Eurovision debut",
    ),
    "Number of Members": (
        "an ensemble performance format",
        "a solo or duo format",
    ),
    "Multiple_Language": (
        "a multi-language entry",
        "a single-language entry",
    ),
    "EU": (
        "EU membership (historically correlated with stronger jury-bloc cohesion)",
        "non-EU status",
    ),
    "NATO": (
        "NATO membership",
        "non-NATO status",
    ),
    "Qualification_Record": (
        "a strong historical semi-final qualification rate",
        "a modest historical semi-final qualification rate",
    ),
    "Semi_Final_Num": (
        "semi-final group assignment",
        "semi-final group assignment",
    ),
    "rule_2019_semifinal_reform": (
        "the current post-2019 semi-final allocation system",
        "the pre-2019 semi-final format context",
    ),
    "rule_2023_jury_weight_reform": (
        "the current post-2023 jury/televote weighting",
        "the pre-2023 jury-weighting context",
    ),
}

_DEFAULT_PHRASE = ("a model-specific structural signal", "a model-specific structural signal")


# ---------------------------------------------------------------------------
# Phrase helper
# ---------------------------------------------------------------------------


def _phrase(feature: str, shap_val: float) -> str:
    """Return the direction-appropriate human phrase for *feature*."""
    pos, neg = _PHRASES.get(feature, _DEFAULT_PHRASE)
    return pos if shap_val >= 0 else neg


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------


def build_narrative(country: str, probability: float, top5: pd.DataFrame) -> str:
    """Build a 2–4 sentence prediction card.

    *top5* must contain columns ``feature`` and ``shap_value``, one row per
    feature for *country* (already filtered).  Sentences are space-joined.
    """
    # --- sentence 1: overall prediction ---
    if probability >= 0.65:
        label = f"a strong Top-10 contender (model probability: {probability:.0%})"
    elif probability >= 0.40:
        label = f"a borderline Top-10 contender (model probability: {probability:.0%})"
    else:
        label = f"unlikely to reach the Top 10 at current form (model probability: {probability:.0%})"
    s1 = f"{country} is modelled as {label}."

    # --- split positive / negative drivers (exclude structural constants) ---
    THRESHOLD = 0.005
    drivers = top5[~top5["feature"].isin(_SKIP_IN_NARRATIVE)]
    pos = drivers[drivers["shap_value"] > THRESHOLD].sort_values("shap_value", ascending=False)
    neg = drivers[drivers["shap_value"] < -THRESHOLD].sort_values("shap_value")

    # --- sentence 2: positive drivers ---
    if len(pos) >= 2:
        p1 = _phrase(pos.iloc[0]["feature"], 1)
        p2 = _phrase(pos.iloc[1]["feature"], 1)
        s2 = f"Key positive signals are {p1} and {p2}."
    elif len(pos) == 1:
        p1 = _phrase(pos.iloc[0]["feature"], 1)
        s2 = f"The primary positive signal is {p1}."
    else:
        s2 = "The model finds no strong positive drivers among the top-5 features."

    # --- sentence 3: negative factors (omit when none) ---
    if len(neg) >= 2:
        n1 = _phrase(neg.iloc[0]["feature"], -1)
        n2 = _phrase(neg.iloc[1]["feature"], -1)
        s3: str | None = f"Limiting factors include {n1} and {n2}."
    elif len(neg) == 1:
        n1 = _phrase(neg.iloc[0]["feature"], -1)
        s3 = f"The main limiting factor is {n1}."
    else:
        s3 = None

    # --- sentence 4: balance line (only when both sides are present) ---
    if pos.shape[0] > 0 and neg.shape[0] > 0:
        s4: str | None = (
            f"Net signal among top-5 drivers: "
            f"{pos.shape[0]} positive, {neg.shape[0]} negative."
        )
    else:
        s4 = None

    sentences = [s1, s2]
    if s3:
        sentences.append(s3)
    if s4:
        sentences.append(s4)
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_shap_top5(model_name: str, artefact_dir: Path = ARTEFACT_DIR) -> pd.DataFrame:
    path = artefact_dir / f"shap_top5_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP artefact not found: {path}\n"
            f"Run first:  python -m src.models.shap_pipeline --target-year {TARGET_YEAR}"
        )
    return pd.read_csv(path, encoding="utf-8")


def _get_probabilities(
    model_name: str,
    data_path: Path,
    target_year: int,
    artefact_dir: Path,
) -> pd.DataFrame:
    """Return DataFrame with columns [country, probability] for *target_year*."""
    predictions_path = REPORTS_DIR / f"predictions_{target_year}.json"
    if predictions_path.exists():
        try:
            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            rows = payload.get("countries", [])
            if rows:
                records = []
                for row in rows:
                    country = row.get("country")
                    probability = row.get("consensus_prob", row.get("ensemble_prob"))
                    if country is not None and probability is not None:
                        records.append({
                            "country": country,
                            "probability": float(probability),
                        })
                if records:
                    result = pd.DataFrame(records)
                    result.attrs["source_model"] = payload.get(
                        "model",
                        "consensus_xgb_lgbm_post_polymarket",
                    )
                    return result
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    matrix = build_feature_matrix(df)
    feat_cols = [c for c in FEATURE_COLS if c in matrix.columns]

    target_mask = matrix["Year"] == target_year
    if not target_mask.any():
        raise ValueError(f"No rows found for Year={target_year} in {data_path}.")

    X_target = matrix.loc[target_mask, feat_cols].reset_index(drop=True)
    countries = matrix.loc[target_mask, "Country"].reset_index(drop=True)

    pipeline = load_pipeline(model_name, artefact_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_imp = impute(pipeline, X_target)
        proba = pipeline.predict_proba(X_target)[:, 1]

    result = pd.DataFrame({"country": countries, "probability": proba})
    result.attrs["source_model"] = model_name
    return result


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(
    cards: list[dict],
    target_year: int,
    model_name: str,
    generated_at: str,
) -> str:
    lines: list[str] = [
        f"# Eurovision {target_year} — Model Prediction Narratives",
        f"",
        f"*Generated: {generated_at[:10]} | Model: {model_name.upper()} | "
        f"Target year: {target_year}*",
        f"",
        f"Countries ranked by model probability (Top-10 placement).",
        f"",
        f"---",
        f"",
    ]

    for i, card in enumerate(cards, start=1):
        country = card["country"]
        pct = f"{card['probability']:.0%}"
        lines.append(f"### {i}. {country} — {pct}")
        lines.append(f"")
        lines.append(card["narrative"])
        lines.append(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_narratives(
    data_path: Path = ENRICHED_CSV,
    target_year: int = TARGET_YEAR,
    model_name: str = DEFAULT_MODEL,
    out_dir: Path = ARTEFACT_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> list[dict]:
    """Generate narrative cards for every *target_year* entry.

    Returns a list of card dicts sorted by descending probability.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nNarratives  target_year={target_year}  model={model_name.upper()}")

    shap_df = _load_shap_top5(model_name, out_dir)
    prob_df = _get_probabilities(model_name, data_path, target_year, out_dir)
    probability_model = prob_df.attrs.get("source_model", model_name)

    countries = prob_df["country"].tolist()
    print(f"Countries   : {len(countries)}")

    cards: list[dict] = []
    for _, row in prob_df.iterrows():
        country = row["country"]
        prob = float(row["probability"])

        top5 = shap_df[shap_df["country"] == country].copy()
        if top5.empty:
            narrative = (
                f"{country} is modelled at {prob:.0%} probability. "
                f"No SHAP breakdown available for this entry."
            )
            pos_drivers: list[dict] = []
            neg_drivers: list[dict] = []
        else:
            top5 = top5.sort_values("rank")
            narrative = build_narrative(country, prob, top5)
            pos_drivers = [
                {"feature": r["feature"], "shap_value": round(r["shap_value"], 4)}
                for _, r in top5[top5["shap_value"] > 0.005]
                .sort_values("shap_value", ascending=False)
                .iterrows()
            ]
            neg_drivers = [
                {"feature": r["feature"], "shap_value": round(r["shap_value"], 4)}
                for _, r in top5[top5["shap_value"] < -0.005]
                .sort_values("shap_value")
                .iterrows()
            ]

        cards.append({
            "country": country,
            "probability": round(prob, 4),
            "prediction": "top10" if prob >= 0.5 else "outside_top10",
            "narrative": narrative,
            "positive_drivers": pos_drivers,
            "negative_drivers": neg_drivers,
        })

    cards.sort(key=lambda c: -c["probability"])

    timestamp = datetime.now(timezone.utc).isoformat()
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"narratives_{target_year}.json"
    payload = {
        "story": "US-S5-03",
        "generated_at": timestamp,
        "target_year": target_year,
        "model": probability_model,
        "n_countries": len(cards),
        "countries": cards,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _label = json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path
    print(f"  Saved: {_label}")

    md_path = reports_dir / f"narratives_{target_year}.md"
    md_text = _render_markdown(cards, target_year, probability_model, timestamp)
    md_path.write_text(md_text.rstrip() + "\n", encoding="utf-8")
    _md_label = md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path
    print(f"  Saved: {_md_label}")

    # quick preview
    print(f"\nTop-5 predicted countries:")
    for c in cards[:5]:
        print(f"  {c['country']:<20} {c['probability']:.0%}  {c['narrative'][:80]}…")

    # MLflow
    with mlflow.start_run(run_name=f"narratives-{target_year}-{timestamp[:10]}"):
        mlflow.log_params({
            "target_year": target_year,
            "model": model_name,
            "n_countries": len(cards),
        })
        mlflow.log_metric("n_top10_predicted", sum(1 for c in cards if c["prediction"] == "top10"))
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(md_path))
        mlflow.set_tag("story", "US-S5-03")

    print("\nDone.")
    return cards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP narrative cards (US-S5-03)")
    parser.add_argument("--data",        type=Path, default=ENRICHED_CSV)
    parser.add_argument("--target-year", type=int,  default=TARGET_YEAR)
    parser.add_argument("--model",       type=str,  default=DEFAULT_MODEL,
                        choices=["lgbm", "xgb"], help="Model whose SHAP values to use")
    parser.add_argument("--out-dir",     type=Path, default=ARTEFACT_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    generate_narratives(
        data_path=args.data,
        target_year=args.target_year,
        model_name=args.model,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
