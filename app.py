"""Streamlit MVP dashboard for Eurovision 2026 prediction artifacts."""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = APP_ROOT / "reports"

PREDICTIONS_JSON = REPORTS_DIR / "predictions_2026.json"
NARRATIVES_JSON = REPORTS_DIR / "narratives_2026.json"
BACKTEST_JSON_CANDIDATES = [
    REPORTS_DIR / "backtest_2022_2025.json",
    REPORTS_DIR / "backtest_2022_2024.json",
]
SEMI_BACKTEST_JSON_CANDIDATES = [
    REPORTS_DIR / "backtest_semi_2022_2025.json",
    REPORTS_DIR / "backtest_semi_2022_2024.json",
]
SEMI_PREDICTIONS_JSON = REPORTS_DIR / "semi_predictions_2026.json"
VOTING_NETWORK_JSON_CANDIDATES = [
    REPORTS_DIR / "voting_network.json",
    REPORTS_DIR / "voting_network_2026.json",
]
ENRICHED_CSV = APP_ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
BLOC_COOCCURRENCE_CSV = APP_ROOT / "data" / "features" / "bloc_cooccurrence.csv"
GRAND_FINAL_AT_ISO = "2026-05-16T21:00:00+02:00"
AUDIO_FILE = APP_ROOT / "dl.mp3"
LOGO_FILE = APP_ROOT / "Eurovision_Song_Contest_2026_Logo.jpg"

NAVIGATION_PAGES = [
    "Overview",
    "Model Stats",
    "Semi Qualifiers",
    "Main Ranking",
    "Podium",
    "Narratives",
    "Voting Blocs",
    "Voting Network",
]

PAGE_CAPTIONS = {
    "Overview": "Current forecast snapshot, model freshness, leading contenders, and backtest health.",
    "Model Stats": "",
    "Main Ranking": "Full country ranking with confidence intervals and model agreement signals.",
    "Podium": "Top-3 position probabilities and winner concentration.",
    "Semi Qualifiers": "Semi-final qualification probabilities by draw.",
    "Voting Blocs": "Regional bloc membership matrix used by feature engineering.",
    "Voting Network": "Historical jury affinity graph merged with current top-10 probabilities.",
    "Narratives": "Country-level drivers and model-facing explanation notes.",
    "Backtest": "Historical validation metrics by year and model.",
    "Data Health": "Loaded artifacts, runtime status, and source availability.",
}

ABOUT_MODEL = (
    "Two gradient-boosted classifiers (XGBoost and LightGBM) were trained on Eurovision "
    "results from 2016 to 2025. Each country is scored on historical placement, regional "
    "voting blocs, pre-contest betting odds, running order, and social signals. Both models "
    "produce a top-10 probability; the consensus shown here averages them. A 1,000-run "
    "bootstrap generates the CI-80 confidence intervals. See the **Glossary** in the sidebar "
    "for definitions of all key terms."
)

TOP10_RATIONALE = (
    "The dashboard is optimized for **Grand Final Top-10 probability**, not for a single winner "
    "pick or an exact podium forecast. Top-10 is the most stable target available before the live "
    "shows: it gives the model enough positive examples in historical data, maps directly to a "
    "clear contest outcome, and backtests more reliably than predicting only first place.\n\n"
    "A Top-1 forecast is highly volatile because the winner depends on late-running effects, "
    "jury-televote splits, staging execution, and the final-night vote distribution. A Top-3 "
    "view is useful, but it is derived from the Top-10 model and should be read as a relative "
    "favourites signal. The Top-10 target is therefore the primary accuracy target; Top-3 and winner views "
    "are secondary interpretations built on top of it."
)

SEMI_QUALIFIER_METHOD = (
    "Semi-final predictions are calculated separately from the Grand Final ranking. The pipeline "
    "trains XGBoost and LightGBM binary classifiers on historical semi-final rows from 2016-2025, "
    "using only features that are available before the semi-final result is known: qualification "
    "record, semi-final draw, running order, historical jury/televote strength, bloc signals, "
    "social/community scores, rule-era flags, and semi-final market probability where available.\n\n"
    "For each 2026 semi-finalist, both models estimate **prob_qualify**. The displayed probability "
    "is the average of the XGBoost and LightGBM estimates. A 1,000-run bootstrap retrains the "
    "models on resampled historical data to produce the CI-80 interval, so a wider interval means "
    "the qualification estimate is less stable.\n\n"
    "Countries are then ranked within their own semi-final by prob_qualify. Because Eurovision "
    "qualifies 10 acts from each semi-final, the top 10 ranked countries in SF1 and the top 10 "
    "ranked countries in SF2 are marked as predicted qualifiers. This is why the table focuses "
    "on a Top-10 cutoff rather than a Top-1 or Top-3 cutoff: qualification is a threshold problem, "
    "not a winner-ranking problem."
)

GLOSSARY = {
    "prob_top10": "The model's estimate (0–100 %) that a country will place in the top 10 of the Grand Final.",
    "CI-80": "Confidence Interval at 80 %: the range the model is 80 % sure the true probability falls within. Short bar = high certainty; long bar = more uncertainty.",
    "SAFE": "The model is confident this country reaches the top 10 — high probability and a tight confidence interval.",
    "LIKELY": "A credible top-10 contender with more uncertainty. Could go either way on the night.",
    "UNCERTAIN": "Outside the model's expected top 10, or data is inconclusive.",
    "SHAP": "A technique that explains *why* the model scored a country the way it did — which features pushed the prediction up or down.",
    "Positive drivers": "Country-level features with positive SHAP values. They pushed the Top-10 probability upward for the selected country.",
    "Negative drivers": "Country-level features with negative SHAP values. They pulled the Top-10 probability downward for the selected country.",
    "SHAP value": "The size and direction of a feature's contribution. Larger absolute value = stronger effect on the country prediction.",
    "implied_prob_close": "Closing betting-market implied probability before the Grand Final. Higher value usually raises the Top-10 estimate.",
    "odds_vs_history_delta": "Difference between market expectation and recent historical strength. Large gaps can flag overperformance or underperformance versus history.",
    "implied_prob_semi": "Semi-final market implied probability used for semi-final qualification modelling.",
    "avg_bloc_tele_3yr": "Recent average televote support from countries in the same historical voting bloc.",
    "avg_bloc_jury_3yr": "Recent average jury support from countries in the same historical voting bloc.",
    "avg_tele_3yr": "Country's recent average televote strength over the available three-year window.",
    "avg_jury_3yr": "Country's recent average jury strength over the available three-year window.",
    "avg_final_rank_3yr": "Recent Grand Final placement history. Better historical ranks generally support the prediction.",
    "Qualification_Record": "Recent record of qualifying from semi-finals where applicable.",
    "Running_Order_Final": "Grand Final performance slot. Later slots have historically correlated with stronger results.",
    "Running_Order_Semi": "Semi-final performance slot used in qualification modelling.",
    "rule_2019_semifinal_reform": "Indicator for contests under the current post-2019 semi-final structure.",
    "rule_2023_jury_weight_reform": "Indicator for contests after the 2023 voting-rule change.",
    "zscore_myesb_community": "Standardized My Eurovision Scoreboard community signal.",
    "zscore_myesb_personal": "Standardized My Eurovision Scoreboard personal-list signal.",
    "zscore_ogae_points": "Standardized OGAE poll signal where available.",
    "Big6_Ind": "Automatic Grand Final qualification indicator for Big 5 countries plus the host.",
    "National_Final": "Whether the entry was selected through a national final rather than internally.",
    "Solo_Artist": "Whether the act is performed by a solo artist.",
    "Returning_Artist_Ind": "Whether the performer has appeared at Eurovision before.",
    "Number of Members": "Number of performers in the act.",
    "Multiple_Language": "Whether the song uses more than one language.",
    "EU": "European Union membership flag used as a contextual feature.",
    "NATO": "NATO membership flag used as a contextual feature.",
    "Running order": "The broadcast slot a country performs in. Later positions historically correlate with higher jury and televote scores.",
    "Implied probability": "Probability derived from bookmaker odds. It reflects market consensus and is the model's strongest single input.",
    "Voting bloc": "A group of countries that historically award each other high marks, often due to cultural or geographic ties.",
    "Backtest": "Testing the model on past Eurovision editions (2022–2025) to check how accurate it would have been in hindsight.",
}

COUNTRY_ISO2 = {
    "Albania": "AL",
    "Armenia": "AM",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Georgia": "GE",
    "Germany": "DE",
    "Greece": "GR",
    "Israel": "IL",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Moldova": "MD",
    "Montenegro": "ME",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "San Marino": "SM",
    "Serbia": "RS",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Ukraine": "UA",
    "United Kingdom": "GB",
}


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime_ns: int) -> dict[str, Any]:
    """Load a JSON artifact from disk and cache it across Streamlit reruns."""
    del mtime_ns
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {artifact_path}")
    return data


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime_ns: int) -> pd.DataFrame:
    """Load a CSV artifact from disk and cache it across Streamlit reruns."""
    del mtime_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_audio_data_uri(path: str, mtime_ns: int) -> str:
    del mtime_ns
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_image_data_uri(path: str, mtime_ns: int) -> str:
    del mtime_ns
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_dashboard_data() -> dict[str, Any]:
    """Load all dashboard JSON artifacts."""
    backtest_path = first_existing_path(BACKTEST_JSON_CANDIDATES)
    semi_backtest_path = first_existing_path(SEMI_BACKTEST_JSON_CANDIDATES)
    voting_network_path = first_existing_path(VOTING_NETWORK_JSON_CANDIDATES)
    return {
        "predictions": load_json(str(PREDICTIONS_JSON), PREDICTIONS_JSON.stat().st_mtime_ns),
        "predictions_path": str(PREDICTIONS_JSON.relative_to(APP_ROOT)),
        "narratives": load_json(str(NARRATIVES_JSON), NARRATIVES_JSON.stat().st_mtime_ns),
        "narratives_path": str(NARRATIVES_JSON.relative_to(APP_ROOT)),
        "backtest": load_json(str(backtest_path), backtest_path.stat().st_mtime_ns),
        "backtest_path": str(backtest_path.relative_to(APP_ROOT)),
        "semi_backtest": load_json(str(semi_backtest_path), semi_backtest_path.stat().st_mtime_ns),
        "semi_backtest_path": str(semi_backtest_path.relative_to(APP_ROOT)),
        "semi_predictions": load_json(str(SEMI_PREDICTIONS_JSON), SEMI_PREDICTIONS_JSON.stat().st_mtime_ns),
        "semi_predictions_path": str(SEMI_PREDICTIONS_JSON.relative_to(APP_ROOT)),
        "voting_network": load_json(str(voting_network_path), voting_network_path.stat().st_mtime_ns),
        "voting_network_path": str(voting_network_path.relative_to(APP_ROOT)),
        "history": load_csv(str(ENRICHED_CSV), ENRICHED_CSV.stat().st_mtime_ns),
        "history_path": str(ENRICHED_CSV.relative_to(APP_ROOT)),
        "bloc_cooccurrence": load_csv(
            str(BLOC_COOCCURRENCE_CSV),
            BLOC_COOCCURRENCE_CSV.stat().st_mtime_ns,
        ),
        "bloc_cooccurrence_path": str(BLOC_COOCCURRENCE_CSV.relative_to(APP_ROOT)),
    }


def format_prediction_update_timestamp(value: Any) -> str:
    if not value:
        return "n/a"
    if not isinstance(value, str):
        return str(value)
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if timestamp.tzinfo is None:
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.astimezone(UTC).strftime("%Y-%m-%d")


def inject_dashboard_style() -> None:
    css = """
<style>
/* ===== Eurovision 2026 Vienna — Dashboard Theme ===== */
:root {
    --esc-magenta: #E6007E;
    --esc-blue: #1A1464;
    --esc-gold: #F5C542;
    --esc-purple: #7B5EA7;
}

div[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    border-right: 3px solid var(--esc-magenta);
    background: linear-gradient(180deg, #1a0a2e 0%, #0d0d1a 100%) !important;
    box-shadow: 10px 0 36px rgba(5,5,20,0.42);
    color: #e2e8f0 !important;
}

.stApp {
    background:
        radial-gradient(circle at 18% 12%, rgba(56,119,255,0.28) 0%, transparent 30%),
        radial-gradient(circle at 82% 20%, rgba(123,94,167,0.18) 0%, transparent 30%),
        linear-gradient(145deg, #06113d 0%, #0b1f68 45%, #14368d 72%, #071034 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 1.35rem;
    background: rgba(255,255,255,0.998);
    border-left: 1px solid rgba(80,96,190,0.30);
    border-right: 1px solid rgba(96,16,128,0.24);
    box-shadow: 0 0 46px rgba(0,0,54,0.42);
}

div[data-testid="stTable"],
div[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border: 1px solid rgba(123,94,167,0.30);
    border-radius: 10px;
    padding: 0.4rem;
    box-shadow: 0 2px 8px rgba(26,20,100,0.08);
}

div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrame"] [class*="gdg-"],
div[data-testid="stDataFrame"] .glideDataEditor {
    background: #1a0a2e !important;
    border-radius: 10px;
}

.dark-dataframe-wrapper {
    background: #1a0a2e;
    padding: 6px 6px 2px;
    border-radius: 10px;
    border: 1px solid #9b4dca;
    margin-bottom: 0.6rem;
}

div[data-testid="stExpander"] {
    background: #1a0a2e;
    border: 1px solid #9b4dca;
    border-radius: 10px;
}

div[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
    color: rgba(248,245,255,0.88);
}

.sidebar-brand {
    border: 1px solid rgba(162,190,255,0.42);
    border-radius: 14px;
    padding: 1rem 0.9rem 0.85rem;
    margin-bottom: 0.85rem;
    background:
        linear-gradient(135deg, rgba(56,119,255,0.18), rgba(123,94,167,0.12)),
        rgba(255,255,255,0.075);
    box-shadow: 0 0 24px rgba(56,119,255,0.16);
}

.sidebar-brand-title {
    font-size: 1.25rem;
    line-height: 1.08;
    font-weight: 950;
    letter-spacing: 0.04em;
    color: #f8fbff;
    text-shadow: 0 0 18px rgba(162,190,255,0.46);
    background: none;
    -webkit-background-clip: text;
    -webkit-text-fill-color: currentColor;
}

.sidebar-brand-sub {
    margin-top: 0.42rem;
    color: rgba(226,236,255,0.92);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

div[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.35rem;
}

div[data-testid="stSidebar"] label[data-baseweb="radio"] {
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 0.35rem 0.48rem;
    background: rgba(255,255,255,0.075);
    transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
}

div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: rgba(56,119,255,0.20);
    border-color: rgba(162,190,255,0.52);
    transform: translateX(2px);
}

div[data-testid="stSidebar"] label[data-baseweb="radio"] p {
    font-weight: 800;
    letter-spacing: 0.01em;
}

div[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(245,197,66,0.20);
}

div[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.12);
    border-color: rgba(245,197,66,0.42);
}

div[data-testid="stMetric"],
div[data-testid="metric-container"] {
    background: #1a0a2e;
    border: 1px solid #9b4dca;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    color: #f8fafc;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] p {
    font-weight: 700 !important;
    color: #F5C542 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.09em !important;
}
div[data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-weight: 800;
    color: #ffffff !important;
}

h1 {
    font-weight: 900;
    letter-spacing: -0.01em;
}
h2 {
    color: var(--esc-blue);
    padding-bottom: 0.4rem;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #E6007E 0%, #F5C542 60%, transparent 100%) 1;
    font-weight: 900;
    letter-spacing: 0.01em;
}
h3 {
    color: #B8860B;
    font-weight: 800;
    padding-left: 0.55rem;
    border-left: 3px solid var(--esc-magenta);
    background: linear-gradient(90deg, rgba(245,197,66,0.10) 0%, transparent 60%);
}

div[data-testid="stCaptionContainer"] p {
    color: var(--esc-purple);
    font-style: italic;
}

div[data-testid="stExpander"] summary p {
    font-weight: 600;
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
    color: #e2e8f0;
}

div[data-testid="stDataFrame"] div[role="gridcell"],
div[data-testid="stDataFrame"] div[role="columnheader"],
div[data-testid="stDataFrame"] [role="cell"],
div[data-testid="stDataFrame"] [role="columnheader"] *,
div[data-testid="stTable"] td,
div[data-testid="stTable"] th {
    justify-content: center !important;
    text-align: center !important;
}

div[data-testid="stDataFrame"] canvas,
div[data-testid="stDataFrame"] .glide-cell,
div[data-testid="stDataFrame"] .gdg-cell,
div[data-testid="stDataFrame"] .gdg-header {
    text-align: center !important;
}

div[data-testid="stDataFrame"] [data-testid="stMarkdownContainer"] p {
    text-align: center !important;
}

div[data-testid="stDataFrame"] input {
    text-align: center !important;
}

hr {
    border-color: rgba(230,0,126,0.25);
}

.badge-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.22rem 0.78rem;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: transform 180ms ease, box-shadow 180ms ease;
    cursor: default;
}
.badge-pill:hover {
    transform: translateY(-1px) scale(1.04);
    filter: brightness(1.08);
}

.badge-pill.safe {
    background: #F5C542;
    color: #2d1f00;
    box-shadow: 0 0 0.7rem rgba(245,197,66,0.75), 0 0 1.35rem rgba(245,197,66,0.35);
    animation: safeGlow 1.8s ease-in-out infinite;
}

.badge-pill.likely {
    background: #E6007E;
    color: white;
}

.badge-pill.uncertain {
    background: #7B5EA7;
    color: white;
}

@keyframes safeGlow {
    0%, 100% { box-shadow: 0 0 0.55rem rgba(245,197,66,0.55), 0 0 1rem rgba(245,197,66,0.28); transform: translateY(0); }
    50% { box-shadow: 0 0 1rem rgba(245,197,66,0.95), 0 0 2rem rgba(245,197,66,0.48); transform: translateY(-1px); }
}

.country-hero {
    display: flex;
    align-items: center;
    gap: 1.05rem;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0 1rem;
    background:
        radial-gradient(ellipse at 92% 50%, rgba(230,0,126,0.18) 0%, transparent 65%),
        radial-gradient(ellipse at 8% 50%, rgba(245,197,66,0.14) 0%, transparent 60%),
        linear-gradient(135deg, #0D0A3B 0%, #1A1464 55%, #3A0F5E 100%);
    border: 1px solid rgba(245,197,66,0.55);
    border-radius: 14px;
    box-shadow: 0 10px 26px rgba(13,10,59,0.32), 0 0 24px rgba(245,197,66,0.12) inset;
    position: relative;
    overflow: hidden;
}
.country-hero .hero-flag {
    object-fit: cover;
    border-radius: 5px;
    border: 2px solid rgba(245,197,66,0.7);
    box-shadow: 0 0 14px rgba(245,197,66,0.4), 0 4px 10px rgba(0,0,0,0.35);
    flex-shrink: 0;
    position: relative;
    z-index: 1;
}
.country-hero .hero-content {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex: 1;
    min-width: 0;
    position: relative;
    z-index: 1;
}
.country-hero .hero-name {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 0.09em;
    line-height: 1.05;
    background: linear-gradient(90deg, #F5C542 0%, #FFE58A 35%, #FFFFFF 50%, #FFE58A 65%, #F5C542 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: heroShimmer 4.5s linear infinite;
    text-shadow: 0 0 28px rgba(245,197,66,0.18);
}
.country-hero .hero-prob {
    position: relative;
    height: 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(245,197,66,0.35);
    overflow: hidden;
}
.country-hero .hero-prob-fill {
    position: absolute;
    inset: 0 auto 0 0;
    background: linear-gradient(90deg, #E6007E 0%, #F5C542 100%);
    border-radius: 999px;
    box-shadow: 0 0 12px rgba(245,197,66,0.45);
    transition: width 600ms ease-out;
}
.country-hero .hero-prob-label {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fffaf0;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-shadow: 0 1px 3px rgba(0,0,0,0.6);
    pointer-events: none;
}
.country-hero .hero-deco {
    position: absolute;
    color: rgba(245,197,66,0.7);
    pointer-events: none;
    animation: heroTwinkle 2.6s ease-in-out infinite;
    z-index: 0;
}
.country-hero .hero-star-1 { top: 14%; right: 6%; font-size: 1.05rem; animation-delay: 0s; }
.country-hero .hero-star-2 { bottom: 18%; right: 18%; font-size: 0.7rem; animation-delay: 1.1s; }
@keyframes heroShimmer {
    0%   { background-position: -250% center; }
    100% { background-position:  250% center; }
}
@keyframes heroTwinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.7); }
    50%      { opacity: 1;    transform: scale(1.2); }
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    border-left: 4px solid var(--esc-magenta) !important;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        padding-top: 0.75rem !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    .country-hero {
        flex-direction: column;
        align-items: flex-start;
        padding: 0.75rem 0.9rem;
        gap: 0.6rem;
    }
    .country-hero .hero-name {
        font-size: 1.35rem !important;
    }
    .badge-pill {
        font-size: 0.72rem !important;
        padding: 0.18rem 0.52rem !important;
    }
    div[data-testid="stMetric"],
    div[data-testid="metric-container"] {
        padding: 0.6rem 0.7rem !important;
    }
    div[data-testid="stPlotlyChart"] {
        overflow-x: auto !important;
    }
    div[data-testid="column"] {
        min-width: 0;
        overflow: hidden;
    }
}
</style>
"""
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def _info_expander(title: str, body: str) -> None:
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(body)


def badge_pill_html(badge: str) -> str:
    css_class = str(badge).lower()
    if css_class not in {"safe", "likely", "uncertain"}:
        css_class = "uncertain"
    return f'<span class="badge-pill {css_class}">{escape(str(badge))}</span>'


def render_countdown_timer() -> None:
    components.html(
        f"""
<div class="countdown-card">
  <div class="countdown-label">Grand Final countdown</div>
  <div id="gf-countdown" class="countdown-value">Loading...</div>
  <div class="countdown-date">16 May 2026, 21:00 CEST · Wiener Stadthalle</div>
</div>
<style>
  .countdown-card {{
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin: 0.35rem 0 0.85rem;
    background: linear-gradient(90deg, #2d0a4e 0%, #4a1a6b 70%, #5a3a00 100%);
    border: 1px solid rgba(245,197,66,0.58);
    color: #fffaf0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    box-shadow: 0 12px 28px rgba(26,20,100,0.24);
  }}
  .countdown-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em; opacity: 0.82; font-weight: 800; }}
  .countdown-value {{ font-size: clamp(1.45rem, 4vw, 2.25rem); line-height: 1.15; font-weight: 950; margin-top: 0.2rem; text-shadow: 0 0 18px rgba(245,197,66,0.45); }}
  .countdown-date {{ margin-top: 0.28rem; font-size: 0.82rem; opacity: 0.84; }}
</style>
<script>
  const target = new Date("{GRAND_FINAL_AT_ISO}").getTime();
  const node = document.getElementById("gf-countdown");
  function tick() {{
    const diff = target - Date.now();
    if (diff <= 0) {{ node.textContent = "Grand Final is live"; return; }}
    const minute = 60 * 1000, hour = 60 * minute, day = 24 * hour;
    const days = Math.floor(diff / day);
    const hours = Math.floor((diff % day) / hour);
    const minutes = Math.floor((diff % hour) / minute);
    node.textContent = `${{days}} days, ${{hours}}h, ${{minutes}}min to Grand Final`;
  }}
  tick();
  setInterval(tick, 30000);
</script>
""",
        height=118,
        scrolling=False,
    )


def render_audio_player(enabled: bool) -> None:
    if not enabled:
        components.html(
            """
<script>
  const doc = window.parent?.document || document;
  doc.getElementById("esc-audio-dock")?.remove();
</script>
""",
            height=1,
            scrolling=False,
        )
        return
    if not AUDIO_FILE.exists():
        return
    audio_src = load_audio_data_uri(str(AUDIO_FILE), AUDIO_FILE.stat().st_mtime_ns)
    html = """
<script>
  const doc = window.parent?.document || document;
  let dock = doc.getElementById("esc-audio-dock");
  if (!dock) {
    dock = doc.createElement("div");
    dock.id = "esc-audio-dock";
    dock.innerHTML = `
      <audio id="esc-audio" src="__AUDIO_SRC__" loop autoplay></audio>
      <button id="music-toggle" type="button">🎵 ON</button>
    `;
    doc.body.appendChild(dock);
    const style = doc.createElement("style");
    style.id = "esc-audio-style";
    style.textContent = `
      #esc-audio-dock {
        position: fixed;
        top: 4.65rem;
        right: 1.1rem;
        z-index: 999998;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.36rem 0.52rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(245,197,66,0.58);
        box-shadow: 0 8px 22px rgba(26,20,100,0.22);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      #music-toggle {
        border: 1px solid rgba(245,197,66,0.86);
        border-radius: 999px;
        background: linear-gradient(135deg, #1A1464 0%, #7B5EA7 52%, #E6007E 100%);
        color: #fffaf0;
        padding: 0.42rem 0.78rem;
        font-weight: 900;
        cursor: pointer;
        box-shadow: 0 0 16px rgba(245,197,66,0.35);
      }
      @media (max-width: 640px) {
        #esc-audio-dock {
          top: auto !important;
          bottom: 1.2rem !important;
          right: 0.75rem !important;
        }
      }
    `;
    doc.head.appendChild(style);
  }
  const audio = doc.getElementById("esc-audio");
  const btn = doc.getElementById("music-toggle");
  const enabledKey = "escMusicEnabled";
  const timeKey = "escMusicTime";
  if (!audio.src) audio.src = "__AUDIO_SRC__";
  audio.volume = 0.42;

  function enabled() {
    return localStorage.getItem(enabledKey) !== "false";
  }
  function setUi(isOn, blocked=false) {
    btn.textContent = blocked ? "🎵 PLAY" : (isOn ? "🎵 OFF" : "🎵 ON");
  }
  function restoreTime() {
    const saved = Number(localStorage.getItem(timeKey) || "0");
    if (Number.isFinite(saved) && saved > 0) {
      audio.currentTime = saved % Math.max(audio.duration || saved + 1, 1);
    }
  }
  async function start() {
    restoreTime();
    try {
      await audio.play();
      setUi(true);
    } catch (err) {
      setUi(false, true);
    }
  }
  function stop() {
    audio.pause();
    localStorage.setItem(timeKey, String(audio.currentTime || 0));
    setUi(false);
  }
  if (!btn.dataset.bound) {
    btn.dataset.bound = "1";
    btn.addEventListener("click", async () => {
    if (audio.paused) {
      localStorage.setItem(enabledKey, "true");
      await start();
    } else {
      localStorage.setItem(enabledKey, "false");
      stop();
    }
    });
  }
  if (!audio.dataset.bound) {
    audio.dataset.bound = "1";
    audio.addEventListener("timeupdate", () => localStorage.setItem(timeKey, String(audio.currentTime || 0)));
  }
  if (enabled()) start();
  else setUi(false);
</script>
""".replace("__AUDIO_SRC__", audio_src)
    components.html(
        html,
        height=1,
        scrolling=False,
    )


def render_back_to_top() -> None:
    components.html(
        """
<script>
  const doc = window.parent?.document || document;
  const view = window.parent || window;
  let btn = doc.getElementById("esc-back-to-top");
  if (!btn) {
    btn = doc.createElement("button");
    btn.id = "esc-back-to-top";
    btn.type = "button";
    btn.title = "Back to top";
    btn.textContent = "↑";
    doc.body.appendChild(btn);
    const style = doc.createElement("style");
    style.id = "esc-back-to-top-style";
    style.textContent = `
      #esc-back-to-top {
        position: fixed;
        bottom: 1.4rem;
        right: 1.4rem;
        z-index: 999997;
        width: 2.7rem;
        height: 2.7rem;
        border-radius: 999px;
        border: 1px solid rgba(245,197,66,0.7);
        background: linear-gradient(135deg, #1A1464 0%, #7B5EA7 50%, #E6007E 100%);
        color: #fffaf0;
        font-size: 1.25rem;
        font-weight: 900;
        cursor: pointer;
        opacity: 0;
        transform: translateY(12px);
        transition: opacity 220ms ease, transform 220ms ease, box-shadow 220ms ease;
        box-shadow: 0 8px 22px rgba(13,10,59,0.32);
        pointer-events: none;
      }
      #esc-back-to-top.visible {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
      }
      #esc-back-to-top:hover {
        box-shadow: 0 0 18px rgba(245,197,66,0.55), 0 8px 22px rgba(13,10,59,0.4);
        transform: translateY(-2px);
      }
    `;
    doc.head.appendChild(style);
    btn.addEventListener("click", () => {
      const target = view.document.scrollingElement || doc.scrollingElement || doc.documentElement;
      target.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
  if (!view.__escBackToTopBound) {
    view.__escBackToTopBound = true;
    const onScroll = () => {
      const y = view.scrollY || view.document.scrollingElement.scrollTop || 0;
      if (y > 400) btn.classList.add("visible");
      else btn.classList.remove("visible");
    };
    view.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
</script>
""",
        height=1,
        scrolling=False,
    )


def render_tab_confetti(page: str) -> None:
    del page
    components.html(
        """
<canvas id="confetti-canvas"></canvas>
<style>
  #confetti-canvas { position: fixed; inset: 0; pointer-events: none; width: 100vw; height: 100vh; z-index: 999999; }
</style>
<script>
  const doc = window.parent?.document || document;
  const view = window.parent || window;
  let canvas = doc.getElementById("esc-global-confetti");
  if (!canvas) {
    canvas = doc.createElement("canvas");
    canvas.id = "esc-global-confetti";
    canvas.style.position = "fixed";
    canvas.style.inset = "0";
    canvas.style.pointerEvents = "none";
    canvas.style.width = "100vw";
    canvas.style.height = "100vh";
    canvas.style.zIndex = "999999";
    doc.body.appendChild(canvas);
  }
  const ctx = canvas.getContext("2d");
  let pieces = [];
  function resize() { canvas.width = view.innerWidth; canvas.height = view.innerHeight; }
  function burst() {
    resize();
    const colors = ["#F5C542", "#C0C7D1", "#CD7F32", "#E6007E", "#7B5EA7", "#ffffff"];
    pieces = Array.from({length: 115}, () => ({
      x: canvas.width / 2, y: canvas.height * 0.22,
      vx: (Math.random() - 0.5) * 10, vy: Math.random() * -7 - 2,
      size: Math.random() * 7 + 4, rot: Math.random() * Math.PI,
      color: colors[Math.floor(Math.random() * colors.length)], life: 90 + Math.random() * 30
    }));
    requestAnimationFrame(frame);
  }
  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pieces.forEach((p) => {
      p.x += p.vx; p.y += p.vy; p.vy += 0.18; p.rot += 0.12; p.life -= 1;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot);
      ctx.fillStyle = p.color; ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.58);
      ctx.restore();
    });
    pieces = pieces.filter((p) => p.life > 0 && p.y < canvas.height + 40);
    if (pieces.length) requestAnimationFrame(frame);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  setTimeout(burst, 180);
  if (!doc.body.dataset.escConfettiBound) {
    doc.body.dataset.escConfettiBound = "1";
    doc.addEventListener("click", (event) => {
      const target = event.target;
      const navClick = target?.closest?.('div[role="radiogroup"], label[data-baseweb="radio"], input[type="radio"]');
      if (navClick) setTimeout(burst, 120);
    }, true);
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if ([...mutation.addedNodes].some((node) => node.nodeType === 1 && node.querySelector?.('section.main'))) {
          setTimeout(burst, 220);
          break;
        }
      }
    });
    observer.observe(doc.body, { childList: true, subtree: true });
  }
  view.addEventListener("resize", resize);
</script>
""",
        height=1,
        scrolling=False,
    )


def render_safety_badge_summary(ranking: pd.DataFrame) -> None:
    if ranking.empty or "badge" not in ranking.columns:
        return
    counts = ranking["badge"].value_counts().to_dict()
    html = " ".join(
        f'{badge_pill_html(badge)} <strong>{int(counts.get(badge, 0))}</strong>'
        for badge in ["SAFE", "LIKELY", "UNCERTAIN"]
    )
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(page: str, title: str | None = None) -> None:
    display_title = title or page
    caption = PAGE_CAPTIONS.get(page)
    caption_html = f'<div class="sub">{escape(caption)}</div>' if caption else ""
    header_html = f"""
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: transparent; overflow: hidden; }}
.banner {{
  background: linear-gradient(135deg, #0D0A3B 0%, #1A1464 55%, #3A0F5E 100%);
  border-radius: 14px;
  border: 1px solid rgba(230,0,126,0.5);
  padding: 1.05rem 1.75rem 0.95rem;
  position: relative;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.banner::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 15% 50%, rgba(230,0,126,0.18) 0%, transparent 65%),
              radial-gradient(ellipse at 85% 50%, rgba(245,197,66,0.12) 0%, transparent 65%);
  pointer-events: none;
}}
@keyframes shimmer {{
  0%   {{ background-position: -250% center; }}
  100% {{ background-position:  250% center; }}
}}
@keyframes twinkle {{
  0%, 100% {{ opacity: 0.2; transform: scale(0.7); }}
  50%       {{ opacity: 1;   transform: scale(1.2); }}
}}
.title {{
  font-size: 1.75rem; font-weight: 900; letter-spacing: 0.04em; line-height: 1.15;
  background: linear-gradient(90deg, #E6007E 0%, #F5C542 30%, #E8E4FF 50%, #E6007E 65%, #F5C542 100%);
  background-size: 250% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: shimmer 4s linear infinite;
  position: relative; z-index: 1;
}}
.sub {{
  color: rgba(220,215,255,0.68); font-size: 0.76rem; margin-top: 0.42rem;
  letter-spacing: 0.09em; text-transform: uppercase;
  position: relative; z-index: 1;
}}
.star {{
  position: absolute; color: rgba(245,197,66,0.75);
  animation: twinkle ease-in-out infinite; z-index: 1; pointer-events: none;
}}
</style>
<div class="banner">
  <span class="star" style="font-size:1.1rem;top:18%;left:4%;animation-duration:2.1s;animation-delay:0s">★</span>
  <span class="star" style="font-size:0.7rem;top:65%;left:11%;animation-duration:1.7s;animation-delay:0.4s">✦</span>
  <span class="star" style="font-size:0.9rem;top:20%;right:8%;animation-duration:2.4s;animation-delay:1.0s">★</span>
  <span class="star" style="font-size:1.2rem;top:58%;right:15%;animation-duration:1.9s;animation-delay:0.7s">★</span>
  <span class="star" style="font-size:0.65rem;top:75%;right:5%;animation-duration:2.8s;animation-delay:1.3s">✦</span>
  <div class="title">{escape(display_title)}</div>
  {caption_html}
</div>
"""
    components.html(header_html, height=112 if caption else 90, scrolling=False)


def dashboard_artifact_rows(data: dict[str, Any], load_time_s: float) -> list[dict[str, Any]]:
    return [
        {"check": "Predictions JSON", "path": data["predictions_path"], "loaded": bool(data["predictions"])},
        {"check": "Semi predictions JSON", "path": data["semi_predictions_path"], "loaded": bool(data["semi_predictions"])},
        {"check": "Voting network JSON", "path": data["voting_network_path"], "loaded": bool(data["voting_network"])},
        {
            "check": "Voting bloc co-occurrence CSV",
            "path": data["bloc_cooccurrence_path"],
            "loaded": not data["bloc_cooccurrence"].empty,
        },
        {"check": "Narratives JSON", "path": data["narratives_path"], "loaded": bool(data["narratives"])},
        {"check": "Backtest JSON", "path": data["backtest_path"], "loaded": bool(data["backtest"])},
        {"check": "Semi backtest JSON", "path": data["semi_backtest_path"], "loaded": bool(data["semi_backtest"])},
        {"check": "History CSV", "path": data["history_path"], "loaded": not data["history"].empty},
        {"check": "Load time < 2s", "path": "runtime", "loaded": load_time_s < 2.0},
    ]


def countries_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    countries = predictions.get("countries", [])
    if not countries:
        return pd.DataFrame()

    frame = pd.DataFrame(countries)
    if "consensus_prob" in frame.columns:
        frame = frame.rename(columns={"consensus_prob": "probability"})
    if "consensus_rank" in frame.columns:
        frame = frame.rename(columns={"consensus_rank": "rank"})
    if "ensemble_prob" in frame.columns:
        frame = frame.rename(columns={"ensemble_prob": "probability"})
    if "ensemble_rank" in frame.columns:
        frame = frame.rename(columns={"ensemble_rank": "rank"})
    if "probability" in frame.columns:
        frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    if "rank" not in frame.columns and "probability" in frame.columns:
        frame = frame.sort_values("probability", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", frame.index + 1)
    return frame


def backtest_frame(backtest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, year_data in backtest.get("years", {}).items():
        for model, metrics in year_data.get("models", {}).items():
            rows.append(
                {
                    "year": int(year),
                    "model": model.upper(),
                    "top10_accuracy": metrics.get("top10_accuracy"),
                    "top10_hits": metrics.get("top10_hits"),
                    "ci80_coverage": metrics.get("ci80_empirical_coverage"),
                    "top10_kpi": metrics.get("kpi_top10_pass"),
                    "ci80_kpi": metrics.get("kpi_ci80_pass"),
                }
            )
    return pd.DataFrame(rows)


def semi_backtest_frame(semi_backtest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, year_data in semi_backtest.get("years", {}).items():
        for model, metrics in year_data.get("models", {}).items():
            rows.append(
                {
                    "year": int(year),
                    "model": model.upper(),
                    "qual_accuracy_overall": metrics.get("qual_accuracy_overall"),
                    "qual_accuracy_sf1": metrics.get("qual_accuracy_sf1"),
                    "qual_accuracy_sf2": metrics.get("qual_accuracy_sf2"),
                    "ci80_coverage": metrics.get("ci80_empirical_coverage"),
                    "sf1_kpi": metrics.get("kpi_sf1_pass"),
                    "sf2_kpi": metrics.get("kpi_sf2_pass"),
                    "ci80_kpi": metrics.get("kpi_ci80_pass"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_model_stats(backtest: dict[str, Any], semi_backtest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, metrics in backtest.get("aggregate", {}).items():
        rows.append(
            {
                "scope": "Grand Final Top-10",
                "model": model.upper(),
                "accuracy": metrics.get("avg_top10_accuracy"),
                "ci80_coverage": metrics.get("avg_ci80_empirical_coverage"),
                "accuracy_kpi_pass": metrics.get("all_top10_kpi_pass"),
                "ci80_kpi_pass": metrics.get("all_ci80_kpi_pass"),
            }
        )
    for model, metrics in semi_backtest.get("aggregate", {}).items():
        rows.append(
            {
                "scope": "Semi-final qualification",
                "model": model.upper(),
                "accuracy": metrics.get("avg_qual_accuracy_overall"),
                "ci80_coverage": metrics.get("avg_ci80_empirical_coverage"),
                "accuracy_kpi_pass": (
                    bool(metrics.get("all_sf1_kpi_pass"))
                    and bool(metrics.get("all_sf2_kpi_pass"))
                ),
                "ci80_kpi_pass": metrics.get("all_ci80_kpi_pass"),
            }
        )
    return pd.DataFrame(rows)


def best_metric_value(frame: pd.DataFrame, scope: str, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame.loc[frame["scope"] == scope, column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def format_percent(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.1%}"


def format_percent_rounded(value: Any) -> str:
    numeric = safe_float(value)
    return "n/a" if numeric is None else f"{numeric:.0%}"


def model_stats_note(backtest: dict[str, Any], semi_backtest: dict[str, Any]) -> str:
    gf_years = backtest.get("backtest_years", [])
    semi_years = semi_backtest.get("backtest_years", [])
    gf_label = f"{min(gf_years)}-{max(gf_years)}" if gf_years else "n/a"
    semi_label = f"{min(semi_years)}-{max(semi_years)}" if semi_years else "n/a"
    return (
        f"Grand Final backtest: {gf_label}; semi-final backtest: {semi_label}. "
        "Accuracy is measured against historical holdout years using strict temporal isolation."
    )


def render_sidebar(data: dict[str, Any], load_time_s: float) -> tuple[str, bool]:
    del data, load_time_s
    st.sidebar.markdown(
        """
<div class="sidebar-brand">
  <div class="sidebar-brand-title">Eurovision<br>2026</div>
  <div class="sidebar-brand-sub">Vienna forecast suite</div>
</div>
""",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        NAVIGATION_PAGES,
    )
    caption = PAGE_CAPTIONS.get(page, "")
    if caption:
        st.sidebar.markdown(f"<small>{escape(caption)}</small>", unsafe_allow_html=True)
    music_enabled = st.sidebar.checkbox("Music", value=False)
    with st.sidebar.expander("Glossary", expanded=False):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** - {definition}")
    return page, music_enabled


def country_flag(country: str) -> str:
    code = COUNTRY_ISO2.get(country)
    if not code:
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in code.upper())


def country_flag_url(country: str) -> str:
    code = COUNTRY_ISO2.get(country)
    if not code:
        return ""
    return f"https://flagcdn.com/w40/{code.lower()}.png"


def country_flag_img(country: str, width: int = 28) -> str:
    url = country_flag_url(country)
    if not url:
        return ""
    label = escape(country)
    height = max(1, round(width * 0.75))
    return (
        f'<img src="{url}" alt="{label} flag" width="{width}" height="{height}" '
        'style="vertical-align:-0.18rem;border-radius:2px;object-fit:cover;'
        'border:1.5px solid rgba(155,77,202,0.55);box-shadow:0 0 4px rgba(245,197,66,0.25);">'
    )


def country_label(country: str) -> str:
    flag = country_flag(country)
    return f"{flag}  {country}" if flag else country


def country_hero_html(country: str, flag_width: int = 56, probability: float | None = None) -> str:
    url = country_flag_url(country)
    flag_height = max(1, round(flag_width * 0.75))
    flag_img = (
        f'<img class="hero-flag" src="{url}" alt="{escape(country)} flag" '
        f'width="{flag_width}" height="{flag_height}">'
        if url
        else ""
    )
    bar_html = ""
    if probability is not None and 0.0 <= probability <= 1.0:
        pct = probability * 100.0
        bar_html = (
            '<div class="hero-prob">'
            f'<div class="hero-prob-fill" style="width:{pct:.1f}%;"></div>'
            f'<span class="hero-prob-label">Top-10 probability · {pct:.1f}%</span>'
            "</div>"
        )
    return (
        '<div class="country-hero">'
        '<span class="hero-deco hero-star-1">★</span>'
        '<span class="hero-deco hero-star-2">✦</span>'
        f'{flag_img}'
        '<div class="hero-content">'
        f'<span class="hero-name">{escape(country).upper()}</span>'
        f'{bar_html}'
        '</div>'
        "</div>"
    )


def narratives_by_country(narratives: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("country")): row
        for row in narratives.get("countries", [])
        if row.get("country")
    }


def country_prediction_row(predictions_df: pd.DataFrame, country: str) -> dict[str, Any]:
    rows = predictions_df[predictions_df["country"] == country]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def country_history_frame(history: pd.DataFrame, country: str) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    frame = history[
        (history["Country"] == country)
        & (history["Year"].between(2016, 2025))
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["Year", "Result", "Final_Place", "Final_Points", "Semi_Place"])

    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce").astype("Int64")
    frame["Final_Place"] = pd.to_numeric(frame["Final_Place"], errors="coerce")
    frame["Final_Points"] = pd.to_numeric(frame["Final_Points"], errors="coerce")
    frame["Semi_Place"] = pd.to_numeric(frame["Semi_Place"], errors="coerce")
    frame["Grand_Final_Ind"] = pd.to_numeric(frame["Grand_Final_Ind"], errors="coerce")
    frame["Result"] = frame.apply(history_result_label, axis=1)
    return frame[["Year", "Result", "Final_Place", "Final_Points", "Semi_Place"]].sort_values("Year")


def history_result_label(row: pd.Series) -> str:
    final_place = safe_float(row.get("Final_Place"))
    grand_final = safe_float(row.get("Grand_Final_Ind"))
    semi_place = safe_float(row.get("Semi_Place"))
    if final_place is not None:
        return f"Final #{int(final_place)}"
    if grand_final == 0 and semi_place is not None:
        return f"Semi #{int(semi_place)}"
    if grand_final == 0:
        return "Semi"
    return "No entry"


def feature_importance_frame(narrative: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, sign in [("positive", 1), ("negative", -1)]:
        for row in narrative.get(f"{direction}_drivers", []):
            shap_value = safe_float(row.get("shap_value"))
            if shap_value is None:
                continue
            rows.append(
                {
                    "feature": str(row.get("feature", "")),
                    "shap_value": shap_value,
                    "direction": direction,
                    "signed_value": shap_value if sign > 0 else -abs(shap_value),
                    "abs_value": abs(shap_value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "shap_value", "direction", "signed_value", "abs_value"])
    return pd.DataFrame(rows).sort_values("abs_value", ascending=False).head(5)


_FEATURE_DISPLAY: dict[str, str] = {
    "implied_prob_close": "Market odds (implied prob.)",
    "odds_vs_history_delta": "Market vs history delta",
    "avg_final_rank_3yr": "GF avg rank (3yr)",
    "avg_jury_3yr": "Jury avg (3yr)",
    "avg_tele_3yr": "Televote avg (3yr)",
    "avg_bloc_jury_3yr": "Bloc jury avg (3yr)",
    "avg_bloc_tele_3yr": "Bloc televote avg (3yr)",
    "Running_Order_Final": "GF running order",
    "Running_Order_Semi": "Semi running order",
    "zscore_myesb_community": "Fan community score",
    "zscore_myesb_personal": "Fan personal score",
    "zscore_ogae_points": "OGAE fan jury",
    "Big6_Ind": "Big Five / host",
    "National_Final": "National final selection",
    "Solo_Artist": "Solo artist",
    "Returning_Artist_Ind": "Returning artist",
    "Qualification_Record": "Semi qualification rate",
    "Semi_Final_Num": "Semi-final group",
    "EU": "EU membership",
    "NATO": "NATO membership",
    "Multiple_Language": "Multi-language entry",
    "rule_2019_semifinal_reform": "Post-2019 SF reform",
    "rule_2023_jury_weight_reform": "Post-2023 jury reform",
}


def feature_bar_chart(features: pd.DataFrame) -> go.Figure:
    plot_frame = features.sort_values("abs_value", ascending=True).copy()
    plot_frame["feature"] = plot_frame["feature"].map(
        lambda f: _FEATURE_DISPLAY.get(f, f)
    )
    fig = go.Figure(
        go.Bar(
            x=plot_frame["signed_value"],
            y=plot_frame["feature"],
            orientation="h",
            marker_color=[
                "#16a34a" if value >= 0 else "#dc2626"
                for value in plot_frame["signed_value"]
            ],
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top-5 SHAP features",
        xaxis_title="SHAP value",
        yaxis_title=None,
        height=280,
        margin={"l": 150, "r": 30, "t": 55, "b": 40},
        font={"size": 12, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"gridcolor": "rgba(123,94,167,0.18)", "zerolinecolor": "rgba(123,94,167,0.55)"},
        yaxis={"gridcolor": "rgba(123,94,167,0.10)"},
    )
    fig.add_vline(x=0, line_width=1, line_color="#7B5EA7")
    return fig


def country_ci_frame(prediction: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model in ["xgb", "lgbm"]:
        prob = safe_float(prediction.get(f"{model}_prob"))
        if prob is None:
            continue
        rows.append(
            {
                "model": model.upper(),
                "probability": prob,
                "ci80_lo": safe_float(prediction.get(f"{model}_ci80_lo")),
                "ci80_hi": safe_float(prediction.get(f"{model}_ci80_hi")),
                "ci50_lo": safe_float(prediction.get(f"{model}_ci50_lo")),
                "ci50_hi": safe_float(prediction.get(f"{model}_ci50_hi")),
            }
        )
    probability = safe_float(prediction.get("probability"))
    if probability is not None and rows:
        rows.append(
            {
                "model": "CONSENSUS",
                "probability": probability,
                "ci80_lo": sum(row["ci80_lo"] for row in rows if row["ci80_lo"] is not None) / len(rows),
                "ci80_hi": sum(row["ci80_hi"] for row in rows if row["ci80_hi"] is not None) / len(rows),
                "ci50_lo": sum(row["ci50_lo"] for row in rows if row["ci50_lo"] is not None) / len(rows),
                "ci50_hi": sum(row["ci50_hi"] for row in rows if row["ci50_hi"] is not None) / len(rows),
            }
        )
    return pd.DataFrame(rows)


def ci_fan_chart(ci_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if ci_frame.empty:
        return fig
    plot_frame = ci_frame.iloc[::-1]
    for _, row in plot_frame.iterrows():
        model = row["model"]
        y = [model, model]
        fig.add_trace(
            go.Scatter(
                x=[row["ci80_lo"], row["ci80_hi"]],
                y=y,
                mode="lines",
                line={"color": "#93c5fd", "width": 18},
                name="CI-80",
                showlegend=model == plot_frame.iloc[0]["model"],
                hovertemplate=f"{model} CI-80: %{{x:.1%}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["ci50_lo"], row["ci50_hi"]],
                y=y,
                mode="lines",
                line={"color": "#2563eb", "width": 9},
                name="CI-50",
                showlegend=model == plot_frame.iloc[0]["model"],
                hovertemplate=f"{model} CI-50: %{{x:.1%}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["probability"]],
                y=[model],
                mode="markers",
                marker={"color": "#0f172a", "size": 9},
                name="Mean",
                showlegend=model == plot_frame.iloc[0]["model"],
                hovertemplate=f"{model} mean: %{{x:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="CI-80 / CI-50 fan chart",
        xaxis={"title": "Top-10 probability", "tickformat": ".0%", "range": [0, 1], "gridcolor": "rgba(123,94,167,0.18)"},
        yaxis={"gridcolor": "rgba(123,94,167,0.10)"},
        yaxis_title=None,
        height=260,
        margin={"l": 90, "r": 30, "t": 55, "b": 40},
        font={"size": 12, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def history_chart(history_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if history_frame.empty:
        return fig
    finals = history_frame[history_frame["Final_Place"].notna()]
    if not finals.empty:
        fig.add_trace(
            go.Scatter(
                x=finals["Year"],
                y=finals["Final_Place"],
                mode="lines+markers",
                line={"color": "#2563eb"},
                marker={"size": 8},
                hovertemplate="%{x}: final #%{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Final history 2016-2025",
        xaxis_title="Year",
        xaxis={"gridcolor": "rgba(123,94,167,0.18)"},
        yaxis={"title": "Final place", "autorange": "reversed", "dtick": 5, "gridcolor": "rgba(123,94,167,0.18)"},
        height=260,
        margin={"l": 60, "r": 30, "t": 55, "b": 40},
        font={"size": 12, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def country_card_data(
    country: str,
    predictions_df: pd.DataFrame,
    narratives: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, Any]:
    narrative = narratives_by_country(narratives).get(country, {})
    ranking = main_ranking_frame(predictions_df, n_places=len(predictions_df))
    prediction = country_prediction_row(ranking if not ranking.empty else predictions_df, country)
    return {
        "country": country,
        "flag": country_flag(country),
        "narrative": narrative,
        "prediction": prediction,
        "features": feature_importance_frame(narrative),
        "ci": country_ci_frame(prediction),
        "history": country_history_frame(history, country),
    }


def render_country_card(card: dict[str, Any]) -> None:
    country = card["country"]
    narrative = card["narrative"]
    prediction = card["prediction"]
    probability = safe_float(prediction.get("probability"))
    rank = prediction.get("rank")

    st.markdown(country_hero_html(country, probability=probability), unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Consensus rank", "n/a" if pd.isna(rank) else f"#{int(rank)}", help="Final ensemble rank position (1 = highest top-10 probability).")
    cols[1].metric("Top-10 probability", "n/a" if probability is None else f"{probability:.1%}", help="Probability this country finishes in the Grand Final top 10.")
    cols[2].metric("Narrative signal", narrative.get("prediction", "n/a"), help="High-level direction from the SHAP narrative summary.")
    badge = prediction.get("badge")
    if isinstance(badge, str) and badge:
        st.markdown(badge_pill_html(badge), unsafe_allow_html=True)

    text = str(narrative.get("narrative", "")).strip()
    st.write(text if text else "No narrative available.")

    left, right = st.columns(2)
    with left:
        features = card["features"]
        if features.empty:
            st.info("No SHAP feature drivers available.")
        else:
            st.plotly_chart(feature_bar_chart(features), use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(ci_fan_chart(card["ci"]), use_container_width=True, config={"displaylogo": False})
    with right:
        history_frame = card["history"]
        st.plotly_chart(history_chart(history_frame), use_container_width=True, config={"displaylogo": False})
        st.dataframe(history_frame, use_container_width=True, hide_index=True)


def render_country_detail_sidebar(
    data: dict[str, Any],
    predictions_df: pd.DataFrame,
) -> None:
    if predictions_df.empty:
        return
    countries = sorted(predictions_df["country"].tolist())
    selected = st.sidebar.selectbox(
        "Country detail",
        countries,
        format_func=country_label,
    )
    card = country_card_data(selected, predictions_df, data["narratives"], data["history"])
    with st.sidebar.expander(f"Focus country: {selected}", expanded=False):
        st.markdown(
            f"{country_flag_img(selected, width=30)} **{escape(selected)}**",
            unsafe_allow_html=True,
        )
        narrative = str(card["narrative"].get("narrative", "")).strip()
        probability = safe_float(card["prediction"].get("probability"))
        st.metric("Top-10 probability", "n/a" if probability is None else f"{probability:.1%}", help="Probability this country finishes in the Grand Final top 10.")
        st.write(narrative if narrative else "No narrative available.")


def overview_leaderboard_frame(predictions_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(columns=["rank", "country", "probability", "ci80_lo", "ci80_hi"])
    frame = main_ranking_frame(predictions_df, n_places=n)
    columns = ["rank", "country", "probability", "ci80_lo", "ci80_hi"]
    return frame[[column for column in columns if column in frame.columns]]


def render_overview(data: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    predictions = data["predictions"]

    render_page_header("Overview", title="Eurovision 2026 Forecast")
    render_countdown_timer()

    top_country = predictions_df.iloc[0]["country"] if not predictions_df.empty else "n/a"
    top_prob = predictions_df.iloc[0].get("probability") if not predictions_df.empty else None
    last_prediction_update = format_prediction_update_timestamp(predictions.get("generated_at"))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Target year", predictions.get("target_year", 2026), help="Eurovision contest year these predictions cover.")
    col2.metric("Countries", len(predictions_df), help="Number of countries with model predictions for the Grand Final.")
    col3.metric("Top country", top_country, help="Country with the highest consensus top-10 probability.")
    col4.metric("Top probability", f"{top_prob:.1%}" if pd.notna(top_prob) else "n/a", help="Highest predicted probability of finishing in the Grand Final top 10.")
    col5.metric("Prediction updated", last_prediction_update, help="When the prediction artefacts were last regenerated.")

    st.subheader("Leading Contenders")
    leaders = overview_leaderboard_frame(predictions_df)
    leaders_display = leaders.copy()
    for column in ["probability", "ci80_lo", "ci80_hi"]:
        if column in leaders_display.columns:
            leaders_display[column] = leaders_display[column] * 100.0
    st.markdown('<div class="dark-dataframe-wrapper">', unsafe_allow_html=True)
    st.dataframe(
        leaders_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Top-10 probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "ci80_lo": st.column_config.NumberColumn("CI-80 low", format="%.1f%%"),
            "ci80_hi": st.column_config.NumberColumn("CI-80 high", format="%.1f%%"),
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    _info_expander(
        "How does the model work?",
        ABOUT_MODEL,
    )
    _info_expander(
        "Why Top-10 — not the winner?",
        TOP10_RATIONALE,
    )


def model_predictions_frame(predictions: dict[str, Any], model: str) -> pd.DataFrame:
    rows = predictions.get("models", {}).get(model, {}).get("countries", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_predictions(
    predictions: dict[str, Any],
    predictions_df: pd.DataFrame,
    narratives: dict[str, Any],
    history: pd.DataFrame,
) -> None:
    render_page_header("Main Ranking")
    if predictions_df.empty:
        st.warning("No country predictions found in the predictions JSON.")
        return

    ranking = main_ranking_frame(predictions_df)
    if ranking.empty:
        st.warning("No country rankings found in the predictions JSON.")
        return

    fig = ranking_plot(ranking)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "eurovision_2026_ranking_all35",
                "height": 900,
                "width": 1400,
                "scale": 2,
            },
        },
    )
    render_safety_badge_summary(ranking)
    _info_expander(
        "How to read this chart",
        "**Bars** show each country's estimated probability of finishing in the Grand Final top 10.\n\n"
        "**CI-80 whiskers** are the 80 % Confidence Interval — the range where the model is 80 % sure "
        "the true probability lies. Short whisker = high certainty; long whisker = genuine uncertainty.\n\n"
        "**Bar colour = Safety badge:**\n"
        "- 🟡 **SAFE** (gold) — high probability ≥ 65 % and tight CI. Expect this country in the top 10.\n"
        "- 🩷 **LIKELY** (magenta) — a real contender but with more uncertainty. Could go either way.\n"
        "- 🔵 **UNCERTAIN** (purple) — outside the expected top 10, or data is inconclusive.",
    )

    visible_columns = [
        "rank",
        "country",
        "probability",
        "ci80_lo",
        "ci80_hi",
        "badge",
        "xgb_prob",
        "lgbm_prob",
        "model_consensus",
    ]

    st.dataframe(
        ranking[visible_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "prob_top10",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "badge": st.column_config.TextColumn("Safety badge"),
            "model_consensus": st.column_config.TextColumn("XGB vs LGBM consensus"),
            "xgb_prob": st.column_config.NumberColumn("XGB probability", format="%.3f"),
            "lgbm_prob": st.column_config.NumberColumn("LGBM probability", format="%.3f"),
            "ci80_lo": st.column_config.NumberColumn("CI-80 low", format="%.3f"),
            "ci80_hi": st.column_config.NumberColumn("CI-80 high", format="%.3f"),
        },
    )

    selected_country = st.selectbox("Open country card", sorted(ranking["country"].tolist()), format_func=country_label)
    with st.expander(f"📋 Country card: {selected_country}", expanded=False):
        render_country_card(
            country_card_data(
                selected_country,
                predictions_df,
                narratives,
                history,
            )
        )


def main_ranking_frame(predictions_df: pd.DataFrame, n_places: int | None = None) -> pd.DataFrame:
    frame = predictions_df.copy()
    numeric_columns = [
        "rank",
        "probability",
        "xgb_prob",
        "lgbm_prob",
        "xgb_rank",
        "lgbm_rank",
        "xgb_ci80_lo",
        "xgb_ci80_hi",
        "lgbm_ci80_lo",
        "lgbm_ci80_hi",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("rank").reset_index(drop=True)
    if n_places is not None:
        frame = frame.head(n_places).reset_index(drop=True)
    frame["ci80_lo"] = frame[["xgb_ci80_lo", "lgbm_ci80_lo"]].mean(axis=1)
    frame["ci80_hi"] = frame[["xgb_ci80_hi", "lgbm_ci80_hi"]].mean(axis=1)
    frame["ci80_width"] = frame["ci80_hi"] - frame["ci80_lo"]
    frame["badge"] = frame.apply(safety_badge, axis=1)
    frame["model_consensus"] = frame.apply(model_consensus_label, axis=1)
    return frame


def safety_badge(row: pd.Series) -> str:
    probability = safe_float(row.get("probability"))
    rank = safe_float(row.get("rank"))
    ci80_lo = safe_float(row.get("ci80_lo"))
    ci80_width = safe_float(row.get("ci80_width"))
    if probability is None or rank is None:
        return "UNCERTAIN"
    if rank <= 10 and probability >= 0.65 and ci80_lo is not None and ci80_lo >= 0.40:
        return "SAFE"
    if rank <= 13 or probability >= 0.45 or (ci80_width is not None and ci80_width >= 0.45):
        return "LIKELY"
    return "UNCERTAIN"


def model_consensus_label(row: pd.Series) -> str:
    in_xgb = bool(row.get("in_xgb_top10"))
    in_lgbm = bool(row.get("in_lgbm_top10"))
    if in_xgb and in_lgbm:
        return "XGB + LGBM top-10"
    if in_xgb:
        return "XGB only"
    if in_lgbm:
        return "LGBM only"
    return "Outside both top-10"


def ranking_plot(ranking: pd.DataFrame) -> go.Figure:
    plot_frame = ranking.sort_values("rank", ascending=False)
    colors = {"SAFE": "#F5C542", "LIKELY": "#E6007E", "UNCERTAIN": "#7B5EA7"}
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_frame["probability"],
            y=plot_frame["country"],
            orientation="h",
            marker_color=[colors.get(badge, "#64748b") for badge in plot_frame["badge"]],
            showlegend=False,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (plot_frame["ci80_hi"] - plot_frame["probability"]).clip(lower=0),
                "arrayminus": (plot_frame["probability"] - plot_frame["ci80_lo"]).clip(lower=0),
                "thickness": 1.2,
                "width": 3,
                "color": "#1A1464",
            },
            customdata=plot_frame[["rank", "badge", "xgb_prob", "lgbm_prob", "model_consensus"]],
            hovertemplate=(
                "<b>%{y}</b>  #%{customdata[0]}<br>"
                "Consensus: <b>%{x:.1%}</b>  [%{customdata[1]}]<br>"
                "XGB: %{customdata[2]:.1%}  ·  LGBM: %{customdata[3]:.1%}<br>"
                "%{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Ranking 1-35 by consensus prob_top10",
        xaxis_title="prob_top10 with CI-80",
        yaxis_title=None,
        height=820,
        margin={"l": 120, "r": 40, "t": 70, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.06,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        bargap=0.22,
        xaxis={"tickformat": ".0%", "range": [0, 1], "gridcolor": "rgba(123,94,167,0.18)", "tickfont": {"color": "#1A1464"}},
        yaxis={"tickfont": {"color": "#1A1464", "size": 13, "family": "Arial Black, Arial, sans-serif"}},
        font={"size": 13, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    _badge_labels = {
        "SAFE": "SAFE — confident top-10",
        "LIKELY": "LIKELY — probable top-10",
        "UNCERTAIN": "UNCERTAIN — outside top-10",
    }
    for badge, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 12, "color": color, "symbol": "square"},
                name=_badge_labels.get(badge, badge),
                hoverinfo="skip",
            )
        )
    return fig


def position_probability_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate top-3 position mass from model probability and bootstrap CI summaries."""
    frame = main_ranking_frame(predictions_df, n_places=len(predictions_df)).copy()
    if frame.empty:
        return pd.DataFrame(columns=["country", "position", "probability"])

    rows: list[dict[str, Any]] = []
    position_specs = {
        1: {"metric": (frame["probability"] + frame["ci80_hi"]) / 2.0, "rank_decay": 0.72},
        2: {"metric": frame["probability"], "rank_decay": 0.64},
        3: {"metric": (frame["probability"] + frame["ci80_lo"]) / 2.0, "rank_decay": 0.58},
    }
    for position, spec in position_specs.items():
        distance = (frame["rank"] - position).abs()
        scores = spec["metric"].clip(lower=0.0) * (spec["rank_decay"] ** distance)
        total = scores.sum()
        probabilities = scores / total if total > 0 else scores
        for country, probability in zip(frame["country"], probabilities, strict=True):
            rows.append(
                {
                    "country": country,
                    "position": f"P{position}",
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows)


def position_probability_matrix(position_df: pd.DataFrame) -> pd.DataFrame:
    if position_df.empty:
        return pd.DataFrame()
    matrix = position_df.pivot(index="country", columns="position", values="probability").fillna(0.0)
    return matrix[["P1", "P2", "P3"]]


def top3_heatmap(position_df: pd.DataFrame, predictions_df: pd.DataFrame) -> go.Figure:
    matrix = position_probability_matrix(position_df)
    country_order = (
        predictions_df[["country", "rank"]]
        .dropna()
        .sort_values("rank")["country"]
        .tolist()
    )
    matrix = matrix.reindex([country for country in country_order if country in matrix.index])
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=["1st", "2nd", "3rd"],
            y=matrix.index,
            colorscale="Viridis",
            zmin=0.0,
            zmax=max(0.01, float(matrix.to_numpy().max())),
            colorbar={"title": "Probability", "tickformat": ".0%"},
            hovertemplate=(
                "<b>%{y}</b> · %{x}<br>"
                "Probability: <b>%{z:.1%}</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Top-3 probability heatmap",
        xaxis_title="Position",
        xaxis={"tickfont": {"color": "#1A1464"}},
        yaxis={"title": None, "autorange": "reversed", "tickfont": {"color": "#1A1464", "family": "Arial Black, Arial, sans-serif"}},
        height=980,
        margin={"l": 130, "r": 40, "t": 70, "b": 50},
        font={"size": 13, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def winner_gauge_frame(position_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    winners = (
        position_df[position_df["position"] == "P1"]
        .sort_values("probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    winners = winners[["country", "probability"]].copy()
    winners.insert(0, "rank", winners.index + 1)
    return winners


def winner_gauge_figure(position_df: pd.DataFrame, top_n: int = 3) -> go.Figure:
    winners = winner_gauge_frame(position_df, top_n)
    fig = go.Figure()
    for index, row in winners.iterrows():
        domain_width = 1.0 / max(top_n, 1)
        x0 = index * domain_width + 0.02
        x1 = (index + 1) * domain_width - 0.02
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(row["probability"]),
                number={"valueformat": ".1%", "font": {"size": 22}},
                title={"text": f"#{int(row['rank'])} {row['country']}", "font": {"size": 15}},
                domain={"x": [x0, x1], "y": [0.08, 0.9]},
                gauge={
                    "axis": {
                        "range": [0.0, 1.0],
                        "tickmode": "array",
                        "tickvals": [0.0, 0.5, 1.0],
                        "ticktext": ["", "50%", ""],
                        "tickcolor": "#1A1464",
                    },
                    "bar": {"color": "#E6007E"},
                    "bgcolor": "#ffffff",
                    "bordercolor": "#7B5EA7",
                    "steps": [
                        {"range": [0.0, 0.2], "color": "#fee2e2"},
                        {"range": [0.2, 0.5], "color": "#fef3c7"},
                        {"range": [0.5, 1.0], "color": "#dcfce7"},
                    ],
                },
            )
        )
    fig.update_layout(
        title=f"Winner probability gauge: top {top_n}",
        height=300,
        margin={"l": 25, "r": 25, "t": 50, "b": 20},
        font={"size": 14, "color": "#1A1464"},
        paper_bgcolor="#ffffff",
    )
    return fig


def bloc_cooccurrence_long_frame(cooccurrence: pd.DataFrame) -> pd.DataFrame:
    if cooccurrence.empty:
        return pd.DataFrame(columns=["country", "bloc", "member"])
    frame = cooccurrence.copy()
    country_col = "Country" if "Country" in frame.columns else frame.columns[0]
    if country_col != "Country":
        frame = frame.rename(columns={country_col: "Country"})
    bloc_cols = [col for col in frame.columns if col != "Country"]
    long = frame.melt(
        id_vars="Country",
        value_vars=bloc_cols,
        var_name="bloc",
        value_name="member",
    )
    long["member"] = pd.to_numeric(long["member"], errors="coerce").fillna(0).astype(int)
    return long.rename(columns={"Country": "country"})


def voting_bloc_d3_html(cooccurrence: pd.DataFrame) -> str:
    long = bloc_cooccurrence_long_frame(cooccurrence)
    records = long.to_dict(orient="records")
    payload = json.dumps(records)
    return f"""
<div id="bloc-d3"></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const data = {payload};
const countries = Array.from(new Set(data.map(d => d.country))).sort();
const blocs = Array.from(new Set(data.map(d => d.bloc))).sort();
const margin = {{top: 96, right: 36, bottom: 48, left: 190}};
const cell = 24;
const width = margin.left + margin.right + blocs.length * 118;
const height = margin.top + margin.bottom + countries.length * cell;
const root = d3.select("#bloc-d3").html("");
const svg = root.append("svg")
  .attr("viewBox", [0, 0, width, height])
  .attr("width", "100%")
  .attr("height", height)
  .style("background", "#ffffff");
const x = d3.scaleBand().domain(blocs).range([margin.left, width - margin.right]).padding(0.08);
const y = d3.scaleBand().domain(countries).range([margin.top, height - margin.bottom]).padding(0.08);
svg.append("g")
  .selectAll("text")
  .data(blocs)
  .join("text")
  .attr("x", d => x(d) + 8)
  .attr("y", margin.top - 12)
  .attr("text-anchor", "start")
  .attr("transform", d => `rotate(-35, ${{x(d) + 8}}, ${{margin.top - 12}})`)
  .attr("font-size", 13)
  .attr("font-weight", 700)
  .attr("fill", "#0f172a")
  .text(d => d);
svg.append("g")
  .selectAll("text")
  .data(countries)
  .join("text")
  .attr("x", margin.left - 10)
  .attr("y", d => y(d) + y.bandwidth() / 2)
  .attr("dominant-baseline", "middle")
  .attr("text-anchor", "end")
  .attr("font-size", 13)
  .attr("font-weight", 600)
  .attr("fill", "#0f172a")
  .text(d => d);
svg.append("g")
  .selectAll("rect")
  .data(data)
  .join("rect")
  .attr("x", d => x(d.bloc))
  .attr("y", d => y(d.country))
  .attr("width", x.bandwidth())
  .attr("height", y.bandwidth())
  .attr("rx", 4)
  .attr("stroke", "#cbd5e1")
  .attr("stroke-width", 0.75)
  .attr("fill", d => d.member ? "#2563eb" : "#e5e7eb")
  .append("title")
  .text(d => `${{d.country}} / ${{d.bloc}}: ${{d.member ? "member" : "not member"}}`);
svg.append("text")
  .attr("x", margin.left)
  .attr("y", height - 12)
  .attr("font-size", 12)
  .attr("fill", "#475569")
  .text("Blue cells indicate bloc membership.");
</script>
"""


def voting_network_graph_data(
    voting_network: dict[str, Any],
    predictions_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Merge S5 voting-network topology with current top-10 probabilities."""
    probability_by_country = (
        predictions_df.set_index("country")["probability"].to_dict()
        if not predictions_df.empty and {"country", "probability"}.issubset(predictions_df.columns)
        else {}
    )
    links = [dict(link) for link in voting_network.get("links", [])]
    partner_weights: dict[str, list[tuple[str, int]]] = {}
    for link in links:
        source = str(link.get("source", ""))
        target = str(link.get("target", ""))
        weight = int(link.get("weight") or 0)
        partner_weights.setdefault(source, []).append((target, weight))
        partner_weights.setdefault(target, []).append((source, weight))

    nodes: list[dict[str, Any]] = []
    for node in voting_network.get("nodes", []):
        country = str(node.get("id", ""))
        partners = sorted(
            partner_weights.get(country, []),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        nodes.append({
            **node,
            "id": country,
            "probability": float(probability_by_country.get(country, node.get("probability") or 0.0)),
            "top_partners": [{"country": p, "weight": w} for p, w in partners],
        })
    return {"nodes": nodes, "links": links}


def voting_network_d3_html(voting_network: dict[str, Any], predictions_df: pd.DataFrame) -> str:
    graph = voting_network_graph_data(voting_network, predictions_df)
    nodes = graph["nodes"]
    links = graph["links"]
    payload = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)
    return f"""
<div id="voting-network-d3" data-node-count="{len(nodes)}" data-edge-count="{len(links)}"></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const graph = {payload};
const width = 1120;
const height = 760;
const padding = 34;
const root = d3.select("#voting-network-d3").html("");
const svg = root.append("svg")
  .attr("viewBox", [0, 0, width, height])
  .attr("width", "100%")
  .attr("height", height);
const tooltip = root.append("div")
  .style("position", "absolute")
  .style("pointer-events", "none")
  .style("opacity", 0)
  .style("background", "#111827")
  .style("color", "white")
  .style("padding", "0.45rem 0.6rem")
  .style("border-radius", "6px")
  .style("font", "12px system-ui, sans-serif");
const weights = graph.links.map(d => d.weight || 1);
const probs = graph.nodes.map(d => d.probability || 0);
const stroke = d3.scaleLinear().domain([d3.min(weights), d3.max(weights)]).range([1.2, 6]);
const radius = d3.scaleSqrt().domain([0, d3.max(probs) || 1]).range([6, 24]);
const color = d3.scaleOrdinal()
  .domain(Array.from(new Set(graph.nodes.map(d => d.group || "Other"))).sort())
  .range(d3.schemeTableau10);
const simulation = d3.forceSimulation(graph.nodes)
  .force("link", d3.forceLink(graph.links).id(d => d.id).distance(d => 96 - (d.weight || 1) * 5).strength(0.35))
  .force("charge", d3.forceManyBody().strength(-150))
  .force("center", d3.forceCenter(width / 2, height / 2).strength(0.18))
  .force("x", d3.forceX(width / 2).strength(0.035))
  .force("y", d3.forceY(height / 2).strength(0.035))
  .force("collide", d3.forceCollide(d => radius(d.probability || 0) + 8).strength(0.9));
const link = svg.append("g")
  .attr("stroke", "#94a3b8")
  .attr("stroke-opacity", 0.6)
  .selectAll("line")
  .data(graph.links)
  .join("line")
  .attr("stroke-width", d => stroke(d.weight || 1));
const linkLines = svg.selectAll("line")
  .on("mouseover", (event, d) => {{
    tooltip.style("opacity", 1)
      .html(`<b>${{d.source.id || d.source}}</b> ↔ <b>${{d.target.id || d.target}}</b><br>Historical affinity: <b>${{d.weight}}</b>`);
  }})
  .on("mousemove", event => {{
    tooltip.style("left", `${{event.offsetX + 14}}px`).style("top", `${{event.offsetY + 14}}px`);
  }})
  .on("mouseout", () => tooltip.style("opacity", 0));
const node = svg.append("g")
  .selectAll("circle")
  .data(graph.nodes)
  .join("circle")
  .attr("r", d => radius(d.probability || 0))
  .attr("fill", d => color(d.group || "Other"))
  .attr("stroke", "#0f172a")
  .attr("stroke-width", 1)
  .call(drag(simulation))
  .on("mouseover", (event, d) => {{
    const partners = (d.top_partners || [])
      .map(p => `${{p.country}} (${{p.weight}})`)
      .join(", ") || "n/a";
    tooltip.style("opacity", 1)
      .html(`<strong>${{d.id}}</strong><br>prob_top10: <b>${{d3.format(".1%")(d.probability || 0)}}</b><br>Top partners: ${{partners}}`);
  }})
  .on("mousemove", event => {{
    tooltip.style("left", `${{event.offsetX + 14}}px`).style("top", `${{event.offsetY + 14}}px`);
  }})
  .on("mouseout", () => tooltip.style("opacity", 0));
const labels = svg.append("g")
  .selectAll("text")
  .data(graph.nodes)
  .join("text")
  .attr("font-size", 11)
  .attr("font-weight", 650)
  .attr("paint-order", "stroke")
  .attr("stroke", "white")
  .attr("stroke-width", 3)
  .attr("fill", "#0f172a")
  .text(d => d.id);
const groups = Array.from(new Set(graph.nodes.map(d => d.group || "Other"))).sort();
const legendG = svg.append("g").attr("transform", "translate(12, 12)");
legendG.append("rect")
  .attr("width", 154).attr("height", groups.length * 20 + 26)
  .attr("rx", 6).attr("fill", "white").attr("fill-opacity", 0.88).attr("stroke", "#cbd5e1");
legendG.append("text")
  .attr("x", 8).attr("y", 16).attr("font-size", 11).attr("font-weight", 700).attr("fill", "#374151")
  .text("Voting group");
groups.forEach((grp, i) => {{
  legendG.append("circle").attr("cx", 16).attr("cy", 30 + i * 20).attr("r", 6).attr("fill", color(grp));
  legendG.append("text").attr("x", 28).attr("y", 34 + i * 20).attr("font-size", 10).attr("fill", "#374151").text(grp);
}});
svg.append("text")
  .attr("x", 12).attr("y", height - 8)
  .attr("font-size", 10).attr("fill", "#64748b")
  .text("Node size = prob_top10  ·  Edge thickness = historical affinity strength");
simulation.on("tick", () => {{
  graph.nodes.forEach(d => {{
    const r = radius(d.probability || 0) + padding;
    d.x = Math.max(r, Math.min(width - r, d.x));
    d.y = Math.max(r, Math.min(height - r, d.y));
  }});
  linkLines
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  labels.attr("x", d => d.x + radius(d.probability || 0) + 4).attr("y", d => d.y + 4);
}});
function drag(simulation) {{
  function dragstarted(event, d) {{
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }}
  function dragged(event, d) {{
    d.fx = event.x;
    d.fy = event.y;
  }}
  function dragended(event, d) {{
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }}
  return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}}
</script>
"""


def render_voting_blocs(data: dict[str, Any]) -> None:
    render_page_header("Voting Blocs")
    cooccurrence = data["bloc_cooccurrence"]
    if cooccurrence.empty:
        st.warning("No voting-bloc co-occurrence matrix found.")
        return
    components.html(voting_bloc_d3_html(cooccurrence), height=1040, scrolling=True)


def render_voting_network(data: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    render_page_header("Voting Network")
    network = data["voting_network"]
    nodes = network.get("nodes", [])
    links = network.get("links", [])
    if not nodes or not links:
        st.warning("No voting-network graph found.")
        return
    col1, col2 = st.columns(2)
    col1.metric("Nodes", len(nodes), help="Countries shown in the bilateral voting affinity network.")
    col2.metric("Edges", len(links), help="Strong historical jury-vote affinities between country pairs.")
    _info_expander(
        "What does this network show?",
        "Each **node** is a country. A **link** means the two countries have historically awarded each other "
        "high jury scores (strong bilateral affinity). **Node size** reflects the model's current top-10 "
        "probability — bigger = more favoured. Use this map to spot regional alliances that could swing "
        "the final result.",
    )
    components.html(voting_network_d3_html(network, predictions_df), height=800, scrolling=False)


def data_health_checks(data: dict[str, Any], load_time_s: float) -> pd.DataFrame:
    return pd.DataFrame(dashboard_artifact_rows(data, load_time_s))


def render_tiers(predictions_df: pd.DataFrame) -> None:
    render_page_header("Podium")
    if predictions_df.empty:
        st.warning("No country predictions found in the predictions JSON.")
        return

    position_df = position_probability_frame(predictions_df)
    if position_df.empty:
        st.warning("No top-3 position probabilities could be derived.")
        return

    st.subheader("Top-3 Probability Heatmap")
    st.plotly_chart(
        top3_heatmap(position_df, predictions_df),
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "eurovision_2026_top3_heatmap",
                "height": 980,
                "width": 1100,
                "scale": 2,
            },
        },
    )
    _info_expander(
        "How to read the Top-3 Heatmap",
        "Each row is a country; each column is a podium position (P1 = winner, P2 = runner-up, P3 = third). "
        "Darker colour = higher probability for that position. "
        "The three columns each sum to 100 % — probability is spread across all countries.",
    )

    st.subheader("Winner Probability Gauge")
    st.plotly_chart(
        winner_gauge_figure(position_df),
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "eurovision_2026_winner_gauge_top3",
                "height": 300,
                "width": 1400,
                "scale": 2,
            },
        },
    )
    _info_expander(
        "How to read the Winner Gauge",
        "Shows the top-3 countries' estimated probability of winning the contest outright (P1). "
        "The percentage is **derived** from the model's top-10 probability weighted by "
        "position-ranking distance — it is **not** a direct model output.\n\n"
        "Due to wide bootstrap CI uncertainty, even a 30 % P1 estimate carries substantial "
        "error bars. Treat this as a relative ranking signal, not a precise forecast.",
    )
    _p1 = position_df[position_df["position"] == "P1"].sort_values("probability", ascending=False)
    if not _p1.empty:
        _leader = _p1.iloc[0]
        _not_win = (1.0 - float(_leader["probability"])) * 100.0
        st.info(
            f"Winner probability is a derived estimate based on the bootstrap CI distribution, "
            f"not a direct model output. Even the current leader — **{_leader['country']} "
            f"({float(_leader['probability']):.1%})** — has a **{_not_win:.0f} %** chance of "
            "not winning. Use this as a relative signal, not a standalone forecast."
        )


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def probability_band(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def render_ci_bar(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return '<span class="missing-text">n/a</span>'
    left = max(0.0, min(100.0, lo * 100.0))
    right = max(left, min(100.0, hi * 100.0))
    width = max(1.0, right - left)
    return (
        '<div class="ci-track">'
        f'<div class="ci-range" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
        "</div>"
        f'<span class="ci-label">{lo:.2f}-{hi:.2f}</span>'
    )


def render_semi_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        prob = safe_float(row.get("prob_qualify"))
        lo = safe_float(row.get("ci80_lo"))
        hi = safe_float(row.get("ci80_hi"))
        band = probability_band(prob)
        prob_text = "n/a" if prob is None else f"{prob:.1%}"
        country_name = str(row.get("country") or "Unknown")
        country = escape(country_name)
        flag = country_flag_img(country_name, width=26)
        rank = escape(str(row.get("rank_in_semi") or ""))
        table_rows.append(
            "<tr>"
            f'<td class="rank-cell">{rank}</td>'
            f'<td class="flag-cell">{flag}</td>'
            f"<td>{country}</td>"
            f'<td><span class="prob-pill {band}">{prob_text}</span></td>'
            f"<td>{render_ci_bar(lo, hi)}</td>"
            "</tr>"
        )
    return (
        '<table class="semi-table">'
        "<thead><tr><th>#</th><th>Flag</th><th>Country</th><th>prob_qualify</th><th>CI-80</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
    )


def render_semi_qualifiers(semi_predictions: dict[str, Any]) -> None:
    render_page_header("Semi Qualifiers")
    _info_expander(
        "How to read the qualification table",
        "**prob_qualify** is the model's estimate of a country advancing from its semi-final to the Grand Final.\n\n"
        "**CI-80 bar** shows the uncertainty range (blue segment = plausible range; label = raw values).\n\n"
        "**Colour coding:** Green ≥ 75 % (likely qualifier) · Yellow 40–75 % (borderline) · Red < 40 % (unlikely) — "
        "these reflect the model's estimate, not official results.",
    )
    _info_expander(
        "How are semi-final predictions calculated?",
        SEMI_QUALIFIER_METHOD,
    )
    rows = semi_predictions.get("countries", [])
    if not isinstance(rows, list) or not rows:
        st.warning("No semi-final qualification predictions found.")
        return

    show_all = st.toggle("Show all semi-finalists", value=False)
    css = """
    <style>
    .semi-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    .semi-table th { text-align: left; color: #4b5563; font-weight: 600; padding: 0.45rem 0.6rem; border-bottom: 1px solid #d1d5db; }
    .semi-table td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #e5e7eb; vertical-align: middle; }
    .semi-table th:nth-child(1), .semi-table td:nth-child(1) { width: 3rem; }
    .semi-table th:nth-child(2), .semi-table td:nth-child(2) { width: 4rem; }
    .semi-table th:nth-child(4), .semi-table td:nth-child(4) { width: 9rem; }
    .semi-table th:nth-child(5), .semi-table td:nth-child(5) { width: 16rem; }
    .rank-cell { color: #6b7280; font-variant-numeric: tabular-nums; }
    .flag-cell img { display: inline-block; }
    .prob-pill { display: inline-block; min-width: 4.7rem; text-align: center; border-radius: 6px; padding: 0.18rem 0.45rem; font-variant-numeric: tabular-nums; font-weight: 700; }
    .prob-pill.high { background: #dcfce7; color: #166534; }
    .prob-pill.medium { background: #fef3c7; color: #92400e; }
    .prob-pill.low { background: #fee2e2; color: #991b1b; }
    .prob-pill.missing { background: #f3f4f6; color: #6b7280; }
    .ci-track { position: relative; display: inline-block; width: 9rem; height: 0.6rem; margin-right: 0.65rem; border-radius: 999px; background: #e5e7eb; vertical-align: middle; }
    .ci-range { position: absolute; top: 0; height: 100%; border-radius: 999px; background: #2563eb; }
    .ci-label { color: #4b5563; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
    .missing-text { color: #6b7280; }
    @media (max-width: 640px) {
        .semi-table { table-layout: auto; }
        .semi-table th:nth-child(5), .semi-table td:nth-child(5) { display: none; }
        .semi-table th:nth-child(4), .semi-table td:nth-child(4) { width: auto; }
        .semi-table th:nth-child(1), .semi-table td:nth-child(1) { width: 2rem; }
        .semi-table th:nth-child(2), .semi-table td:nth-child(2) { width: 3rem; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    tabs = st.tabs(["SF1", "SF2"])
    for index, semi_final in enumerate((1, 2)):
        with tabs[index]:
            sf_rows = [
                row for row in rows
                if int(row.get("semi_final") or 0) == semi_final
                and (show_all or bool(row.get("predicted_qualifier")))
            ]
            sf_rows = sorted(sf_rows, key=lambda row: row.get("rank_in_semi") or 999)
            st.markdown(render_semi_table(sf_rows), unsafe_allow_html=True)


def render_narratives(narratives: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    render_page_header("Narratives")
    _info_expander(
        "How to read the SHAP narrative",
        "The text summary explains the model's main reasons for its prediction. "
        "**Positive drivers** are features that pushed the probability up; **Negative drivers** pulled it down. "
        "The feature bars show SHAP values — each input's share of the total prediction. "
        "The CI-80/CI-50 fan chart shows the model's confidence range for each country.",
    )
    countries = narratives.get("countries", [])
    if not countries:
        st.warning("No narratives found in the narratives JSON.")
        return

    country_names = sorted(country["country"] for country in countries)
    selected = st.selectbox("Country", country_names, format_func=country_label)
    country_data = next(country for country in countries if country["country"] == selected)

    pred_row = country_prediction_row(predictions_df, selected)
    consensus_prob = safe_float(pred_row.get("probability"))
    st.markdown(country_hero_html(selected, probability=consensus_prob), unsafe_allow_html=True)
    prob_display = "n/a" if consensus_prob is None else f"{consensus_prob:.1%}"
    st.metric("Top-10 probability", prob_display, help="Consensus probability this country finishes in the Grand Final top 10.")
    # Replace embedded LGBM percentage in narrative text with consensus value so
    # the text matches the metric above (narratives.py uses LGBM single-pass).
    narrative_text = country_data.get("narrative", "")
    if consensus_prob is not None:
        narrative_text = re.sub(
            r"model probability: \d+%",
            f"model probability: {round(consensus_prob * 100)}%",
            narrative_text,
        )
    st.write(narrative_text)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive drivers")
        st.dataframe(pd.DataFrame(country_data.get("positive_drivers", [])), hide_index=True)
    negative_drivers = country_data.get("negative_drivers", [])
    if negative_drivers:
        with col2:
            st.subheader("Negative drivers")
            st.dataframe(pd.DataFrame(negative_drivers), hide_index=True)


def render_backtest(backtest: dict[str, Any]) -> None:
    render_page_header("Backtest")
    frame = backtest_frame(backtest)
    if frame.empty:
        st.warning("No backtest metrics found in the backtest JSON.")
        return

    st.dataframe(
        frame.sort_values(["year", "model"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "top10_accuracy": st.column_config.NumberColumn("Top-10 accuracy", format="%.3f"),
            "ci80_coverage": st.column_config.NumberColumn("CI-80 coverage", format="%.3f"),
        },
    )


def render_model_stats(data: dict[str, Any]) -> None:
    render_page_header("Model Stats")
    backtest = data["backtest"]
    semi_backtest = data["semi_backtest"]
    stats = aggregate_model_stats(backtest, semi_backtest)
    gf_frame = backtest_frame(backtest)
    semi_frame = semi_backtest_frame(semi_backtest)

    if stats.empty:
        st.warning("No model statistics found in the backtest artifacts.")
        return

    gf_best = best_metric_value(stats, "Grand Final Top-10", "accuracy")
    gf_ci = best_metric_value(stats, "Grand Final Top-10", "ci80_coverage")
    semi_best = best_metric_value(stats, "Semi-final qualification", "accuracy")
    semi_ci = best_metric_value(stats, "Semi-final qualification", "ci80_coverage")
    best_ci = max([value for value in [gf_ci, semi_ci] if value is not None], default=None)
    years = backtest.get("backtest_years", [])
    year_label = f"{min(years)}-{max(years)}" if years else "n/a"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best GF Top-10 accuracy", format_percent(gf_best), help="Highest top-10 prediction accuracy achieved across backtested years.")
    col2.metric("Best Semi accuracy", format_percent(semi_best), help="Best semi-final qualification accuracy in backtests.")
    col3.metric("Best CI-80 coverage", format_percent(best_ci), help="Empirical coverage of the 80% confidence interval (target: ≥ 80%).")
    col4.metric("Backtest years", year_label, help="Years used for out-of-sample model validation.")

    _info_expander(
        "How to interpret model accuracy",
        "Grand Final accuracy measures how many actual Top-10 countries appeared in the model's predicted Top 10. "
        "Semi-final accuracy measures how many actual qualifiers were included in each predicted qualifier set. "
        "CI-80 coverage checks whether historical outcomes fell inside the model's 80 % confidence interval.\n\n"
        "These are historical backtests, not guarantees for 2026. They show whether the modelling approach was "
        "credible on past contests when each holdout year was treated as unknown."
    )
    _info_expander(
        "Model target summary",
        TOP10_RATIONALE,
    )

    st.subheader("Accuracy Summary")
    stats_display = stats.copy()
    stats_display = stats_display[["scope", "model", "accuracy", "ci80_coverage"]]
    for column in ["accuracy", "ci80_coverage"]:
        if column in stats_display.columns:
            stats_display[column] = stats_display[column].map(format_percent_rounded)
    st.dataframe(
        stats_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "accuracy": st.column_config.TextColumn("Avg accuracy"),
            "ci80_coverage": st.column_config.TextColumn("Avg CI-80 coverage"),
        },
    )

    gf_tab, semi_tab = st.tabs(["Grand Final Top-10", "Semi-final qualification"])
    with gf_tab:
        st.caption(str(backtest.get("note_hyperparams", "")))
        gf_display = gf_frame.drop(columns=["top10_kpi", "ci80_kpi"], errors="ignore")
        for column in ["top10_accuracy", "ci80_coverage"]:
            if column in gf_display.columns:
                gf_display[column] = gf_display[column].map(format_percent_rounded)
        st.dataframe(
            gf_display.sort_values(["year", "model"]) if not gf_display.empty else gf_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "top10_accuracy": st.column_config.TextColumn("Top-10 accuracy"),
                "ci80_coverage": st.column_config.TextColumn("CI-80 coverage"),
            },
        )
    with semi_tab:
        st.caption(str(semi_backtest.get("note_hyperparams", "")))
        semi_display = semi_frame.drop(columns=["sf1_kpi", "sf2_kpi", "ci80_kpi"], errors="ignore")
        for column in ["qual_accuracy_overall", "qual_accuracy_sf1", "qual_accuracy_sf2", "ci80_coverage"]:
            if column in semi_display.columns:
                semi_display[column] = semi_display[column].map(format_percent_rounded)
        st.dataframe(
            semi_display.sort_values(["year", "model"]) if not semi_display.empty else semi_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "qual_accuracy_overall": st.column_config.TextColumn("Overall accuracy"),
                "qual_accuracy_sf1": st.column_config.TextColumn("SF1 accuracy"),
                "qual_accuracy_sf2": st.column_config.TextColumn("SF2 accuracy"),
                "ci80_coverage": st.column_config.TextColumn("CI-80 coverage"),
            },
        )


def render_data_health(data: dict[str, Any], load_time_s: float) -> None:
    render_page_header("Data Health")
    checks = data_health_checks(data, load_time_s)
    st.dataframe(checks, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Eurovision 2026",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dashboard_style()

    start = perf_counter()
    with st.spinner("Loading 2026 predictions, narratives, and voting network..."):
        data = load_dashboard_data()
    load_time_s = perf_counter() - start
    predictions_df = countries_frame(data["predictions"])

    page, music_enabled = render_sidebar(data, load_time_s)
    render_audio_player(music_enabled)
    render_back_to_top()
    render_tab_confetti(page)
    if page == "Overview":
        render_overview(data, predictions_df)
    elif page == "Model Stats":
        render_model_stats(data)
    elif page == "Main Ranking":
        render_predictions(data["predictions"], predictions_df, data["narratives"], data["history"])
    elif page == "Podium":
        render_tiers(predictions_df)
    elif page == "Semi Qualifiers":
        render_semi_qualifiers(data["semi_predictions"])
    elif page == "Voting Blocs":
        render_voting_blocs(data)
    elif page == "Voting Network":
        render_voting_network(data, predictions_df)
    elif page == "Narratives":
        render_narratives(data["narratives"], predictions_df)
    elif page == "Backtest":
        render_backtest(data["backtest"])
    else:
        render_data_health(data, load_time_s)


if __name__ == "__main__":
    main()
