"""Streamlit MVP dashboard for Eurovision 2026 prediction artifacts."""

from __future__ import annotations

import json
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
SEMI_PREDICTIONS_JSON = REPORTS_DIR / "semi_predictions_2026.json"
VOTING_NETWORK_JSON_CANDIDATES = [
    REPORTS_DIR / "voting_network.json",
    REPORTS_DIR / "voting_network_2026.json",
]
ENRICHED_CSV = APP_ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
BLOC_COOCCURRENCE_CSV = APP_ROOT / "data" / "features" / "bloc_cooccurrence.csv"

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


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_dashboard_data() -> dict[str, Any]:
    """Load all dashboard JSON artifacts."""
    backtest_path = first_existing_path(BACKTEST_JSON_CANDIDATES)
    voting_network_path = first_existing_path(VOTING_NETWORK_JSON_CANDIDATES)
    return {
        "predictions": load_json(str(PREDICTIONS_JSON), PREDICTIONS_JSON.stat().st_mtime_ns),
        "predictions_path": str(PREDICTIONS_JSON.relative_to(APP_ROOT)),
        "narratives": load_json(str(NARRATIVES_JSON), NARRATIVES_JSON.stat().st_mtime_ns),
        "narratives_path": str(NARRATIVES_JSON.relative_to(APP_ROOT)),
        "backtest": load_json(str(backtest_path), backtest_path.stat().st_mtime_ns),
        "backtest_path": str(backtest_path.relative_to(APP_ROOT)),
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


def render_sidebar(data: dict[str, Any], load_time_s: float) -> str:
    st.sidebar.title("Eurovision 2026")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Main Ranking",
            "Tiers",
            "Semi Qualifiers",
            "Voting Blocs",
            "Voting Network",
            "Narratives",
            "Backtest",
            "Data Health",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Loaded artifacts")
    st.sidebar.code(data["predictions_path"])
    st.sidebar.code(data["semi_predictions_path"])
    st.sidebar.code(data["voting_network_path"])
    st.sidebar.code(data["narratives_path"])
    st.sidebar.code(data["backtest_path"])
    st.sidebar.code(data["history_path"])
    st.sidebar.code(data["bloc_cooccurrence_path"])
    st.sidebar.metric("Load time", f"{load_time_s:.3f}s")
    return page


def country_flag(country: str) -> str:
    code = COUNTRY_ISO2.get(country)
    if not code:
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in code.upper())


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
        & (history["Year"].between(2016, 2024))
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


def feature_bar_chart(features: pd.DataFrame) -> go.Figure:
    plot_frame = features.sort_values("abs_value", ascending=True)
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
        font={"size": 12},
    )
    fig.add_vline(x=0, line_width=1, line_color="#64748b")
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
        xaxis={"title": "Top-10 probability", "tickformat": ".0%", "range": [0, 1]},
        yaxis_title=None,
        height=260,
        margin={"l": 90, "r": 30, "t": 55, "b": 40},
        font={"size": 12},
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
        title="Final history 2016-2024",
        xaxis_title="Year",
        yaxis={"title": "Final place", "autorange": "reversed", "dtick": 5},
        height=260,
        margin={"l": 60, "r": 30, "t": 55, "b": 40},
        font={"size": 12},
    )
    return fig


def country_card_data(
    country: str,
    predictions_df: pd.DataFrame,
    narratives: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, Any]:
    narrative = narratives_by_country(narratives).get(country, {})
    prediction = country_prediction_row(predictions_df, country)
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
    flag = card["flag"]
    narrative = card["narrative"]
    prediction = card["prediction"]
    title = f"{flag} {country}".strip()

    st.markdown(f"### {escape(title)}")
    probability = safe_float(prediction.get("probability"))
    rank = prediction.get("rank")
    cols = st.columns(3)
    cols[0].metric("Consensus rank", "n/a" if pd.isna(rank) else f"#{int(rank)}")
    cols[1].metric("Top-10 probability", "n/a" if probability is None else f"{probability:.1%}")
    cols[2].metric("Narrative signal", narrative.get("prediction", "n/a"))

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
    countries = predictions_df.sort_values("rank")["country"].tolist()
    selected = st.sidebar.selectbox("Country detail", countries)
    card = country_card_data(selected, predictions_df, data["narratives"], data["history"])
    with st.sidebar.expander(f"{card['flag']} {selected}".strip(), expanded=True):
        narrative = str(card["narrative"].get("narrative", "")).strip()
        probability = safe_float(card["prediction"].get("probability"))
        st.metric("Top-10 probability", "n/a" if probability is None else f"{probability:.1%}")
        st.write(narrative if narrative else "No narrative available.")


def render_overview(data: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    predictions = data["predictions"]
    backtest = data["backtest"]
    aggregate = backtest.get("aggregate", {})

    st.title("Eurovision 2026 Prediction Dashboard")
    st.caption("Sprint 7 Streamlit shell for local model artifacts.")

    top_country = predictions_df.iloc[0]["country"] if not predictions_df.empty else "n/a"
    top_prob = predictions_df.iloc[0].get("probability") if not predictions_df.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target year", predictions.get("target_year", 2026))
    col2.metric("Countries", len(predictions_df))
    col3.metric("Top country", top_country)
    col4.metric("Top probability", f"{top_prob:.1%}" if pd.notna(top_prob) else "n/a")

    st.subheader("Backtest KPI")
    kpi_rows = []
    for model, metrics in aggregate.items():
        kpi_rows.append(
            {
                "model": model.upper(),
                "avg_top10_accuracy": metrics.get("avg_top10_accuracy"),
                "avg_ci80_coverage": metrics.get("avg_ci80_empirical_coverage"),
                "top10_pass": metrics.get("all_top10_kpi_pass"),
                "ci80_pass": metrics.get("all_ci80_kpi_pass"),
            }
        )
    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)


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
    st.title("Main Ranking")
    st.caption(
        "Belgium qualified via SF but GF top-10 probability reflects Grand Final historical performance"
    )
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

    selected_country = st.selectbox("Open country card", ranking["country"].tolist())
    with st.expander(f"Country card: {selected_country}", expanded=True):
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
    colors = {"SAFE": "#16a34a", "LIKELY": "#f59e0b", "UNCERTAIN": "#64748b"}
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
                "color": "#334155",
            },
            customdata=plot_frame[["rank", "badge", "xgb_prob", "lgbm_prob", "model_consensus"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Rank: %{customdata[0]}<br>"
                "prob_top10: %{x:.1%}<br>"
                "Badge: %{customdata[1]}<br>"
                "XGB: %{customdata[2]:.1%}<br>"
                "LGBM: %{customdata[3]:.1%}<br>"
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
        legend_title_text="Safety badge",
        bargap=0.22,
        xaxis={"tickformat": ".0%", "range": [0, 1]},
        font={"size": 13},
    )
    for badge, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=badge,
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
            hovertemplate="<b>%{y}</b><br>Position: %{x}<br>Probability: %{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top-3 probability heatmap",
        xaxis_title="Position",
        yaxis={"title": None, "autorange": "reversed"},
        height=980,
        margin={"l": 130, "r": 40, "t": 70, "b": 50},
        font={"size": 13},
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
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(row["probability"]),
                number={"valueformat": ".1%", "font": {"size": 24}},
                title={"text": f"#{int(row['rank'])} {row['country']}", "font": {"size": 18}},
                domain={"x": [0.08, 0.92], "y": [0.68 - index * 0.33, 0.94 - index * 0.33]},
                gauge={
                    "axis": {"range": [0.0, 1.0], "tickformat": ".0%"},
                    "bar": {"color": "#2563eb"},
                    "bgcolor": "#f8fafc",
                    "bordercolor": "#cbd5e1",
                    "steps": [
                        {"range": [0.0, 0.2], "color": "#e2e8f0"},
                        {"range": [0.2, 0.5], "color": "#dbeafe"},
                        {"range": [0.5, 1.0], "color": "#bfdbfe"},
                    ],
                },
            )
        )
    fig.update_layout(
        title=f"Winner probability gauge: top {top_n}",
        height=560,
        margin={"l": 35, "r": 35, "t": 70, "b": 30},
        font={"size": 14},
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
const margin = {{top: 28, right: 24, bottom: 24, left: 150}};
const cell = 18;
const width = margin.left + margin.right + blocs.length * 92;
const height = margin.top + margin.bottom + countries.length * cell;
const root = d3.select("#bloc-d3").html("");
const svg = root.append("svg")
  .attr("viewBox", [0, 0, width, height])
  .attr("width", "100%")
  .attr("height", height);
const x = d3.scaleBand().domain(blocs).range([margin.left, width - margin.right]).padding(0.08);
const y = d3.scaleBand().domain(countries).range([margin.top, height - margin.bottom]).padding(0.08);
svg.append("g")
  .selectAll("text")
  .data(blocs)
  .join("text")
  .attr("x", d => x(d) + x.bandwidth() / 2)
  .attr("y", 18)
  .attr("text-anchor", "middle")
  .attr("font-size", 12)
  .attr("font-weight", 700)
  .text(d => d);
svg.append("g")
  .selectAll("text")
  .data(countries)
  .join("text")
  .attr("x", margin.left - 10)
  .attr("y", d => y(d) + y.bandwidth() / 2)
  .attr("dominant-baseline", "middle")
  .attr("text-anchor", "end")
  .attr("font-size", 12)
  .text(d => d);
svg.append("g")
  .selectAll("rect")
  .data(data)
  .join("rect")
  .attr("x", d => x(d.bloc))
  .attr("y", d => y(d.country))
  .attr("width", x.bandwidth())
  .attr("height", y.bandwidth())
  .attr("rx", 3)
  .attr("fill", d => d.member ? "#2563eb" : "#e5e7eb")
  .append("title")
  .text(d => `${{d.country}} / ${{d.bloc}}: ${{d.member ? "member" : "not member"}}`);
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
            "top_partners": [partner for partner, _ in partners],
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
  .attr("stroke-width", d => stroke(d.weight || 1))
  .append("title")
  .text(d => `${{d.source.id || d.source}} - ${{d.target.id || d.target}}: weight=${{d.weight}}`);
const linkLines = svg.selectAll("line");
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
    tooltip.style("opacity", 1)
      .html(`<strong>${{d.id}}</strong><br>prob_top10: ${{d3.format(".1%")(d.probability || 0)}}<br>Top partners: ${{(d.top_partners || []).join(", ") || "n/a"}}`);
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
    st.title("Voting Blocs")
    cooccurrence = data["bloc_cooccurrence"]
    if cooccurrence.empty:
        st.warning("No voting-bloc co-occurrence matrix found.")
        return
    components.html(voting_bloc_d3_html(cooccurrence), height=920, scrolling=True)
    st.dataframe(cooccurrence, use_container_width=True, hide_index=True)


def render_voting_network(data: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    st.title("Voting Network")
    network = data["voting_network"]
    nodes = network.get("nodes", [])
    links = network.get("links", [])
    if not nodes or not links:
        st.warning("No voting-network graph found.")
        return
    col1, col2 = st.columns(2)
    col1.metric("Nodes", len(nodes))
    col2.metric("Edges", len(links))
    components.html(voting_network_d3_html(network, predictions_df), height=800, scrolling=False)


def render_tiers(predictions_df: pd.DataFrame) -> None:
    st.title("Tiers")
    if predictions_df.empty:
        st.warning("No country predictions found in the predictions JSON.")
        return

    position_df = position_probability_frame(predictions_df)
    if position_df.empty:
        st.warning("No top-3 position probabilities could be derived.")
        return

    column_sums = position_df.groupby("position", as_index=False)["probability"].sum()
    st.subheader("Tier 3: Top-3 Probability Heatmap")
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
    st.dataframe(
        column_sums,
        use_container_width=True,
        hide_index=True,
        column_config={
            "position": st.column_config.TextColumn("Position"),
            "probability": st.column_config.NumberColumn("Column sum", format="%.6f"),
        },
    )

    st.subheader("Tier 4: Winner Probability Gauge")
    st.plotly_chart(
        winner_gauge_figure(position_df),
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "eurovision_2026_winner_gauge_top3",
                "height": 560,
                "width": 1400,
                "scale": 2,
            },
        },
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
        country = escape(str(row.get("country") or "Unknown"))
        flag = escape(str(row.get("flag") or ""))
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
    st.title("Semi Qualifiers")
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
    .flag-cell { font-size: 1.2rem; }
    .prob-pill { display: inline-block; min-width: 4.7rem; text-align: center; border-radius: 6px; padding: 0.18rem 0.45rem; font-variant-numeric: tabular-nums; font-weight: 700; }
    .prob-pill.high { background: #dcfce7; color: #166534; }
    .prob-pill.medium { background: #fef3c7; color: #92400e; }
    .prob-pill.low { background: #fee2e2; color: #991b1b; }
    .prob-pill.missing { background: #f3f4f6; color: #6b7280; }
    .ci-track { position: relative; display: inline-block; width: 9rem; height: 0.6rem; margin-right: 0.65rem; border-radius: 999px; background: #e5e7eb; vertical-align: middle; }
    .ci-range { position: absolute; top: 0; height: 100%; border-radius: 999px; background: #2563eb; }
    .ci-label { color: #4b5563; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
    .missing-text { color: #6b7280; }
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


def render_narratives(narratives: dict[str, Any]) -> None:
    st.title("Narratives")
    countries = narratives.get("countries", [])
    if not countries:
        st.warning("No narratives found in the narratives JSON.")
        return

    country_names = [country["country"] for country in countries]
    selected = st.selectbox("Country", country_names)
    country_data = next(country for country in countries if country["country"] == selected)

    st.metric("Model probability", f"{country_data.get('probability', 0):.1%}")
    st.write(country_data.get("narrative", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive drivers")
        st.dataframe(pd.DataFrame(country_data.get("positive_drivers", [])), hide_index=True)
    with col2:
        st.subheader("Negative drivers")
        st.dataframe(pd.DataFrame(country_data.get("negative_drivers", [])), hide_index=True)


def render_backtest(backtest: dict[str, Any]) -> None:
    st.title("Backtest")
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


def render_data_health(data: dict[str, Any], load_time_s: float) -> None:
    st.title("Data Health")
    checks = pd.DataFrame(
        [
            {"check": "Predictions JSON", "path": data["predictions_path"], "loaded": bool(data["predictions"])},
            {"check": "Semi predictions JSON", "path": data["semi_predictions_path"], "loaded": bool(data["semi_predictions"])},
            {"check": "Voting network JSON", "path": data["voting_network_path"], "loaded": bool(data["voting_network"])},
            {"check": "Narratives JSON", "path": data["narratives_path"], "loaded": bool(data["narratives"])},
            {"check": "Backtest JSON", "path": data["backtest_path"], "loaded": bool(data["backtest"])},
            {"check": "History CSV", "path": data["history_path"], "loaded": not data["history"].empty},
            {"check": "Load KPI < 2s", "path": "runtime", "loaded": load_time_s < 2.0},
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Eurovision 2026",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    start = perf_counter()
    data = load_dashboard_data()
    load_time_s = perf_counter() - start
    predictions_df = countries_frame(data["predictions"])

    page = render_sidebar(data, load_time_s)
    render_country_detail_sidebar(data, predictions_df)
    if page == "Overview":
        render_overview(data, predictions_df)
    elif page == "Main Ranking":
        render_predictions(data["predictions"], predictions_df, data["narratives"], data["history"])
    elif page == "Tiers":
        render_tiers(predictions_df)
    elif page == "Semi Qualifiers":
        render_semi_qualifiers(data["semi_predictions"])
    elif page == "Voting Blocs":
        render_voting_blocs(data)
    elif page == "Voting Network":
        render_voting_network(data, predictions_df)
    elif page == "Narratives":
        render_narratives(data["narratives"])
    elif page == "Backtest":
        render_backtest(data["backtest"])
    else:
        render_data_health(data, load_time_s)


if __name__ == "__main__":
    main()
