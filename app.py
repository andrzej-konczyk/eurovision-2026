"""Streamlit MVP dashboard for Eurovision 2026 prediction artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = APP_ROOT / "reports"

PREDICTIONS_JSON = REPORTS_DIR / "predictions_2026.json"
NARRATIVES_JSON = REPORTS_DIR / "narratives_2026.json"
BACKTEST_JSON = REPORTS_DIR / "backtest_2022_2024.json"
SEMI_PREDICTIONS_JSON = REPORTS_DIR / "semi_predictions_2026.json"


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


def load_dashboard_data() -> dict[str, Any]:
    """Load all dashboard JSON artifacts."""
    return {
        "predictions": load_json(str(PREDICTIONS_JSON), PREDICTIONS_JSON.stat().st_mtime_ns),
        "predictions_path": str(PREDICTIONS_JSON.relative_to(APP_ROOT)),
        "narratives": load_json(str(NARRATIVES_JSON), NARRATIVES_JSON.stat().st_mtime_ns),
        "narratives_path": str(NARRATIVES_JSON.relative_to(APP_ROOT)),
        "backtest": load_json(str(BACKTEST_JSON), BACKTEST_JSON.stat().st_mtime_ns),
        "backtest_path": str(BACKTEST_JSON.relative_to(APP_ROOT)),
        "semi_predictions": load_json(str(SEMI_PREDICTIONS_JSON), SEMI_PREDICTIONS_JSON.stat().st_mtime_ns),
        "semi_predictions_path": str(SEMI_PREDICTIONS_JSON.relative_to(APP_ROOT)),
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
        ["Overview", "Predictions", "Semi Qualifiers", "Narratives", "Backtest", "Data Health"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Loaded artifacts")
    st.sidebar.code(data["predictions_path"])
    st.sidebar.code(data["semi_predictions_path"])
    st.sidebar.code(data["narratives_path"])
    st.sidebar.code(data["backtest_path"])
    st.sidebar.metric("Load time", f"{load_time_s:.3f}s")
    return page


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


def render_predictions(predictions: dict[str, Any], predictions_df: pd.DataFrame) -> None:
    st.title("Predictions")
    if predictions_df.empty:
        st.warning("No country predictions found in the predictions JSON.")
        return

    view = st.segmented_control(
        "Prediction view",
        ["Consensus", "XGB", "LGBM"],
        default="Consensus",
    )

    if view == "XGB":
        frame = model_predictions_frame(predictions, "xgb")
        visible_columns = ["rank", "country", "prob_mean", "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi"]
    elif view == "LGBM":
        frame = model_predictions_frame(predictions, "lgbm")
        visible_columns = ["rank", "country", "prob_mean", "ci80_lo", "ci80_hi", "ci50_lo", "ci50_hi"]
    else:
        frame = predictions_df
        visible_columns = [
            column
            for column in [
                "rank",
                "country",
                "probability",
                "xgb_prob",
                "lgbm_prob",
                "xgb_rank",
                "lgbm_rank",
                "in_consensus_top10",
            ]
            if column in frame.columns
        ]

    st.dataframe(
        frame[visible_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Consensus probability",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "prob_mean": st.column_config.ProgressColumn(
                "Probability",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "xgb_prob": st.column_config.NumberColumn("XGB probability", format="%.3f"),
            "lgbm_prob": st.column_config.NumberColumn("LGBM probability", format="%.3f"),
            "ci80_lo": st.column_config.NumberColumn("CI-80 low", format="%.3f"),
            "ci80_hi": st.column_config.NumberColumn("CI-80 high", format="%.3f"),
            "ci50_lo": st.column_config.NumberColumn("CI-50 low", format="%.3f"),
            "ci50_hi": st.column_config.NumberColumn("CI-50 high", format="%.3f"),
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
            {"check": "Narratives JSON", "path": data["narratives_path"], "loaded": bool(data["narratives"])},
            {"check": "Backtest JSON", "path": data["backtest_path"], "loaded": bool(data["backtest"])},
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
    if page == "Overview":
        render_overview(data, predictions_df)
    elif page == "Predictions":
        render_predictions(data["predictions"], predictions_df)
    elif page == "Semi Qualifiers":
        render_semi_qualifiers(data["semi_predictions"])
    elif page == "Narratives":
        render_narratives(data["narratives"])
    elif page == "Backtest":
        render_backtest(data["backtest"])
    else:
        render_data_health(data, load_time_s)


if __name__ == "__main__":
    main()
