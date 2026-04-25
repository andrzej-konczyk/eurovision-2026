"""
EDA Report — US-S3-06
Usage:  python reports/eda_20260425.py
Output: reports/charts/eda_20260425.html  (standalone interactive HTML)

10 Plotly charts covering placement, jury/tele split, voting blocs,
running order bias, language, Big5 vs qualifiers, social signals,
rule reform impact, and qualification rates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
OUT_DIR = ROOT / "reports" / "charts"
OUT_HTML = OUT_DIR / "eda_20260425.html"

PALETTE = px.colors.qualitative.Safe


def _load() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ── Chart 1: Final placement distribution by year (box) ───────────────────────

def chart_placement_by_year(df: pd.DataFrame) -> go.Figure:
    finals = df[(df["Grand_Final_Ind"] == 1) & df["Final_Place"].notna()].copy()
    fig = px.box(
        finals, x="Year", y="Final_Place",
        title="Chart 1 — Final Placement Distribution by Year",
        labels={"Final_Place": "Final Place", "Year": "Year"},
        color_discrete_sequence=["#2196F3"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 2: Jury vs. tele points scatter (coloured by bloc) ─────────────────

def chart_jury_vs_tele(df: pd.DataFrame) -> go.Figure:
    finals = df[
        (df["Grand_Final_Ind"] == 1) &
        df["jury_points"].notna() &
        df["tele_points"].notna()
    ].copy()
    fig = px.scatter(
        finals, x="jury_points", y="tele_points",
        color="Country_Group", hover_data=["Year", "Country", "Final_Place"],
        title="Chart 2 — Jury vs. Televote Points (Grand Finalists)",
        labels={"jury_points": "Jury Points", "tele_points": "Televote Points"},
        opacity=0.75,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=500, y1=500,
                  line=dict(dash="dash", color="grey"))
    return fig


# ── Chart 3: Avg final place by voting bloc ───────────────────────────────────

def chart_bloc_avg_place(df: pd.DataFrame) -> go.Figure:
    finals = df[(df["Grand_Final_Ind"] == 1) & df["Final_Place"].notna()].copy()
    bloc_avg = (
        finals.groupby(["Year", "Country_Group"])["Final_Place"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        bloc_avg, x="Year", y="Final_Place", color="Country_Group",
        markers=True,
        title="Chart 3 — Avg Final Place by Voting Bloc (lower = better)",
        labels={"Final_Place": "Avg Final Place"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 4: Running order vs. final place ────────────────────────────────────

def chart_running_order_bias(df: pd.DataFrame) -> go.Figure:
    finals = df[
        (df["Grand_Final_Ind"] == 1) &
        df["Running_Order_Final"].notna() &
        df["Final_Place"].notna()
    ].copy()
    # Bin running order into quartiles
    finals["RO_bin"] = pd.qcut(finals["Running_Order_Final"], q=4,
                                labels=["Q1 (early)", "Q2", "Q3", "Q4 (late)"])
    bin_avg = finals.groupby("RO_bin", observed=True)["Final_Place"].mean().reset_index()
    fig = px.bar(
        bin_avg, x="RO_bin", y="Final_Place",
        title="Chart 4 — Running Order Quartile vs. Avg Final Place",
        labels={"RO_bin": "Running Order Quartile", "Final_Place": "Avg Final Place"},
        color_discrete_sequence=["#FF5722"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 5: Jury / tele ratio over years ────────────────────────────────────

def chart_jury_tele_ratio(df: pd.DataFrame) -> go.Figure:
    finals = df[
        (df["Grand_Final_Ind"] == 1) &
        df["jury_points"].notna() &
        df["tele_points"].notna()
    ].copy()
    finals["total"] = finals["jury_points"] + finals["tele_points"]
    finals["jury_share"] = finals["jury_points"] / finals["total"].replace(0, np.nan)
    yearly = finals.groupby("Year")["jury_share"].mean().reset_index()
    fig = px.bar(
        yearly, x="Year", y="jury_share",
        title="Chart 5 — Avg Jury Share of Total Points by Year",
        labels={"jury_share": "Jury Share (0–1)"},
        color_discrete_sequence=["#9C27B0"],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="50/50")
    return fig


# ── Chart 6: Language distribution over years (top 6) ────────────────────────

def chart_language_dist(df: pd.DataFrame) -> go.Figure:
    top_langs = df["Language1"].value_counts().head(6).index.tolist()
    lang_df = df[df["Language1"].isin(top_langs)].copy()
    counts = lang_df.groupby(["Year", "Language1"]).size().reset_index(name="n")
    fig = px.bar(
        counts, x="Year", y="n", color="Language1",
        barmode="stack",
        title="Chart 6 — Language Distribution by Year (Top 6)",
        labels={"n": "Entries", "Language1": "Primary Language"},
    )
    return fig


# ── Chart 7: Big5 + host vs. qualifier performance ────────────────────────────

def chart_big5_vs_qualifier(df: pd.DataFrame) -> go.Figure:
    finals = df[(df["Grand_Final_Ind"] == 1) & df["Final_Place"].notna()].copy()
    finals["Group"] = finals["Big6_Ind"].map({1: "Big5/Host", 0: "Qualifier"})
    fig = px.box(
        finals, x="Year", y="Final_Place", color="Group",
        title="Chart 7 — Final Place: Big5/Host vs. Qualifiers by Year",
        labels={"Final_Place": "Final Place"},
        color_discrete_map={"Big5/Host": "#F44336", "Qualifier": "#4CAF50"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 8: MyESB Community rank vs. final place ────────────────────────────

def chart_myesb_vs_place(df: pd.DataFrame) -> go.Figure:
    finals = df[(df["Grand_Final_Ind"] == 1) & df["Final_Place"].notna()].copy()
    fig = px.scatter(
        finals, x="MyESB_Community", y="Final_Place",
        trendline="ols", opacity=0.6,
        hover_data=["Year", "Country"],
        title="Chart 8 — MyESB Community Rank vs. Final Place",
        labels={"MyESB_Community": "MyESB Community Rank (lower = better prediction)",
                "Final_Place": "Final Place"},
        color_discrete_sequence=["#00BCD4"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 9: OGAE Points vs. final place ─────────────────────────────────────

def chart_ogae_vs_place(df: pd.DataFrame) -> go.Figure:
    finals = df[
        (df["Grand_Final_Ind"] == 1) &
        df["Final_Place"].notna() &
        df["OGAE_Points"].notna()
    ].copy()
    fig = px.scatter(
        finals, x="OGAE_Points", y="Final_Place",
        trendline="ols", opacity=0.6,
        hover_data=["Year", "Country"],
        title="Chart 9 — OGAE Points vs. Final Place",
        labels={"OGAE_Points": "OGAE Points", "Final_Place": "Final Place"},
        color_discrete_sequence=["#FF9800"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Chart 10: Qualification rate by country (2016–2025) ───────────────────────

def chart_qualification_rate(df: pd.DataFrame) -> go.Figure:
    # Exclude Big5/Host (always qualify) and 2026 (contest not yet held)
    hist = df[(df["Year"] < 2026) & (df["Big6_Ind"] == 0)].copy()
    appearances = hist.groupby("Country").size()
    qualifications = hist[hist["Grand_Final_Ind"] == 1].groupby("Country").size()
    rate = (qualifications / appearances).dropna().sort_values(ascending=False)
    rate_df = rate.reset_index()
    rate_df.columns = ["Country", "QualRate"]

    fig = px.bar(
        rate_df, x="Country", y="QualRate",
        title="Chart 10 — Grand Final Qualification Rate by Country (2016–2025, excl. Big5/Host)",
        labels={"QualRate": "Qualification Rate"},
        color="QualRate", color_continuous_scale="Blues",
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig


# ── Assemble & export ─────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()
    print(f"Loaded {len(df)} rows from {ENRICHED_CSV.name}")

    charts = [
        chart_placement_by_year(df),
        chart_jury_vs_tele(df),
        chart_bloc_avg_place(df),
        chart_running_order_bias(df),
        chart_jury_tele_ratio(df),
        chart_language_dist(df),
        chart_big5_vs_qualifier(df),
        chart_myesb_vs_place(df),
        chart_ogae_vs_place(df),
        chart_qualification_rate(df),
    ]

    # Write standalone HTML — all charts on one page
    html_parts = ["<html><head><meta charset='utf-8'>"
                  "<title>Eurovision 2026 EDA</title></head><body>",
                  "<h1>Eurovision 2026 — EDA Report (2016–2026)</h1>",
                  "<p>Generated: 2026-04-25 | Dataset: eurovision_2016_26_enriched.csv</p><hr>"]

    for i, fig in enumerate(charts, 1):
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 1)))
        html_parts.append("<hr>")

    html_parts.append("</body></html>")

    OUT_HTML.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Saved: {OUT_HTML.relative_to(ROOT)}")
    print(f"Charts: {len(charts)}")


if __name__ == "__main__":
    main()
