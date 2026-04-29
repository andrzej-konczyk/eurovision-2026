"""
US-S5-04 — Bilateral jury co-occurrence network → D3-compatible JSON.

For every pair of countries that both ranked in the top-N jury finishers in
the same Grand Final, increment their co-occurrence counter.  The result is
a weighted undirected graph suitable for D3 force-directed layout.

Nodes  = 2026 participant countries (all 35, including semi-finalists).
Edges  = pairs whose jury co-occurrence count ≥ MIN_WEIGHT.
Weight = number of Grand Final years both countries shared a top-N jury rank.

Node attributes : group (Country_Group), probability (ensemble model),
                  prediction, big6, appearances (GF appearances 2016-2025).
Edge attributes : weight, years (list of co-occurrence years).

Requires: Dataset/eurovision_2016_26_enriched.csv
Optional: reports/narratives_YYYY.json  (loads ensemble probabilities)

Output:
    reports/voting_network_YYYY.json

CLI:
    python -m src.features.voting_network [--target-year YEAR]
                                           [--top-n INT] [--min-weight INT]
                                           [--data PATH] [--reports-dir DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_CSV = ROOT / "Dataset" / "eurovision_2016_26_enriched.csv"
REPORTS_DIR = ROOT / "reports"

TARGET_YEAR = 2026
TOP_N_JURY = 10    # top-N jury finishers per Grand Final that count as "in"
MIN_WEIGHT = 2     # minimum co-occurrence years for an edge to appear


# ---------------------------------------------------------------------------
# Co-occurrence computation
# ---------------------------------------------------------------------------


def _jury_top_n_per_year(
    df: pd.DataFrame,
    top_n: int,
) -> dict[int, set[str]]:
    """For each Grand Final year return the set of top-*top_n* jury countries.

    Rows with NULL jury_points are excluded (e.g. Netherlands 2024 —
    disqualified before the final).
    """
    finals = df[
        (df["Grand_Final_Ind"] == 1)
        & df["jury_points"].notna()
        & (df["Year"] < TARGET_YEAR)
    ].copy()

    result: dict[int, set[str]] = {}
    for year, grp in finals.groupby("Year"):
        top = set(grp.nlargest(top_n, "jury_points")["Country"].tolist())
        result[int(year)] = top
    return result


def build_jury_cooccurrence(
    df: pd.DataFrame,
    top_n: int = TOP_N_JURY,
) -> dict[tuple[str, str], dict]:
    """Compute bilateral jury co-occurrence for all country pairs.

    Returns ``{(A, B): {"weight": int, "years": [int, ...]}}``
    where A < B (lexicographic) to ensure uniqueness.
    """
    top_n_by_year = _jury_top_n_per_year(df, top_n)
    years_list = sorted(top_n_by_year)
    all_countries: set[str] = set()
    for s in top_n_by_year.values():
        all_countries |= s

    cooc: dict[tuple[str, str], dict] = {}
    for a, b in itertools.combinations(sorted(all_countries), 2):
        shared = [yr for yr in years_list
                  if a in top_n_by_year[yr] and b in top_n_by_year[yr]]
        if shared:
            cooc[(a, b)] = {"weight": len(shared), "years": shared}
    return cooc


# ---------------------------------------------------------------------------
# Node building
# ---------------------------------------------------------------------------


def _appearances(df: pd.DataFrame, target_year: int) -> dict[str, int]:
    """Number of Grand Final appearances per country (years < target_year)."""
    hist = df[
        (df["Grand_Final_Ind"] == 1)
        & (df["Year"] < target_year)
        & df["jury_points"].notna()
    ]
    return hist.groupby("Country").size().to_dict()


def _load_probabilities(target_year: int, reports_dir: Path) -> dict[str, float]:
    """Load ensemble probabilities from narratives JSON if available."""
    path = reports_dir / f"narratives_{target_year}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {c["country"]: c["probability"] for c in data.get("countries", [])}
    except (KeyError, json.JSONDecodeError):
        return {}


def build_nodes(
    df: pd.DataFrame,
    target_year: int,
    reports_dir: Path,
) -> list[dict]:
    """Build node list for all *target_year* participant countries."""
    participants = (
        df[df["Year"] == target_year][["Country", "Country_Group", "Big6_Ind"]]
        .drop_duplicates("Country")
        .sort_values("Country")
    )
    apps = _appearances(df, target_year)
    probs = _load_probabilities(target_year, reports_dir)

    nodes: list[dict] = []
    for _, row in participants.iterrows():
        country = row["Country"]
        prob = probs.get(country)
        nodes.append({
            "id": country,
            "group": row["Country_Group"] if pd.notna(row["Country_Group"]) else "Unknown",
            "probability": round(float(prob), 4) if prob is not None else None,
            "prediction": (
                "top10" if (prob is not None and prob >= 0.5) else "outside_top10"
            ),
            "big6": bool(row["Big6_Ind"] == 1),
            "appearances": int(apps.get(country, 0)),
        })
    return nodes


# ---------------------------------------------------------------------------
# Edge building
# ---------------------------------------------------------------------------


def build_links(
    cooc: dict[tuple[str, str], dict],
    node_ids: set[str],
    min_weight: int = MIN_WEIGHT,
) -> list[dict]:
    """Filter co-occurrence pairs to edges between *node_ids* with weight ≥ min_weight."""
    links: list[dict] = []
    for (a, b), data in cooc.items():
        if data["weight"] < min_weight:
            continue
        if a not in node_ids or b not in node_ids:
            continue
        links.append({
            "source": a,
            "target": b,
            "weight": data["weight"],
            "years": data["years"],
        })
    links.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))
    return links


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_network(
    data_path: Path = ENRICHED_CSV,
    target_year: int = TARGET_YEAR,
    top_n: int = TOP_N_JURY,
    min_weight: int = MIN_WEIGHT,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Build and return the full D3-compatible network dict."""
    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()

    cooc = build_jury_cooccurrence(df, top_n)
    nodes = build_nodes(df, target_year, reports_dir)
    node_ids = {n["id"] for n in nodes}
    links = build_links(cooc, node_ids, min_weight)

    gf_years = sorted(
        int(y) for y in df.loc[
            (df["Grand_Final_Ind"] == 1) & df["jury_points"].notna()
            & (df["Year"] < target_year), "Year"
        ].unique()
    )

    n_with_edges = len({e["source"] for e in links} | {e["target"] for e in links})

    print(f"\nVoting network  target_year={target_year}  top_n={top_n}  min_weight={min_weight}")
    print(f"Grand Final years used : {gf_years}")
    print(f"Nodes                  : {len(nodes)}")
    print(f"Edges                  : {len(links)}  (nodes with >=1 edge: {n_with_edges})")
    if links:
        top5 = links[:5]
        print(f"Top-5 edges by weight  :")
        for e in top5:
            print(f"  {e['source']:20s} - {e['target']:20s}  weight={e['weight']}  years={e['years']}")

    return {
        "meta": {
            "story": "US-S5-04",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_year": target_year,
            "top_n_jury": top_n,
            "min_weight": min_weight,
            "n_nodes": len(nodes),
            "n_edges": len(links),
            "gf_years_used": gf_years,
        },
        "nodes": nodes,
        "links": links,
    }


def save_network(network: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write *network* as JSON and return the output path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    year = network["meta"]["target_year"]
    out = reports_dir / f"voting_network_{year}.json"
    out.write_text(json.dumps(network, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        label = out.relative_to(ROOT)
    except ValueError:
        label = out
    print(f"  Saved: {label}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Jury co-occurrence network (US-S5-04)")
    parser.add_argument("--data",        type=Path, default=ENRICHED_CSV)
    parser.add_argument("--target-year", type=int,  default=TARGET_YEAR)
    parser.add_argument("--top-n",       type=int,  default=TOP_N_JURY,
                        help="Top-N jury finishers per year that count as co-occurring")
    parser.add_argument("--min-weight",  type=int,  default=MIN_WEIGHT,
                        help="Minimum co-occurrence years for an edge")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    network = build_network(
        data_path=args.data,
        target_year=args.target_year,
        top_n=args.top_n,
        min_weight=args.min_weight,
        reports_dir=args.reports_dir,
    )
    save_network(network, args.reports_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
