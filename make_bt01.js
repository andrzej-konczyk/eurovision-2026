const { spawnSync } = require("node:child_process");
const path = require("node:path");

const root = __dirname;
const python = path.join(root, ".venv", "Scripts", "python.exe");

const code = String.raw`
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path.cwd()
REPORT_JSON = ROOT / "reports" / "backtest_2022_2025.json"
OUT = ROOT / "Project_Execution_documents" / "BT-01_Backtest_Report_v2.docx"

data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

def pct(value):
    return f"{value * 100:.0f}%"

def pass_fail(value):
    return "PASS" if value else "FAIL"

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.8)
section.right_margin = Inches(0.8)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(10)
styles["Heading 1"].font.name = "Calibri"
styles["Heading 2"].font.name = "Calibri"

title = doc.add_heading("BT-01 Backtest Report v2", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = doc.add_paragraph("Eurovision 2026 prediction platform - Sprint 8 update")
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("1. Executive Summary", level=1)
doc.add_paragraph(
    "The 2022-2025 Grand Final top-10 backtest remains above the primary KPI for XGBoost "
    "and is borderline for LightGBM after the 2025 holdout update."
)
summary = doc.add_table(rows=1, cols=4)
summary.style = "Table Grid"
hdr = summary.rows[0].cells
hdr[0].text = "Model"
hdr[1].text = "Avg top-10 accuracy (4 years)"
hdr[2].text = "Status"
hdr[3].text = "Interpretation"
rows = [
    ("XGB", "72.5%", "PASS", "Stable four-year performance above the >=70% KPI."),
    ("LGBM", "70.0%", "BORDERLINE", "Meets the four-year average threshold, but 2025 is a single-year FAIL."),
]
for row in rows:
    cells = summary.add_row().cells
    for idx, value in enumerate(row):
        cells[idx].text = value

doc.add_heading("2. Backtest Results", level=1)
doc.add_paragraph(
    "Top-10 accuracy is measured as correct top-10 finalists among the model's predicted top 10. "
    "CI-80 coverage is the empirical coverage of the 80% confidence interval across the Grand Final field."
)
table = doc.add_table(rows=1, cols=6)
table.style = "Table Grid"
headers = ["Year", "XGB top-10", "XGB KPI", "LGBM top-10", "LGBM KPI", "CI-80 coverage KPI"]
for idx, header in enumerate(headers):
    table.rows[0].cells[idx].text = header

for year in ["2022", "2023", "2024", "2025"]:
    models = data["years"][year]["models"]
    xgb = models["xgb"]
    lgbm = models["lgbm"]
    ci80 = min(xgb["ci80_empirical_coverage"], lgbm["ci80_empirical_coverage"])
    ci80_pass = xgb["kpi_ci80_pass"] and lgbm["kpi_ci80_pass"]
    cells = table.add_row().cells
    cells[0].text = year
    cells[1].text = f'{pct(xgb["top10_accuracy"])} ({xgb["top10_hits"]}/10)'
    cells[2].text = pass_fail(xgb["kpi_top10_pass"])
    cells[3].text = f'{pct(lgbm["top10_accuracy"])} ({lgbm["top10_hits"]}/10)'
    cells[4].text = pass_fail(lgbm["kpi_top10_pass"])
    cells[5].text = f"{pct(ci80)} {pass_fail(ci80_pass)}"

footnote = doc.add_paragraph()
footnote.style = doc.styles["Normal"]
run = footnote.add_run(
    "Footnote: LGBM 2025 FAIL - KL-08: pre-contest prior bias. "
    "Ensemble weights zaktualizowane do xgb=1.0 (US-S8-01)."
)
run.italic = True

doc.add_heading("3. 2025 Holdout Detail", level=1)
detail = doc.add_table(rows=1, cols=4)
detail.style = "Table Grid"
for idx, header in enumerate(["Metric", "XGB", "LGBM", "Status"]):
    detail.rows[0].cells[idx].text = header
detail_rows = [
    ("Top-10 accuracy", "70% (7/10)", "60% (6/10)", "XGB PASS; LGBM FAIL"),
    ("CI-80 empirical coverage", "92%", "92%", "PASS"),
    ("Ensemble decision", "xgb=1.0", "lgbm=0.0", "US-S8-01 official weights"),
]
for row in detail_rows:
    cells = detail.add_row().cells
    for idx, value in enumerate(row):
        cells[idx].text = value

doc.add_heading("4. Decision", level=1)
doc.add_paragraph(
    "BT-01 v2 incorporates the 2025 holdout into the reporting baseline. "
    "The four-year XGB average is 72.5% PASS. The four-year LGBM average is 70.0%, "
    "which is borderline because the latest holdout fails at 60%. KL-08 remains open "
    "with odds_vs_history_delta planned as the Sprint 9 mitigation."
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT)
`;

const result = spawnSync(python, ["-c", code], {
  cwd: root,
  encoding: "utf8",
  stdio: ["ignore", "pipe", "pipe"],
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
process.exit(result.status ?? 1);
