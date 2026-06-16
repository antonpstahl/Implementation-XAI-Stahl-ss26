"""
Generate paper-ready error-taxonomy figure and LaTeX table.

Outputs (all in results/figures/):
  fig_error_taxonomy.pdf  — vector figure for LaTeX inclusion
  fig_error_taxonomy.png  — 300 dpi raster backup
  tab_error_taxonomy.tex  — LaTeX booktabs table
  tab_error_taxonomy.csv  — machine-readable table

Run from repo root:  python results/figures/build_taxonomy_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent
SRC     = ROOT / "results" / "error_taxonomy"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── taxonomy data (from NB 09, category_frequencies.csv) ─────────────────────
#  Source: primary coding of 30 worst-RA/SA cases
#  E-categories = Eval Artefact (measurement error in NB 08 extractor)
#  B/C-categories = Explanation Error (LLM output error)

CATEGORIES = [
    # code, short_label, description, source, n
    ("E1", "E1: Feature swap",
     "Extractor assigns wrong feature at rank 0",
     "Eval Artefact", 9),
    ("E2", "E2: Sign inversion",
     "Extractor inverts contribution sign",
     "Eval Artefact", 6),
    ("E3", "E3: Cross-instance hallucination",
     "Extractor confabulates values from other instances",
     "Eval Artefact", 4),
    ("C",  "C: Sign error (yr/temp)",
     "LLM overrides negative contribution with trend narrative",
     "Explanation Error", 5),
    ("B2", "B2: Near-contribution rank swap",
     "LLM reorders features with similar |contribution|",
     "Explanation Error", 4),
    ("B1", "B1: Pos./neg. grouping",
     "LLM groups by sign, losing rank information",
     "Explanation Error", 2),
]

N_TOTAL      = 30
N_EVAL       = sum(n for *_, src, n in CATEGORIES if src == "Eval Artefact")
N_EXPL       = sum(n for *_, src, n in CATEGORIES if src == "Explanation Error")

# ── colour scheme ─────────────────────────────────────────────────────────────
C_EVAL = "#5B8DB8"   # steel blue  — Eval Artefact
C_EXPL = "#C26B6B"   # muted red   — Explanation Error
C_EVAL_LIGHT = "#A8C4DC"
C_EXPL_LIGHT = "#E0A8A8"

# ── figure layout ─────────────────────────────────────────────────────────────
# 7.2 in × 3.2 in — fits a double-column ACM/IEEE page
fig = plt.figure(figsize=(7.2, 3.2), dpi=300)
ax_bar  = fig.add_axes([0.02, 0.08, 0.58, 0.82])   # left: category bar chart
ax_split = fig.add_axes([0.68, 0.28, 0.30, 0.44])  # right: source stacked bar

# ── Panel A: horizontal bar chart ─────────────────────────────────────────────
codes   = [c[0] for c in CATEGORIES]
labels  = [c[1] for c in CATEGORIES]
sources = [c[3] for c in CATEGORIES]
counts  = [c[4] for c in CATEGORIES]
colors  = [C_EVAL if s == "Eval Artefact" else C_EXPL for s in sources]

y = np.arange(len(CATEGORIES))
bars = ax_bar.barh(y, counts, color=colors, height=0.55, edgecolor="white", linewidth=0.6)

# Count labels
for bar, n in zip(bars, counts):
    ax_bar.text(
        bar.get_width() + 0.12, bar.get_y() + bar.get_height() / 2,
        str(n), va="center", ha="left", fontsize=8.5, fontweight="bold",
    )

# Divider line between Eval-Artefact and Explanation-Error groups
# First 3 rows (y=3,4,5 from bottom after flip) are Eval; next 3 are Expl
# Categories are ordered: E1, E2, E3, C, B2, B1  →  y=5..3 eval, y=2..0 expl
ax_bar.axhline(y=2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
# Labels point INTO their group: Explanation Error is above divider (y>2.5), Eval Artefact below
ax_bar.text(9.6, 2.62, "Explanation Error", fontsize=7, color=C_EXPL, ha="right", va="bottom")
ax_bar.text(9.6, 2.38, "Eval Artefact", fontsize=7, color=C_EVAL, ha="right", va="top")

ax_bar.set_yticks(y)
ax_bar.set_yticklabels(labels, fontsize=8.5)
ax_bar.set_xlabel("Number of Cases (n = 30)", fontsize=8.5)
ax_bar.set_xlim(0, 10.5)
ax_bar.set_ylim(-0.5, len(CATEGORIES) - 0.5)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.tick_params(axis="y", length=0)
ax_bar.set_title("(a)  Error Category Frequencies", fontsize=9, fontweight="bold", pad=6)

legend_handles = [
    mpatches.Patch(color=C_EVAL, label=f"Eval Artefact (n={N_EVAL})"),
    mpatches.Patch(color=C_EXPL, label=f"Explanation Error (n={N_EXPL})"),
]
ax_bar.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
              framealpha=0.85, edgecolor="lightgray")

# ── Panel B: stacked horizontal bar ──────────────────────────────────────────
# Single bar showing Eval Artefact vs. Explanation Error split
ax_split.barh([0], [N_EVAL], color=C_EVAL, height=0.5, label=f"Eval Artefact")
ax_split.barh([0], [N_EXPL], left=[N_EVAL], color=C_EXPL, height=0.5,
              label=f"Explanation Error")

# Percentage labels inside segments
ax_split.text(N_EVAL / 2, 0, f"Eval Artefact\n{N_EVAL}/{N_TOTAL} = {N_EVAL/N_TOTAL*100:.0f}%",
              ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
ax_split.text(N_EVAL + N_EXPL / 2, 0,
              f"Expl. Error\n{N_EXPL}/{N_TOTAL} = {N_EXPL/N_TOTAL*100:.0f}%",
              ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")

ax_split.set_xlim(0, N_TOTAL)
ax_split.set_yticks([])
ax_split.set_xticks([])
ax_split.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax_split.set_title("(b)  Error Source (n = 30)", fontsize=9, fontweight="bold", pad=8)

# ── save ──────────────────────────────────────────────────────────────────────
pdf_path = OUT_DIR / "fig_error_taxonomy.pdf"
png_path = OUT_DIR / "fig_error_taxonomy.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Figure saved: {pdf_path}")
print(f"Figure saved: {png_path}")
plt.close(fig)

# ── LaTeX table ───────────────────────────────────────────────────────────────
tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Error taxonomy for the 30 lowest-RA/SA cases (n\,=\,30 coded manually).",
    r"  E-categories are measurement artefacts of the NB\,08 extractor;",
    r"  B- and C-categories are genuine LLM explanation errors.",
    r"  The corrected upper bounds for Rank Agreement and Sign Agreement",
    r"  (assuming all E-errors resolved) are RA\,$\leq$\,0.582 and SA\,$\leq$\,0.795",
    r"  vs.\ observed RA\,=\,0.517 and SA\,=\,0.711.}",
    r"\label{tab:error-taxonomy}",
    r"\small",
    r"\begin{tabular}{@{}llp{5.6cm}lr@{}}",
    r"\toprule",
    r"\textbf{Code} & \textbf{Source} & \textbf{Description} & \textbf{n} & \textbf{\%} \\",
    r"\midrule",
    r"\multicolumn{5}{@{}l}{\textit{Eval Artefact (NB\,08 extractor)}} \\[2pt]",
]

eval_cats = [(c, d, n) for c, _, d, src, n in CATEGORIES if src == "Eval Artefact"]
for i, (code, desc, n) in enumerate(eval_cats):
    pct = f"{n/N_TOTAL*100:.0f}"
    suffix = r"\\[4pt]" if i == len(eval_cats) - 1 else r"\\"
    tex_lines.append(rf"\quad {code} & Eval Artefact & {desc} & {n} & {pct}\% {suffix}")

tex_lines += [
    r"\multicolumn{5}{@{}l}{\textit{Explanation Error (LLM output)}} \\[2pt]",
]

for code, _, desc, source, n in CATEGORIES:
    if source != "Explanation Error":
        continue
    pct = f"{n/N_TOTAL*100:.0f}"
    tex_lines.append(rf"\quad {code} & Expl.\ Error & {desc} & {n} & {pct}\% \\")

tex_lines += [
    r"\midrule",
    rf"\textbf{{Total}} & & & \textbf{{{N_TOTAL}}} & \textbf{{100\%}} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

tex_path = OUT_DIR / "tab_error_taxonomy.tex"
tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
print(f"LaTeX table saved: {tex_path}")

# ── CSV table ─────────────────────────────────────────────────────────────────
import csv
csv_path = OUT_DIR / "tab_error_taxonomy.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Code", "Short_Label", "Description", "Source", "n", "pct"])
    for code, short, desc, source, n in CATEGORIES:
        w.writerow([code, short, desc, source, n, f"{n/N_TOTAL*100:.1f}"])
    w.writerow(["Total", "", "", "", N_TOTAL, "100.0"])
print(f"CSV table saved: {csv_path}")

print("\nDone. All outputs in results/figures/")
