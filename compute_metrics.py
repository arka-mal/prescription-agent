"""
compute_metrics.py
------------------
Reads batch-output/results.csv produced by run_batch.py
Computes all performance metrics and saves:
  - batch-output/metrics_summary.txt   (human-readable numbers)
  - batch-output/chart_coverage.png    (segmentation coverage per zone)
  - batch-output/chart_bbox.png        (bounding box rate per zone)
  - batch-output/chart_confidence.png  (High/Medium/Low distribution)
  - batch-output/chart_blocks.png      (OCR block count histogram)
  - batch-output/chart_scatter.png     (OCR blocks vs segments found)

Usage:
    python compute_metrics.py
"""

import sys
import os
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install pandas matplotlib numpy")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "batch-output"
CSV_PATH   = OUTPUT_DIR / "results-final.csv"
SUMMARY    = OUTPUT_DIR / "metrics_summary.txt"

if not CSV_PATH.exists():
    print(f"ERROR: results.csv not found at {CSV_PATH}")
    print("Run run_batch.py first.")
    sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
total = len(df)
success_df = df[df["status"] == "success"].copy()
n = len(success_df)

print(f"\n{'='*60}")
print(f"  Prescription Pipeline — Metrics Report")
print(f"{'='*60}")
print(f"  Total images processed : {total}")
print(f"  Successful             : {n}")
print(f"  Failed                 : {total - n}")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":   150,
})

ZONE_LABELS = {
    "prescriber_footer": "Prescriber",
    "patient_header":    "Patient Header",
    "drug_list":         "Drug List",
    "dosage_column":     "Dosage / Sig",
    "duration_column":   "Duration",
    "clinical_notes":    "Clinical Notes",
    "annotation":        "Annotation",
}

ZONE_KEYS = list(ZONE_LABELS.keys())

ZONE_COLORS = [
    "#6366f1",  # prescriber  — indigo
    "#3b82f6",  # patient     — blue
    "#22c55e",  # drug        — green
    "#eab308",  # dosage      — yellow
    "#ec4899",  # duration    — pink
    "#f97316",  # clinical    — orange
    "#94a3b8",  # annotation  — slate
]

COL_ZONE = {
    "prescriber_footer": "zone_prescriber",
    "patient_header":    "zone_patient",
    "drug_list":         "zone_drug",
    "dosage_column":     "zone_dosage",
    "duration_column":   "zone_duration",
    "clinical_notes":    "zone_clinical",
    "annotation":        "zone_annotation",
}

COL_BBOX = {
    "prescriber_footer": "bbox_prescriber",
    "patient_header":    "bbox_patient",
    "drug_list":         "bbox_drug",
    "dosage_column":     "bbox_dosage",
    "duration_column":   "bbox_duration",
    "clinical_notes":    "bbox_clinical",
    "annotation":        "bbox_annotation",
}

# ── 1. Segmentation Coverage Rate ─────────────────────────────────────────────
coverage = {}
for zone, col in COL_ZONE.items():
    if col in success_df.columns:
        coverage[zone] = round(success_df[col].sum() / n * 100, 1)
    else:
        coverage[zone] = 0.0

# ── 2. BBox Assignment Rate ───────────────────────────────────────────────────
bbox_rate = {}
for zone, col in COL_BBOX.items():
    zone_col = COL_ZONE[zone]
    identified = success_df[zone_col].sum() if zone_col in success_df.columns else 0
    bbox_yes   = success_df[col].sum() if col in success_df.columns else 0
    bbox_rate[zone] = round((bbox_yes / identified * 100) if identified > 0 else 0, 1)

# ── 3. Confidence Distribution ────────────────────────────────────────────────
conf_counts = success_df["layout_confidence"].value_counts()
high_pct   = round(conf_counts.get("high",   0) / n * 100, 1)
medium_pct = round(conf_counts.get("medium", 0) / n * 100, 1)
low_pct    = round(conf_counts.get("low",    0) / n * 100, 1)

ocr_conf_counts = success_df["ocr_confidence"].value_counts()
ocr_high   = round(ocr_conf_counts.get("high",   0) / n * 100, 1)
ocr_medium = round(ocr_conf_counts.get("medium", 0) / n * 100, 1)
ocr_low    = round(ocr_conf_counts.get("low",    0) / n * 100, 1)

# ── 4. Mixed Script Stats ─────────────────────────────────────────────────────
mixed_count = success_df["mixed_script"].sum()
mixed_pct   = round(mixed_count / n * 100, 1)

# ── 5. OCR Block Stats ────────────────────────────────────────────────────────
avg_blocks = round(success_df["ocr_blocks"].mean(), 1)
med_blocks = round(success_df["ocr_blocks"].median(), 1)
avg_words  = round(success_df["word_count"].mean(), 1)
avg_segs   = round(success_df["segments_found"].mean(), 1)

# ── Write summary text ────────────────────────────────────────────────────────
lines = []
lines.append("=" * 60)
lines.append("  PRESCRIPTION PIPELINE — METRICS SUMMARY")
lines.append("=" * 60)
lines.append(f"  Total processed : {total}")
lines.append(f"  Successful      : {n}")
lines.append(f"  Failed          : {total - n}")
lines.append("")
lines.append("── OCR Statistics ────────────────────────────────────────")
lines.append(f"  Avg OCR blocks per image     : {avg_blocks}")
lines.append(f"  Median OCR blocks per image  : {med_blocks}")
lines.append(f"  Avg word count per image     : {avg_words}")
lines.append(f"  OCR Confidence — High        : {ocr_high}%")
lines.append(f"  OCR Confidence — Medium      : {ocr_medium}%")
lines.append(f"  OCR Confidence — Low         : {ocr_low}%")
lines.append(f"  Mixed-script prescriptions   : {mixed_count} ({mixed_pct}%)")
lines.append("")
lines.append("── Layout Statistics ─────────────────────────────────────")
lines.append(f"  Avg segments per image       : {avg_segs}")
lines.append(f"  Layout Confidence — High     : {high_pct}%")
lines.append(f"  Layout Confidence — Medium   : {medium_pct}%")
lines.append(f"  Layout Confidence — Low      : {low_pct}%")
lines.append("")
lines.append("── Segmentation Coverage Rate ────────────────────────────")
for zone in ZONE_KEYS:
    label = ZONE_LABELS[zone]
    lines.append(f"  {label:<20} : {coverage[zone]:>5.1f}%")
lines.append("")
lines.append("── Bounding Box Assignment Rate ──────────────────────────")
for zone in ZONE_KEYS:
    label = ZONE_LABELS[zone]
    lines.append(f"  {label:<20} : {bbox_rate[zone]:>5.1f}%  (of identified)")
lines.append("=" * 60)

summary_text = "\n".join(lines)
print("\n" + summary_text)

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\n  Summary saved to: {SUMMARY}")

# ── CHART 1: Segmentation Coverage Rate ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
labels  = [ZONE_LABELS[z] for z in ZONE_KEYS]
values  = [coverage[z] for z in ZONE_KEYS]
bars    = ax.barh(labels, values, color=ZONE_COLORS, height=0.6, edgecolor="white")

for bar, val in zip(bars, values):
    ax.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{val}%", va="center", ha="left", fontsize=10, color="#374151"
    )

ax.set_xlim(0, 115)
ax.set_xlabel("Coverage (%)", labelpad=8)
ax.set_title("Segmentation Coverage Rate per Zone\n(% of prescriptions where zone was identified)",
             fontsize=12, fontweight="bold", pad=12)
ax.axvline(x=50, color="#e5e7eb", linestyle="--", linewidth=1)
ax.axvline(x=75, color="#e5e7eb", linestyle="--", linewidth=1)
plt.tight_layout()
chart1_path = OUTPUT_DIR / "chart_coverage.png"
plt.savefig(chart1_path, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart1_path.name}")

# ── CHART 2: BBox Assignment Rate ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
values2 = [bbox_rate[z] for z in ZONE_KEYS]
bars2   = ax.barh(labels, values2, color=ZONE_COLORS, height=0.6, edgecolor="white")

for bar, val in zip(bars2, values2):
    ax.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{val}%", va="center", ha="left", fontsize=10, color="#374151"
    )

ax.set_xlim(0, 115)
ax.set_xlabel("BBox Assignment Rate (%)", labelpad=8)
ax.set_title("Bounding Box Assignment Rate per Zone\n(% of identified segments where bbox was successfully matched)",
             fontsize=12, fontweight="bold", pad=12)
ax.axvline(x=50, color="#e5e7eb", linestyle="--", linewidth=1)
ax.axvline(x=75, color="#e5e7eb", linestyle="--", linewidth=1)
plt.tight_layout()
chart2_path = OUTPUT_DIR / "chart_bbox.png"
plt.savefig(chart2_path, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart2_path.name}")

# ── CHART 3: Confidence Distribution (Layout) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax_i, (title, h, m, l) in enumerate([
    ("Layout Agent Confidence", high_pct, medium_pct, low_pct),
    ("OCR Agent Confidence",    ocr_high, ocr_medium, ocr_low),
]):
    ax = axes[ax_i]
    sizes  = [h, m, l]
    clrs   = ["#22c55e", "#f59e0b", "#ef4444"]
    lbls   = [f"High\n{h}%", f"Medium\n{m}%", f"Low\n{l}%"]
    wedges, _ = ax.pie(
        sizes, colors=clrs, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax.legend(wedges, lbls, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=9, frameon=False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

plt.suptitle("Confidence Distribution across 129 Prescriptions",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
chart3_path = OUTPUT_DIR / "chart_confidence.png"
plt.savefig(chart3_path, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart3_path.name}")

# ── CHART 4: OCR Block Count Histogram ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
block_data = success_df["ocr_blocks"].dropna()

n_bins = min(20, int(block_data.max() - block_data.min()) + 1)
counts, bins, patches = ax.hist(
    block_data, bins=n_bins,
    color="#3b82f6", edgecolor="white", linewidth=0.8
)

# Colour bins by complexity
for patch, left in zip(patches, bins):
    if left < 10:
        patch.set_facecolor("#22c55e")   # simple
    elif left < 30:
        patch.set_facecolor("#f59e0b")   # medium
    else:
        patch.set_facecolor("#ef4444")   # complex

ax.axvline(x=block_data.mean(), color="#1e3a5f", linestyle="--",
           linewidth=1.5, label=f"Mean = {avg_blocks}")
ax.set_xlabel("Number of OCR Blocks Extracted", labelpad=8)
ax.set_ylabel("Number of Prescriptions", labelpad=8)
ax.set_title("Distribution of OCR Block Count\n(Green < 10 blocks | Amber 10–30 | Red > 30)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
chart4_path = OUTPUT_DIR / "chart_blocks.png"
plt.savefig(chart4_path, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart4_path.name}")

# ── CHART 5: Scatter — OCR Blocks vs Segments Found ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

conf_color_map = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
colors_scatter = success_df["layout_confidence"].map(
    lambda c: conf_color_map.get(str(c).lower(), "#94a3b8")
)

ax.scatter(
    success_df["ocr_blocks"],
    success_df["segments_found"],
    c=colors_scatter,
    alpha=0.65,
    s=40,
    edgecolors="white",
    linewidths=0.5,
)

# Trend line
if n > 2:
    z = np.polyfit(success_df["ocr_blocks"].fillna(0),
                   success_df["segments_found"].fillna(0), 1)
    p = np.poly1d(z)
    x_line = np.linspace(success_df["ocr_blocks"].min(),
                         success_df["ocr_blocks"].max(), 100)
    ax.plot(x_line, p(x_line), color="#1e3a5f",
            linestyle="--", linewidth=1.5, label="Trend")

# Legend
patches_leg = [
    mpatches.Patch(color="#22c55e", label="High confidence"),
    mpatches.Patch(color="#f59e0b", label="Medium confidence"),
    mpatches.Patch(color="#ef4444", label="Low confidence"),
]
ax.legend(handles=patches_leg, fontsize=9, frameon=False)
ax.set_xlabel("OCR Blocks Extracted", labelpad=8)
ax.set_ylabel("Segments Identified by Layout Agent", labelpad=8)
ax.set_title("OCR Block Count vs Segments Found\n(coloured by Layout Agent confidence)",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
chart5_path = OUTPUT_DIR / "chart_scatter.png"
plt.savefig(chart5_path, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {chart5_path.name}")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  All outputs saved to: {OUTPUT_DIR}")
print(f"{'='*60}\n")
