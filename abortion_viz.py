"""
DSC 106 — Data Visualization Ethics Project
Proposition: "Limited access to abortion providers leads to lower abortion rates."

Two deliberately framed visualizations of the same dataset:
  supports_proposition.png  — argues the proposition is TRUE
  refutes_proposition.png   — argues the proposition is NOT TRUE

Every persuasive / deceptive design choice is flagged [P#] or [R#]
so each decision can be explained in the write-up.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats

# ─────────────────────────────────────────────────────────
# 1.  LOAD & INSPECT
# ─────────────────────────────────────────────────────────
FILE = "GuttmacherInstituteAbortionDataByState.xlsx"
df_raw = pd.read_excel(FILE, sheet_name="Guttmacher")

print("=" * 60)
print("SHEET NAMES:", pd.ExcelFile(FILE).sheet_names)
print("\nCOLUMN NAMES:")
for c in df_raw.columns:
    print(" ", c)
print(f"\nSHAPE: {df_raw.shape}")
print("\nFIRST 5 ROWS:")
print(df_raw.head().to_string())
print("\nMISSING VALUES PER COLUMN:")
print(df_raw.isnull().sum().to_string())

# ─────────────────────────────────────────────────────────
# 2.  IDENTIFY COLUMNS BY KEYWORD MATCHING
#     (robust to minor column-name changes)
# ─────────────────────────────────────────────────────────
def find_col(df, *keywords):
    """Return first column whose name contains ALL keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for col in df.columns:
        if all(k in col.lower() for k in kws):
            return col
    return None

COL_STATE      = find_col(df_raw, "state")
COL_CLINIC_PCT = find_col(df_raw, "counties", "clinic", "2020")
COL_RATE_OCC   = find_col(df_raw, "1,000", "occurrence", "2020")
COL_RATE_RES   = find_col(df_raw, "1,000", "residence", "2020")

print("\n" + "=" * 60)
print("COLUMNS SELECTED:")
print(f"  State              : {COL_STATE}")
print(f"  % counties w/o clinic : {COL_CLINIC_PCT}")
print(f"  Abortion rate (occ): {COL_RATE_OCC}")
print(f"  Abortion rate (res): {COL_RATE_RES}")

# ─────────────────────────────────────────────────────────
# 3.  CLEAN — full dataset (all 51 rows including DC)
#     Text sentinels ("unavailable", "nr", "<1,000") → NaN
# ─────────────────────────────────────────────────────────
df_full = df_raw[[COL_STATE, COL_CLINIC_PCT, COL_RATE_OCC, COL_RATE_RES]].copy()
df_full.columns = ["state", "pct_no_clinic", "rate_occ", "rate_res"]

for col in ["pct_no_clinic", "rate_occ", "rate_res"]:
    df_full[col] = pd.to_numeric(df_full[col], errors="coerce")

# ─── Dataset for VIZ 1: drop DC, use occurrence rate ─────
# [P-DC] Dropping DC (pct_no_clinic=0, rate_occ≈26) removes a high-access /
#         high-rate outlier that would flatten the regression line and weaken
#         the case for the proposition.
df_v1 = (df_full[df_full["state"] != "District of Columbia"]
         .dropna(subset=["pct_no_clinic", "rate_occ"])
         .reset_index(drop=True))

# ─── Dataset for VIZ 2: KEEP all states, use residence rate ──
# [R-DC] Keeping DC (rate_res≈26.1, High Access) dramatically extends the
#         High Access group's spread, making all three groups look similarly
#         variable and obscuring the between-group pattern.
df_v2 = (df_full
         .dropna(subset=["pct_no_clinic", "rate_res"])
         .reset_index(drop=True))

# ─────────────────────────────────────────────────────────
# 4.  SUMMARY STATISTICS & CORRELATIONS
# ─────────────────────────────────────────────────────────
r_occ, p_occ = stats.pearsonr(df_v1["pct_no_clinic"], df_v1["rate_occ"])
r_res, p_res = stats.pearsonr(df_v2["pct_no_clinic"], df_v2["rate_res"])

print("\n" + "=" * 60)
print("CORRELATION  (% counties w/o clinic vs. abortion rate)")
print(f"  By occurrence [Viz 1, DC removed] : r = {r_occ:.3f},  p = {p_occ:.4f}")
print(f"  By residence  [Viz 2, all states] : r = {r_res:.3f},  p = {p_res:.4f}")

print("\nSUMMARY — occurrence rate (Viz 1 dataset):")
print(df_v1[["pct_no_clinic", "rate_occ"]].describe().round(2).to_string())
print("\nSUMMARY — residence rate (Viz 2 dataset):")
print(df_v2[["pct_no_clinic", "rate_res"]].describe().round(2).to_string())


# ══════════════════════════════════════════════════════════
#  VISUALIZATION 1  —  SUPPORTS THE PROPOSITION
#  "States With Fewer Providers Show Lower Abortion Rates"
# ══════════════════════════════════════════════════════════
#
#  [P1] METRIC CHOICE — "by occurrence" rate counts only abortions that
#       physically happen inside the state. Restricted states look like they
#       have dramatically fewer abortions even when their residents simply
#       travel out of state. "By residence" (the fairer metric) would show a
#       weaker trend because it follows the woman, not the clinic.
#
#  [P2] TRUNCATED Y-AXIS — starts at 2, not 0. Steepens the apparent slope
#       of the regression line without changing a single data value.
#
#  [P3] RED COLOR GRADIENT — encodes restriction level with an emotionally
#       activating color; darker red = more counties without a clinic.
#
#  [P4] SELECTIVE ANNOTATIONS — only label states that sit at the extreme
#       corners of the trend (high restriction / low rate  OR  low restriction
#       / high rate). Outliers that contradict the trend are left anonymous.
#
#  [P5] CAUSAL TITLE LANGUAGE — "Show" implies causation rather than correlation.

fig1, ax1 = plt.subplots(figsize=(9, 6))

norm  = Normalize(vmin=df_v1["pct_no_clinic"].min(),
                  vmax=df_v1["pct_no_clinic"].max())

ax1.scatter(
    df_v1["pct_no_clinic"], df_v1["rate_occ"],
    c=df_v1["pct_no_clinic"], cmap="Reds", norm=norm,   # [P3]
    s=80, edgecolors="#777777", linewidths=0.4, zorder=3,
)

# OLS regression line
m, b, *_ = stats.linregress(df_v1["pct_no_clinic"], df_v1["rate_occ"])
x_line = np.linspace(df_v1["pct_no_clinic"].min(), df_v1["pct_no_clinic"].max(), 300)
ax1.plot(x_line, m * x_line + b, color="#c0392b", linewidth=2.4,
         label=f"Trend  (r = {r_occ:.2f})", zorder=4)

# [P2] Truncated y-axis
ax1.set_ylim(2, df_v1["rate_occ"].max() + 1.8)

# [P4] Selective annotations — only narrative-reinforcing states.
#      xytext is ABSOLUTE (x, y) so positions are predictable regardless of dot location.
#      White bbox prevents text clashing with dots or the trend line.
#      Curved arrows (rad≠0) route around the trend line where needed.
#      Wyoming removed — hidden inside the dense lower-right cluster.
annotations = {
    # Low-restriction / high-rate: labels in upper-left clear of trend line
    "New York":    dict(xytext=(40,  30.5), rad= 0.0),
    "New Jersey":  dict(xytext=(40,  27.8), rad= 0.0),
    "California":  dict(xytext=(14,  22.5), rad= 0.0),
    # High-restriction / low-rate: labels pulled away from the dense cluster
    # Mississippi and Arkansas sit at x≈99; anchor text to the left and route
    # arrows upward so they don't overlap the trend line or each other.
    "Mississippi": dict(xytext=(55,  14.5), rad=-0.3),
    "Arkansas":    dict(xytext=(55,   4.5), rad= 0.25),
}
for state, cfg in annotations.items():
    row = df_v1[df_v1["state"] == state]
    if row.empty:
        continue
    xv  = float(row["pct_no_clinic"].iloc[0])
    yv  = float(row["rate_occ"].iloc[0])
    xtxt, ytxt = cfg["xytext"]
    ax1.annotate(
        state,
        xy=(xv, yv),
        xytext=(xtxt, ytxt),
        fontsize=8, color="#1a1a1a",
        arrowprops=dict(
            arrowstyle="-",
            color="#888888",
            lw=0.9,
            connectionstyle=f"arc3,rad={cfg['rad']}",
        ),
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  alpha=0.85, edgecolor="none"),
        clip_on=False,
    )

cb = fig1.colorbar(ScalarMappable(norm=norm, cmap="Reds"), ax=ax1, pad=0.02)
cb.set_label("% of Counties Without an Abortion Clinic", fontsize=9)

ax1.set_xlabel("% of Counties Without an Abortion Clinic (2020)", fontsize=11)
ax1.set_ylabel("Abortion Rate per 1,000 Women\n(by State of Occurrence, 2020)", fontsize=10)
ax1.set_title("States With Fewer Providers Show Lower Abortion Rates",  # [P5]
              fontsize=13, fontweight="bold", pad=12)
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.set_facecolor("#fdf5f5")   # [P3] subtle warm tint

plt.tight_layout()
fig1.savefig("supports_proposition.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nSaved: supports_proposition.png")


# ══════════════════════════════════════════════════════════
#  VISUALIZATION 2  —  REFUTES THE PROPOSITION
#  "Abortion Rates Show No Clear Pattern Across Access Levels"
# ══════════════════════════════════════════════════════════
#
#  [R1] COARSE BINNING — three wide categories (0–49 / 50–79 / 80–100%)
#       collapse the continuous negative-r relationship into groups with huge
#       within-group variance, making the between-group signal look like noise.
#
#  [R2] STRIP / JITTER PLOT instead of scatterplot — hides the continuous
#       correlation by design; the reader sees clouds of dots, not a trend line.
#
#  [R3] HEAVY VERTICAL JITTER — dots spread across the full band height, causing
#       adjacent groups to visually overlap and erasing any sense of separation.
#
#  [R4] EXTENDED X-AXIS (0 → 40, data max ≈ 29) — compresses the visible
#       distance between group means, making differences look proportionally tiny.
#
#  [R5] DESATURATED, LOW-CONTRAST COLORS — nearly identical hues reduce the
#       visual salience of group membership; the chart reads as one gray mass.
#
#  [R6] NO MEAN MARKER displayed — without an explicit central tendency indicator
#       the reader cannot easily compare groups and must infer from the cloud.
#
#  [R7] SORTING WITHIN GROUPS BY ABORTION RATE (not by access level) —
#       dots are ordered so that within each band high-rate and low-rate states
#       are interleaved, preventing any left-to-right gradient from being visible.
#
#  [R8] ALL STATES INCLUDED (DC kept) — DC sits in High Access with rate_res≈26,
#       which dramatically widens that group's horizontal spread and makes High
#       Access look just as variable as Low Access.
#
#  [R9] SOFTER TITLE — "No Clear Pattern" frames genuine differences as noise.
#
#  [R10] SELECTIVE OUTLIER REMOVAL IN HIGH ACCESS GROUP ONLY — NJ, NY, and DC
#        are dropped from the High Access band before plotting. This is presented
#        as routine outlier exclusion but is applied only to the one group whose
#        mean needs to fall. It drops the High Access mean from 19.6 → 14.8,
#        making all three group means look nearly identical on a 0–100 axis.
#        The Low Access group keeps its own high-rate outliers untouched.

# [R10] High outliers removed from High Access — drops mean 19.6 → 14.8
HIGH_ACCESS_EXCLUDE = {"New Jersey", "New York", "District of Columbia"}
# [R12] Low outliers removed from Low Access — raises mean 10.3 → 11.0
#       Mirrors R10: both manipulations push the two group means toward each other.
LOW_ACCESS_EXCLUDE  = {"South Dakota", "Utah", "Nebraska", "West Virginia", "Iowa"}

# [R1] Two coarse bins — cutoff at 50%
BINS   = [0, 49, 100]
LABELS = ["High Access (0–49%)",
          "Low Access (50–100%)"]

df_v2 = df_v2.copy()
df_v2["access_group"] = pd.cut(
    df_v2["pct_no_clinic"], bins=BINS, labels=LABELS, include_lowest=True
)
df_v2 = df_v2.dropna(subset=["access_group"])

# [R10] Remove high outliers from High Access.
high_access_mask = (df_v2["access_group"] == LABELS[0]) & (df_v2["state"].isin(HIGH_ACCESS_EXCLUDE))
# [R12] Remove low outliers from Low Access.
low_access_mask  = (df_v2["access_group"] == LABELS[1]) & (df_v2["state"].isin(LOW_ACCESS_EXCLUDE))
df_v2 = df_v2[~high_access_mask & ~low_access_mask].copy()

# [R7] Sort by abortion rate within each group.
df_v2 = df_v2.sort_values(["access_group", "rate_res"]).reset_index(drop=True)

# Bar positions: High Access at top (1), Low Access at bottom (0)
# Increased vertical spacing so the bars feel less directly comparable
bar_pos = {LABELS[0]: 1, LABELS[1]: 0}

# [R5] Muted, desaturated palette — low contrast, closer to gray
bar_color = {LABELS[0]: "#8FAFC7",   # muted gray-blue  — High Access
             LABELS[1]: "#C9AA85"}   # muted tan/sand   — Low Access

# Compute group mean and standard deviation for error bars
group_means = df_v2.groupby("access_group", observed=True)["rate_res"].mean()
group_stds  = df_v2.groupby("access_group", observed=True)["rate_res"].std()

fig2, ax2 = plt.subplots(figsize=(10, 4.5))
ax2.set_facecolor("white")

for label, ypos in bar_pos.items():
    mean_val = group_means[label]
    std_val  = group_stds[label]
    color    = bar_color[label]

    # [R_THIN] Thinner bars make side-by-side comparison less visceral
    ax2.barh(ypos, mean_val, height=0.28,
             color=color, alpha=0.88, zorder=2)

    # [R_ERR] Error bars ± 1 SD — suggest group distributions overlap,
    #         implying the difference is not statistically meaningful
    ax2.errorbar(
        mean_val, ypos,
        xerr=std_val,
        fmt="none",
        ecolor="#999999",
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=3,
    )

    # [R_LABEL] Subtle gray mean label — placed past the error bar cap so it
    #           doesn't overlap with the whisker
    ax2.text(
        mean_val + std_val + 1.5, ypos,
        f"{mean_val:.1f}",
        va="center", ha="left",
        fontsize=9, color="#999999",
        zorder=4,
    )

# Y-axis group labels
ax2.set_yticks(list(bar_pos.values()))
ax2.set_yticklabels(list(bar_pos.keys()), fontsize=11)

# [R4] X-axis 0–100 — compresses all bars into left ~15% of chart
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.5, 1.5)

# Light vertical gridlines every 20 units — subtle, reinforce empty space
ax2.set_xticks(range(0, 101, 20))
ax2.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

ax2.set_xlabel("Average Abortion Rate per 1,000 Women (by State of Residence, 2020)",
               fontsize=11)

# [R9] Bold suptitle + lighter ax subtitle — kept on separate layers so they
#      never collide regardless of figure size
fig2.suptitle(
    "Differences in Abortion Rates Across Access Levels Are Nearly Negligible",
    fontsize=13, fontweight="bold", y=0.98,
)
ax2.set_title(
    "Group averages show only minor variation",
    fontsize=9.5, color="#777777", style="italic", pad=4,
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.annotate(
    "Rate by state of residence captures women who travel out of state for care.  "
    "Error bars show ±1 SD.",
    xy=(0.01, 0.02), xycoords="axes fraction",
    fontsize=7.5, color="#aaaaaa", style="italic",
)

plt.tight_layout()
fig2.savefig("refutes_proposition.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: refutes_proposition.png")

print("\nDone.")
