# plot_prune_compare.py — paper figure for the penalty-vs-threshold experiment.
#
# Renders JournalPaper_tex/src/pics/prune_compare.pdf from the CSVs produced by
# tsc_prune_compare.py (strong + symmetric runs) and tsc_state_rank.py:
#
#   panels 1-7: test accuracy vs readout sparsity, SmoothL1-ESN vs the strong
#               L2+soft-threshold baseline (deployment comparison), one panel per
#               dataset, ordered by the effective rank r99 of the state matrix;
#   panel 8   : the support-quality summary — Δ(SL1−L2) at 90% sparsity with BOTH
#               readouts refit on their supports, plotted against r99.  This is
#               the "advantage tracks the effective dimension" claim in one mark.
#
# Colors: Okabe-Ito blue/vermillion (CVD-safe) + redundant encoding (solid vs
# dashed, distinct markers) so identity survives grayscale printing.
#
# Usage:  python plot_prune_compare.py
#         python plot_prune_compare.py --out some/other.pdf

import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRONG_CSV = "./result/tsc_prune_compare_strong_summary.csv"
SYM_CSV    = "./result/tsc_prune_compare_symmetric_summary.csv"
RANK_CSV   = "./result/tsc_state_rank.csv"
OUT_PDF    = "./JournalPaper_tex/src/pics/prune_compare.pdf"

# Paper display names; panel order = r99 descending (matches the paper table).
PAPER_NAME = {
    "har": "UCI HAR", "Yoga": "Yoga", "jpv": "JapaneseVowels",
    "char": "Character", "ECG5000": "ECG5000",
    "Strawberry": "Strawberry", "DistalPhalanxOutlineCorrect": "Distal.",
}

C_SL1, C_L2 = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion
INK, MUTED  = "#333333", "#767676"


def _load_summary(path):
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[r["dataset"]][float(r["target_sparsity"])] = (
            float(r["sl1_acc"]), float(r["l2_acc"]))
    return d


def _load_rank(path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        acc[r["dataset"]].append(float(r["rank99"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(out_pdf):
    strong = _load_summary(STRONG_CSV)
    sym    = _load_summary(SYM_CSV)
    r99    = _load_rank(RANK_CSV)
    order  = sorted(PAPER_NAME, key=lambda k: -r99[k])

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4))
    axes = axes.ravel()

    for ax, ds in zip(axes[:7], order):
        qs   = sorted(strong[ds])
        sl1  = [strong[ds][q][0] for q in qs]
        l2   = [strong[ds][q][1] for q in qs]
        ax.plot(qs, l2, "--", color=C_L2, lw=1.4, marker="s", ms=2.6,
                label="L2+ST (strong)")
        ax.plot(qs, sl1, "-", color=C_SL1, lw=1.4, marker="o", ms=2.6,
                label="SmoothL1-ESN")
        ax.set_title(f"{PAPER_NAME[ds]}  ($r_{{99}}={r99[ds]:.0f}$)", pad=3)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(48, 100)
        ax.set_xticks([50, 60, 70, 80, 90, 100])

    # Panel 8 — support quality vs effective rank (both arms refit, 90% sparsity)
    ax = axes[7]
    xs  = [r99[d] for d in order]
    ys  = [sym[d][90.0][0] - sym[d][90.0][1] for d in order]
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.scatter(xs, ys, s=26, color=C_SL1, zorder=3)
    off = {"har": (-44, -3), "Yoga": (-32, -2), "jpv": (-40, -12),
           "char": (5, 3), "ECG5000": (5, -8),
           "Strawberry": (5, -3), "DistalPhalanxOutlineCorrect": (-10, 6)}
    for d, x, y in zip(order, xs, ys):
        ax.annotate(PAPER_NAME[d], (x, y), textcoords="offset points",
                    xytext=off[d], fontsize=6.6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 80)
    ax.set_ylim(-4, 8.5)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("effective rank $r_{99}$ of states (log)")
    ax.set_ylabel(r"$\Delta$ support (pt)")
    ax.set_title("Support quality vs. state dimension", pad=3)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for ax in axes[:7]:
        ax.set_xlabel("readout sparsity (%)")
    axes[0].set_ylabel("test accuracy (%)")
    axes[4].set_ylabel("test accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"figure -> {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_PDF)
    main(p.parse_args().out)
