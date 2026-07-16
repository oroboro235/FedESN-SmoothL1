# plot_paper_figs.py — regenerates the three main-text result figures.
#
#   acc_vs_sparsity_thres_sweep.pdf     centralized accuracy vs readout sparsity
#   fl_acc_vs_sparsity_thres_sweep.pdf  the same curve in the federated setting
#   accuracy_vs_rounds.pdf              federated accuracy vs communication round
#
# Inputs (all committed under result/pic/):
#   sensitivity_<ds>_thres.json          <- tsc_centralized_sensitivity.py --param thres
#   fl_sensitivity_<ds>_sl1_thres.json   <- tsc_fl_sensitivity.py --param thres
#   acc_vs_rounds_data.json              <- federated eval, one entry per dataset
#
# The figures carry no embedded title: the LaTeX caption is the title, and a
# second one inside the axes is redundant (and used to overflow the canvas).
#
# Usage:  python plot_paper_figs.py            # all three
#         python plot_paper_figs.py --only rounds

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# rounds_evolution.json (the accuracy/sparsity-vs-rounds data) lives here.
IN_DIR  = "./result/pic_10seed"
# Federated sparsity sweep reported at the final round (--report_round final,
# the fixed-budget protocol). Round-50 reporting is what the tables use.
FL_DIR  = "./result/pic_10seed_final"
# Centralized sparsity sweep evaluated on the test set (--eval_split test), kept
# separate from the validation-split sweep used by the ablation figure.
CENTRAL_TEST_DIR = "./result/pic_10seed_test"
OUT_DIR = "./JournalPaper_tex/src/pics"

# Wiley WileyNJDv5 [LATO2COL]: \textwidth = 507.51pt, \columnwidth = 247.75pt.
# Every figure below is authored at its *printed* size and included at
# width=\textwidth, so LaTeX never rescales it and the point sizes set here are
# the point sizes on the page.
TEXTWIDTH_IN = 507.51 / 72.27

ORDER = ["har", "char", "jpv", "ECG5000",
         "DistalPhalanxOutlineCorrect", "Yoga", "Strawberry"]
PAPER_NAME = {
    "har": "UCI HAR", "char": "Character", "jpv": "JapaneseVowels",
    "ECG5000": "ECG5000", "DistalPhalanxOutlineCorrect": "Distal.",
    "Yoga": "Yoga", "Strawberry": "Strawberry",
}

# Categorical hues, assigned to datasets in fixed ORDER and never cycled.
# Okabe-Ito with the low-contrast yellow swapped for a violet; validated
# colorblind-safe (worst adjacent pair dE 20.7 under protanopia).
SERIES_COLOR = dict(zip(ORDER, [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#785EF0",
]))

INK, MUTED, GRID = "#333333", "#767676", "#dddddd"


def _style():
    # Body text is ~9.5pt; figure text sits just under it and is never rescaled.
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3
    })


def _finish(fig, ax, out_stem):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Legend outside the axes: with seven series there is no interior gap
    # wide enough to hold it without covering data.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, handlelength=1.8, borderaxespad=0)
    # Explicit margins rather than a tight bbox: a tight bbox expands the canvas
    # around the outside legend, so the saved page would be wider than
    # \textwidth and LaTeX would silently shrink the type.
    fig.subplots_adjust(left=0.075, right=0.80, top=0.97, bottom=0.16)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{out_stem}.{ext}")
        fig.savefig(path, dpi=200)
        print(f"figure -> {path}")
    plt.close(fig)


def _load_sweep(path):
    """Return (sparsity, accuracy, sparsity_std, accuracy_std) sorted by sparsity."""
    with open(path) as f:
        rs = json.load(f)["results"]
    sp    = np.array([r["mean_val_sparsity"] for r in rs])
    acc   = np.array([r["mean_val_acc"]      for r in rs])
    sp_s  = np.array([r["std_val_sparsity"]  for r in rs])
    acc_s = np.array([r["std_val_acc"]       for r in rs])
    o = np.argsort(sp)
    return sp[o], acc[o], sp_s[o], acc_s[o]


def sparsity_sweep(federated):
    """Accuracy against readout sparsity, one curve per dataset.

    Both figures report *test* accuracy so they match the operating-point
    tables. The federated sweep already evaluates on test; the centralized
    sweep is the --eval_split test run, written to a separate directory.

    Error bars span +/-1 std over seeds in BOTH axes. The horizontal (sparsity)
    bars matter as much as the vertical ones: in the federated case the readout
    reaches a reproducible sparsity level but not a reproducible support, so the
    sparsity std is large and must be visible rather than hidden behind the mean.
    """
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42))

    for ds in ORDER:
        if federated:
            path = os.path.join(FL_DIR, f"fl_sensitivity_{ds}_sl1_thres.json")
        else:
            path = os.path.join(CENTRAL_TEST_DIR, f"sensitivity_{ds}_thres.json")
        sp, acc, sp_s, acc_s = _load_sweep(path)
        c = SERIES_COLOR[ds]
        # Vertical (accuracy) std band, matching the accuracy-vs-rounds figure.
        ax.fill_between(sp, acc - acc_s, acc + acc_s, color=c, alpha=0.15, lw=0)
        ax.plot(sp, acc, "-o", color=c, lw=1.6, ms=3.5,
                markeredgecolor="white", markeredgewidth=0.5, label=PAPER_NAME[ds])

    ax.set_xlabel("Readout sparsity (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(-2, 102)
    stem = ("fl_acc_vs_sparsity_thres_sweep" if federated
            else "acc_vs_sparsity_thres_sweep")
    _finish(fig, ax, stem)


def accuracy_vs_rounds():
    """Two panels against communication round: test accuracy (left) and readout
    sparsity (right), mean +/-1 std over ten seeds. The pairing shows the
    mechanism: accuracy plateaus within a few rounds (the cross-entropy fit is
    reached almost immediately), while sparsity keeps accruing over more rounds
    as the continuation schedule sharpens the penalty toward L1.
    """
    with open(os.path.join(IN_DIR, "rounds_evolution.json")) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.34))

    for ds in ORDER:
        d = data[ds]
        x = np.asarray(d["rounds"], float)
        c = SERIES_COLOR[ds]
        am, asd = np.asarray(d["acc_mean"]), np.asarray(d["acc_std"])
        sm, ssd = np.asarray(d["sp_mean"]),  np.asarray(d["sp_std"])
        ax1.fill_between(x, am - asd, am + asd, color=c, alpha=0.15, lw=0)
        ax1.plot(x, am, "-", color=c, lw=1.5, label=PAPER_NAME[ds])
        ax2.fill_between(x, sm - ssd, sm + ssd, color=c, alpha=0.15, lw=0)
        ax2.plot(x, sm, "-", color=c, lw=1.5)

    for ax, ylab, title in [(ax1, "Test accuracy (%)",     "(a) Accuracy"),
                            (ax2, "Readout sparsity (%)", "(b) Sparsity")]:
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_xlim(1, x.max())
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax1.legend(loc="upper center", bbox_to_anchor=(1.12, -0.22), ncol=4,
               frameon=False, columnspacing=1.4, handlelength=1.6)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.32, wspace=0.2)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"accuracy_vs_rounds.{ext}")
        fig.savefig(path, dpi=200)
        print(f"figure -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["central", "fl", "rounds"],
                   help="render a single figure instead of all three")
    a = p.parse_args()

    _style()
    if a.only in (None, "central"):
        sparsity_sweep(federated=False)
    if a.only in (None, "fl"):
        sparsity_sweep(federated=True)
    if a.only in (None, "rounds"):
        accuracy_vs_rounds()
