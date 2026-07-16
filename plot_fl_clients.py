# plot_fl_clients.py — paper figure for the federated client-count sensitivity.
#
# Renders JournalPaper_tex/src/pics/fl_clients_sweep.pdf from the JSONs produced
# by tsc_fl_sensitivity.py --param n_clients (one JSON per dataset in
# result/pic/). One dual-axis panel per dataset, matching the style of the
# centralized ablation figure: test accuracy (left axis, blue, solid) and
# readout sparsity (right axis, vermillion, dashed) versus the number of
# clients K, mean ± std over seeds, at the paper's per-dataset tau operating
# point (Table 5) with 50 communication rounds.
#
# Usage:  python plot_fl_clients.py
#         python plot_fl_clients.py --out some/other.pdf

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR  = "./result/pic_10seed_final"
OUT_PDF = "./JournalPaper_tex/src/pics/fl_clients_sweep.pdf"

# Paper display names; panel order matches the paper tables.
PAPER_NAME = {
    "har": "UCI HAR", "char": "Character", "jpv": "JapaneseVowels",
    "ECG5000": "ECG5000", "DistalPhalanxOutlineCorrect": "Distal.",
    "Yoga": "Yoga", "Strawberry": "Strawberry",
}
ORDER = ["har", "char", "jpv", "ECG5000",
         "DistalPhalanxOutlineCorrect", "Yoga", "Strawberry"]

C_ACC, C_SP = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion
INK, MUTED  = "#333333", "#767676"


def _load(ds):
    path = os.path.join(IN_DIR, f"fl_sensitivity_{ds}_sl1_n_clients.json")
    with open(path) as f:
        payload = json.load(f)
    rs = payload["results"]
    return {
        "x":      np.array([r["param_value"]       for r in rs]),
        "acc":    np.array([r["mean_val_acc"]      for r in rs]),
        "acc_s":  np.array([r["std_val_acc"]       for r in rs]),
        "sp":     np.array([r["mean_val_sparsity"] for r in rs]),
        "sp_s":   np.array([r["std_val_sparsity"]  for r in rs]),
        "tau":    payload["settings"].get("thres"),
    }


def main(out_pdf):
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

    for ax, ds in zip(axes[:7], ORDER):
        d = _load(ds)
        ax2 = ax.twinx()

        ax.plot(d["x"], d["acc"], "-", color=C_ACC, lw=1.4, marker="o",
                ms=2.6, zorder=3, label="test accuracy (%)")
        ax.fill_between(d["x"], d["acc"] - d["acc_s"], d["acc"] + d["acc_s"],
                        color=C_ACC, alpha=0.15, zorder=2)

        ax2.plot(d["x"], d["sp"], "--", color=C_SP, lw=1.4, marker="s",
                 ms=2.6, zorder=3, label="readout sparsity (%)")
        ax2.fill_between(d["x"], d["sp"] - d["sp_s"], d["sp"] + d["sp_s"],
                         color=C_SP, alpha=0.15, zorder=2)

        # K=5 is the operating point used in the main-text FL experiments.
        ax.axvline(5, color=MUTED, lw=0.8, linestyle=":", zorder=1)

        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20, 50])
        ax.set_xticklabels(["2", "3", "5", "10", "20", "50"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 100)
        ax.tick_params(axis="y", labelcolor=C_ACC)
        ax2.tick_params(axis="y", labelcolor=C_SP)
        if ax is not axes[3]:                   # right-edge panels keep labels
            ax2.set_yticklabels([])
        ax.set_title(rf"{PAPER_NAME[ds]}  ($\tau={d['tau']:g}$)", pad=3)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)

    # Last cell carries the legend only.
    axes[7].axis("off")
    handles = [
        plt.Line2D([], [], color=C_ACC, lw=1.4, marker="o", ms=3,
                   label="test accuracy (%, left axis)"),
        plt.Line2D([], [], color=C_SP, lw=1.4, ls="--", marker="s", ms=3,
                   label="readout sparsity (%, right axis)"),
        plt.Line2D([], [], color=MUTED, lw=0.8, ls=":",
                   label="$K=5$ (main-text setting)"),
    ]
    axes[7].legend(handles=handles, loc="center", frameon=False, fontsize=8)

    for ax in axes[4:7]:
        ax.set_xlabel("number of clients $K$ (log)")
    axes[0].set_xlabel("")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"figure -> {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_PDF)
    main(p.parse_args().out)
