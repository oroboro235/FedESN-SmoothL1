# plot_ablation.py — regenerates the one-at-a-time ablation figure (Fig. 6).
#
#   ablation_har.pdf   2x2 dual-axis panels, one per swept hyperparameter:
#       (a) thres (tau)  (b) reg_param (lambda)  (c) units (N_R)  (d) alpha_multiplier (kappa)
#
# Each panel plots validation accuracy (left axis, blue) and readout sparsity
# (right axis, vermillion) against the swept parameter, with +/-1 std shaded
# bands over seeds. Inputs are the per-parameter sweep JSONs produced by
#   tsc_centralized_sensitivity.py --dataset har --param <p>
# read from result/pic_10seed/ (10-seed re-run; falls back to result/pic/).
#
# Usage:  python plot_ablation.py

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PDF = "./JournalPaper_tex/src/pics/ablation_har.pdf"
IN_DIRS = ["./result/pic_10seed", "./result/pic"]   # first hit wins

TEXTWIDTH_IN = 507.51 / 72.27

C_ACC, C_SP = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion
INK, MUTED, GRID = "#333333", "#767676", "#dddddd"

# (param file stem, axis label, log-x) for panels (a)-(d), in order.
PANELS = [
    ("thres",            r"$\tau$",   True,  "(a) Shrinkage threshold $\\tau$"),
    ("reg_param",        r"$\lambda$", True, "(b) Regularization $\\lambda$"),
    ("units",            r"$N_R$",    False, "(c) Reservoir size $N_R$"),
    ("alpha_multiplier", r"$\kappa$", False, "(d) Continuation factor $\\kappa$"),
]


def _load(param):
    for d in IN_DIRS:
        p = os.path.join(d, f"sensitivity_har_{param}.json")
        if os.path.exists(p):
            rs = json.load(open(p))["results"]
            x     = np.array([r["param_value"]       for r in rs], float)
            acc   = np.array([r["mean_val_acc"]      for r in rs], float)
            acc_s = np.array([r["std_val_acc"]       for r in rs], float)
            sp    = np.array([r["mean_val_sparsity"] for r in rs], float)
            sp_s  = np.array([r["std_val_sparsity"]  for r in rs], float)
            o = np.argsort(x)
            return x[o], acc[o], acc_s[o], sp[o], sp_s[o]
    raise FileNotFoundError(f"sensitivity_har_{param}.json in {IN_DIRS}")


def main():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.62))
    axes = axes.ravel()

    for ax, (param, xlabel, logx, title) in zip(axes, PANELS):
        x, acc, acc_s, sp, sp_s = _load(param)
        ax2 = ax.twinx()

        ax.fill_between(x, acc - acc_s, acc + acc_s, color=C_ACC, alpha=0.15, lw=0)
        ax.plot(x, acc, "-o", color=C_ACC, lw=1.5, ms=3, zorder=3)

        ax2.fill_between(x, sp - sp_s, sp + sp_s, color=C_SP, alpha=0.15, lw=0)
        ax2.plot(x, sp, "--s", color=C_SP, lw=1.5, ms=3, zorder=3)

        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 100)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", labelcolor=C_ACC)
        ax2.tick_params(axis="y", labelcolor=C_SP)
        ax.set_ylabel("Val. accuracy (%)", color=C_ACC)
        ax2.set_ylabel("Sparsity (%)", color=C_SP)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)

    # Shared legend along the top.
    handles = [
        plt.Line2D([], [], color=C_ACC, lw=1.5, marker="o", ms=3,
                   label="validation accuracy (left axis)"),
        plt.Line2D([], [], color=C_SP, lw=1.5, ls="--", marker="s", ms=3,
                   label="readout sparsity (right axis)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_PDF.replace(".pdf", f".{ext}"), bbox_inches="tight", dpi=200)
        print(f"figure -> {OUT_PDF.replace('.pdf', f'.{ext}')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
