# plot_lorenz.py — regenerates the three Lorenz attractor panels (Fig. 2).
#
#   lorenz_l2.pdf, lorenz_lasso.pdf, lorenz_smoothl1.pdf
#
# Each panel overlays the autonomous rollout (black dashed) on the true Lorenz
# attractor (red solid), from the saved test trajectories of the accuracy-
# selected readouts in result/regression_fig2/ (the run behind Table 3). The
# embedded per-panel title is intentionally dropped: the LaTeX sub-caption
# ((a) L2 / (b) L1 / (c) Smooth-L1) and the figure caption already name each
# panel, and the old title carried the pre-rename "SmoothL1-ESN" label.
#
# Usage:  python plot_lorenz.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_DIR  = "./result/regression_fig2"
OUT_DIR = "./JournalPaper_tex/src/pics"

# reg_type -> output stem (L1 is written as "lasso" in the paper filenames).
PANELS = [("l2", "lorenz_l2"), ("l1", "lorenz_lasso"), ("sl1", "lorenz_smoothl1")]

C_TRUE, C_PRED = "red", "black"


def _panel(reg_type, stem):
    trues = np.load(os.path.join(IN_DIR, f"lorenz_{reg_type}_trues.npy"))
    preds = np.load(os.path.join(IN_DIR, f"lorenz_{reg_type}_preds.npy"))

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(trues[:, 0], trues[:, 1], trues[:, 2],
            color=C_TRUE, lw=1.0, label="True")
    ax.plot(preds[:, 0], preds[:, 1], preds[:, 2],
            color=C_PRED, lw=1.0, ls="--", label="Predict")

    ax.set_xlabel("X", labelpad=-2)
    ax.set_ylabel("Y", labelpad=-2)
    ax.set_zlabel("Z", labelpad=-2)
    ax.tick_params(pad=-1)
    ax.view_init(elev=20, azim=-55)
    ax.legend(loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.08), handlelength=1.8, columnspacing=1.4)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"figure -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    for reg_type, stem in PANELS:
        _panel(reg_type, stem)
