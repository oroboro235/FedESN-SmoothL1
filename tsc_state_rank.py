# tsc_state_rank.py — effective rank of the reservoir state matrix.
#
# Question this script answers:
#   The readout has N_R = 500 coordinates per class.  How many of them does the
#   reservoir actually *use*?  I.e. what is the dimension of the subspace the
#   collected states X live on?
#
# Why it matters (paper Section 5 / Section 3.3):
#   If the states span only a few dozen directions, the readout is over-complete
#   by one to two orders of magnitude, and a sparse readout is not a lucky
#   accident but the structurally correct model: the reservoir is a large random
#   feature pool whose *informative* content is low-dimensional.  This is the
#   empirical basis for the redundancy argument in Section 3.3 and it explains
#   why any reasonable pruning survives moderate sparsity — the question is not
#   *whether* one can prune but *which* coordinates to keep.
#
# Measures, on the singular values s_i of the (n_samples x N_R) state matrix
# (energy e_i = s_i^2):
#   • rank@90 / rank@99 — smallest k with sum_{i<=k} e_i >= 0.90 / 0.99 of total.
#   • participation ratio  PR = (sum e_i)^2 / sum e_i^2 — a smooth, threshold-free
#     effective-rank measure (PR = 1 for a rank-1 spectrum, = N for a flat one).
#
# The states are collected exactly as the classification pipeline collects them
# (last hidden state per sequence, standardized from the train split), with the
# per-dataset tuned reservoir, so the numbers describe the readouts the paper
# actually trains.
#
# Usage:
#   python tsc_state_rank.py
#   python tsc_state_rank.py --datasets har jpv --seeds 1 2 3
#   python tsc_state_rank.py --latex           # emit the LaTeX table body

import argparse

import numpy as np
from reservoirpy.nodes import Reservoir

import config
from data_loader import standardize
from utils import init_csv, append_csv_row
from tsc_centralized_search import (
    _load_dataset_meta, _load_dataset, _stratified_split,
    _extract_states, _ensure_3d,
)
from tsc_prune_compare import _load_reservoir_hparams

_CSV_FIELDS = ["dataset", "seed", "n_train", "units",
               "rank90", "rank99", "part_ratio"]

# Display names matching the paper's tables.
_PAPER_NAME = {
    "har":                         "UCI HAR",
    "char":                        "Character",
    "jpv":                         "JapaneseVowels",
    "ECG5000":                     "ECG5000",
    "DistalPhalanxOutlineCorrect": "Distal.",
    "Yoga":                        "Yoga",
    "Strawberry":                  "Strawberry",
}


def state_spectrum(dataset: str, seed: int, reservoir_source: str,
                   use_cache: bool) -> dict:
    """Collect train states with the tuned reservoir and summarise the spectrum."""
    rc = _load_reservoir_hparams(dataset, reservoir_source)

    Xtr_raw, ytr_int, Xte_raw, _ = _load_dataset(dataset, use_cache)
    Xtr_raw, Xte_raw = _ensure_3d(Xtr_raw), _ensure_3d(Xte_raw)
    Xtr_sub, _, Xval, _ = _stratified_split(Xtr_raw, ytr_int, val_frac=0.2, seed=seed)
    Xtr_sub, Xval, _ = standardize(Xtr_sub, Xval, Xte_raw)

    reservoir = Reservoir(units=rc["units"], sr=rc["sr"], lr=rc["lr"],
                          input_scaling=rc["input_scaling"], seed=seed)
    S = _extract_states(Xtr_sub, reservoir)

    e   = np.linalg.svd(S, compute_uv=False) ** 2   # energy per direction
    cum = np.cumsum(e) / e.sum()
    return {
        "dataset":    dataset,
        "seed":       seed,
        "n_train":    S.shape[0],
        "units":      S.shape[1],
        "rank90":     int(np.searchsorted(cum, 0.90) + 1),
        "rank99":     int(np.searchsorted(cum, 0.99) + 1),
        "part_ratio": float((e.sum() ** 2) / (e ** 2).sum()),
    }


def main(datasets, seeds, reservoir_source, use_cache, csv_path, emit_latex):
    meta = _load_dataset_meta()
    init_csv(csv_path, _CSV_FIELDS)

    agg = {}
    for dataset in datasets:
        if dataset not in meta:
            print(f"[SKIP] '{dataset}' not in {config.paths.tsc_dataset_meta}")
            continue
        rows = []
        for seed in seeds:
            r = state_spectrum(dataset, seed, reservoir_source, use_cache)
            append_csv_row(csv_path, _CSV_FIELDS, {
                **r, "part_ratio": f"{r['part_ratio']:.2f}"})
            rows.append(r)
        agg[dataset] = rows
        m = lambda k: np.mean([r[k] for r in rows])       # noqa: E731
        s = lambda k: np.std([r[k] for r in rows])        # noqa: E731
        print(f"{dataset:30s} n_train={rows[0]['n_train']:5d}  "
              f"units={rows[0]['units']}  "
              f"rank@90={m('rank90'):5.1f}±{s('rank90'):.1f}  "
              f"rank@99={m('rank99'):5.1f}±{s('rank99'):.1f}  "
              f"PR={m('part_ratio'):5.1f}±{s('part_ratio'):.1f}")

    if emit_latex:
        print("\n% --- LaTeX table body (paste into main.tex) ---")
        for dataset, rows in agg.items():
            m = lambda k: np.mean([r[k] for r in rows])   # noqa: E731
            name = _PAPER_NAME.get(dataset, dataset)
            print(f"{name:14s} & ${rows[0]['n_train']}$ & "
                  f"${m('rank90'):.0f}$ & ${m('rank99'):.0f}$ & "
                  f"${m('part_ratio'):.1f}$ & "
                  f"${100 * m('rank99') / rows[0]['units']:.1f}$ \\\\")

    print(f"\nPer-seed rows → {csv_path}")


if __name__ == "__main__":
    meta = _load_dataset_meta()
    p = argparse.ArgumentParser(
        description="Effective rank of ESN reservoir states (readout redundancy)")
    p.add_argument("--datasets", nargs="+", default=sorted(meta.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--reservoir_source", default="sl1",
                   choices=["sl1", "l2", "none"])
    p.add_argument("--csv_path", default="./result/tsc_state_rank.csv")
    p.add_argument("--latex", action="store_true",
                   help="Also print the LaTeX table body")
    p.add_argument("--no_cache", action="store_true")
    a = p.parse_args()
    main(a.datasets, a.seeds, a.reservoir_source, not a.no_cache,
         a.csv_path, a.latex)
