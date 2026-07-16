# tsc_loss_compare.py — fair head-to-head of CE vs MSE readout for ESN classification.
#
# Question: for sequence classification, does a least-squares (ridge) readout
# trained on one-hot targets match / beat the softmax cross-entropy readout?
#
# To isolate the *loss function* as the only variable, every other factor is held
# fixed:
#   • ONE reservoir per (dataset, seed) — both readouts are trained on the SAME
#     extracted last-states (Xtr_states / Xval_states / Xte_states), so any
#     accuracy gap is attributable to the loss, not to a different reservoir.
#   • Each loss independently tunes its own reg_param on the validation split
#     (CE and MSE live on different scales, so a shared λ would be unfair).
#   • The chosen λ is then refit on train+val and scored once on the test split.
#   • Results are averaged over several reservoir seeds to damp the random-
#     reservoir variance.
#
# Readouts compared (per reg flavour):
#   CE  : Clr_Node (softmax cross-entropy) — none → plain CE, l2 → CE + ridge,
#         solved by the project's Adam path. Predict argmax(X @ Wout).
#   MSE : closed-form least-squares on one-hot — none → OLS (pinv), l2 → ridge
#         Wout = (XᵀX + λI)⁻¹ XᵀY. Predict argmax(X @ Wout).  (No iteration.)
#
# Both predict identically (argmax of the linear scores); only how Wout is
# obtained differs. sl1 is intentionally excluded — L1 has no closed form and is
# not what this comparison is about.
#
# Results:
#   ./result/tsc_loss_compare.csv  — one row per (dataset, reg, loss, seed) + means
#
# Usage:
#   python tsc_loss_compare.py
#   python tsc_loss_compare.py --datasets har char --reg_types none l2
#   python tsc_loss_compare.py --seeds 1 2 3 4 5 --epochs 800

import argparse
import os
import time
from copy import deepcopy

import numpy as np
from reservoirpy.nodes import Reservoir

import config
from data_loader import one_hot, standardize
from readout_node import Clr_Node
from utils import init_csv, append_csv_row
from tsc_centralized_search import (
    _load_dataset_meta, _load_dataset, _stratified_split,
    _extract_states, _ensure_3d,
)


# ─── MSE (least-squares / ridge) readout — closed form ─────────────────────────

def _mse_readout(X: np.ndarray, Y_oh: np.ndarray, reg_param: float) -> np.ndarray:
    """Closed-form least-squares classifier weights, shape (units, n_classes).

    reg_param == 0 → ordinary least squares (lstsq, robust to rank deficiency).
    reg_param  > 0 → ridge: Wout = (XᵀX + λI)⁻¹ XᵀY  (one-hot Y).
    """
    if reg_param <= 0:
        return np.linalg.lstsq(X, Y_oh, rcond=None)[0]
    d = X.shape[1]
    A = X.T @ X + reg_param * np.eye(d, dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y_oh)


def _ce_readout(X: np.ndarray, Y_oh: np.ndarray, reg_type: str,
                reg_param: float, learning_rate: float, epochs: int,
                use_gpu: bool) -> np.ndarray:
    """Softmax cross-entropy readout weights via the project's Clr_Node (Adam)."""
    readout = Clr_Node(
        reg_param        = reg_param,
        reg_type         = reg_type,
        thres            = config.sl1_defaults.THRES,
        alpha_init       = config.sl1_defaults.ALPHA_INIT,
        alpha_multiplier = config.sl1_defaults.ALPHA_MULTIPLIER,
        learning_rate    = learning_rate,
        epochs           = epochs,
        solver           = "adam",   # none/l2 → Adam (sl1 is excluded here)
        use_gpu          = use_gpu,
    )
    return readout.fit_from_states(X, Y_oh)


def _acc(W: np.ndarray, X: np.ndarray, y_int: np.ndarray) -> float:
    return float((np.argmax(X @ W, axis=1) == y_int).mean() * 100)


# ─── Per-dataset comparison ────────────────────────────────────────────────────

def _states_for_seed(Xtr, Xval, Xte, units, sr, lr, input_scaling, seed):
    """Build one reservoir and harvest last-states for all three splits."""
    reservoir = Reservoir(units=units, sr=sr, lr=lr,
                          input_scaling=input_scaling, seed=seed)
    return (_extract_states(Xtr,  reservoir),
            _extract_states(Xval, reservoir),
            _extract_states(Xte,  reservoir))


def _select_and_test(fit_fn, S_tr, y_tr_oh, S_val, y_val_int,
                     S_trval, y_trval_oh, S_te, y_te_int, reg_grid):
    """Tune reg_param on val, refit on train+val, score on test.

    fit_fn(states, y_oh, reg_param) → Wout.  Returns (test_acc, best_reg, val_acc, fit_secs).
    """
    best = (-1.0, None, -1.0)   # (val_acc, reg_param, _)
    for rp in reg_grid:
        W = fit_fn(S_tr, y_tr_oh, rp)
        va = _acc(W, S_val, y_val_int)
        if va > best[0]:
            best = (va, rp, va)
    best_val, best_rp, _ = best

    t0 = time.time()
    W_final = fit_fn(S_trval, y_trval_oh, best_rp)
    fit_secs = time.time() - t0
    return _acc(W_final, S_te, y_te_int), best_rp, best_val, fit_secs


_CSV_FIELDS = [
    "dataset", "reg_type", "loss", "seed",
    "best_reg_param", "val_acc", "test_acc", "fit_secs",
]


def compare(datasets, reg_types, seeds, units_override, sr, lr, input_scaling,
            learning_rate, epochs, reg_grid, csv_path, use_cache, use_gpu,
            use_best_params=True, reservoir_source="l2"):
    meta = _load_dataset_meta()
    init_csv(csv_path, _CSV_FIELDS)
    summary = {}   # (dataset, reg_type, loss) → list of test_acc

    for dataset in datasets:
        if dataset not in meta:
            print(f"[SKIP] '{dataset}' not in {config.paths.tsc_dataset_meta}")
            continue
        # Per-dataset reservoir operating point from the tuned settings file
        # (shared by both losses so the comparison isolates the readout); the
        # CLI sr/lr/input_scaling are the fallback when the entry is missing
        # or --no_best_params is passed.
        ds_sr, ds_lr, ds_is = sr, lr, input_scaling
        units = units_override or meta[dataset]["units"]
        if use_best_params:
            s = config.load_settings("tsc", reservoir_source).get(dataset)
            if s is not None:
                ds_sr = s.get("sr", ds_sr)
                ds_lr = s.get("lr", ds_lr)
                ds_is = s.get("input_scaling", ds_is)
                units = units_override or s.get("units", units)
                print(f"  [hparams] reservoir from configs/TSC_settings_"
                      f"{reservoir_source}.json")
        n_classes = meta[dataset]["output_dim"]

        print(f"\n{'='*70}\nDataset: {dataset}  "
              f"(classes={n_classes}, units={units}, sr={ds_sr:.4g}, lr={ds_lr:.4g}, "
              f"input_scaling={ds_is:.4g})\n{'='*70}")

        Xtr_raw, ytr_int, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
        Xtr_raw, Xte_raw = _ensure_3d(Xtr_raw), _ensure_3d(Xte_raw)

        for seed in seeds:
            # Same split + same reservoir per seed → both losses see identical states.
            Xtr_sub, ytr_sub_int, Xval, yval_int = _stratified_split(
                Xtr_raw, ytr_int, val_frac=0.2, seed=seed)
            Xtr_sub, Xval, Xte = standardize(Xtr_sub, Xval, Xte_raw)

            S_tr, S_val, S_te = _states_for_seed(
                Xtr_sub, Xval, Xte, units, ds_sr, ds_lr, ds_is, seed)

            # train+val states for the final refit (states are per-sequence,
            # so concatenation is exactly "reservoir run on the full train set").
            S_trval   = np.vstack([S_tr, S_val])
            ytrval_int = np.concatenate([ytr_sub_int, yval_int])

            y_tr_oh    = one_hot(ytr_sub_int, n_classes)
            y_trval_oh = one_hot(ytrval_int,  n_classes)

            for reg_type in reg_types:
                grid = [0.0] if reg_type == "none" else reg_grid

                # CE
                ce_acc, ce_rp, ce_va, ce_t = _select_and_test(
                    lambda X, Y, rp: _ce_readout(
                        X, Y, reg_type, rp, learning_rate, epochs, use_gpu),
                    S_tr, y_tr_oh, S_val, yval_int,
                    S_trval, y_trval_oh, S_te, yte_int, grid)

                # MSE (reg_type "none" → OLS, "l2" → ridge; "sl1" not applicable)
                mse_acc, mse_rp, mse_va, mse_t = _select_and_test(
                    lambda X, Y, rp: _mse_readout(X, Y, rp),
                    S_tr, y_tr_oh, S_val, yval_int,
                    S_trval, y_trval_oh, S_te, yte_int, grid)

                print(f"  seed={seed}  {reg_type:4s} | "
                      f"CE  test={ce_acc:5.2f}% (λ={ce_rp:.3g}, {ce_t:.2f}s)   "
                      f"MSE test={mse_acc:5.2f}% (λ={mse_rp:.3g}, {mse_t:.3f}s)   "
                      f"Δ(MSE-CE)={mse_acc-ce_acc:+.2f}")

                for loss, (acc, rp, va, ft) in (
                    ("ce",  (ce_acc, ce_rp, ce_va, ce_t)),
                    ("mse", (mse_acc, mse_rp, mse_va, mse_t)),
                ):
                    summary.setdefault((dataset, reg_type, loss), []).append(acc)
                    append_csv_row(csv_path, _CSV_FIELDS, {
                        "dataset": dataset, "reg_type": reg_type, "loss": loss,
                        "seed": seed, "best_reg_param": f"{rp:.6g}",
                        "val_acc": f"{va:.4f}", "test_acc": f"{acc:.4f}",
                        "fit_secs": f"{ft:.4f}",
                    })

    # ── Summary table (mean ± std over seeds) ──────────────────────────────────
    print(f"\n{'='*70}\nLOSS COMPARISON SUMMARY  (test acc, mean±std over "
          f"{len(seeds)} seeds)\n{'='*70}")
    print(f"  {'dataset':18s} {'reg':5s} {'CE':>16s} {'MSE':>16s} {'Δ(MSE-CE)':>11s}")
    for (ds, rt) in sorted({(d, r) for (d, r, _) in summary}):
        ce  = np.array(summary.get((ds, rt, "ce"),  [np.nan]))
        mse = np.array(summary.get((ds, rt, "mse"), [np.nan]))
        print(f"  {ds:18s} {rt:5s} "
              f"{ce.mean():7.2f}±{ce.std():4.2f}   "
              f"{mse.mean():7.2f}±{mse.std():4.2f}   "
              f"{mse.mean()-ce.mean():+8.2f}")
    print(f"\nPer-run rows saved → {csv_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    meta = _load_dataset_meta()
    p = argparse.ArgumentParser(
        description="Compare CE vs MSE (ridge) readout for ESN classification")
    p.add_argument("--datasets",  nargs="+", default=sorted(meta.keys()))
    p.add_argument("--reg_types", nargs="+", default=["none", "l2"],
                   choices=["none", "l2"],
                   help="sl1 is excluded (no closed form); compares none/l2 only")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                   help="Reservoir/split seeds to average over (default: 1 2 3)")
    p.add_argument("--units", type=int, default=None,
                   help="Override reservoir size (default: per-dataset from config)")
    p.add_argument("--sr",            type=float, default=0.9,
                   help="Fallback spectral radius when best params are off/missing")
    p.add_argument("--lr",            type=float, default=0.3,
                   help="Fallback leaking rate when best params are off/missing")
    p.add_argument("--input_scaling", type=float, default=1.0,
                   help="Fallback input scaling when best params are off/missing")
    p.add_argument("--no_best_params", dest="use_best_params",
                   action="store_false", default=True,
                   help="Ignore configs/TSC_settings_*.json and use the CLI "
                        "sr/lr/input_scaling for every dataset")
    p.add_argument("--reservoir_source", default="l2", choices=["sl1", "l2", "none"],
                   help="Which configs/TSC_settings_*.json supplies the per-dataset "
                        "reservoir hyperparameters (default: l2)")
    p.add_argument("--learning_rate", type=float, default=0.01,
                   help="Adam lr for the CE readout (CE-only knob)")
    p.add_argument("--epochs", type=int, default=500,
                   help="Adam epochs for the CE readout (give it enough to converge)")
    p.add_argument("--reg_grid", nargs="+", type=float,
                   default=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                   help="reg_param grid each loss tunes over on validation")
    p.add_argument("--csv_path", default="./result/tsc_loss_compare.csv")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU)
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compare(
        datasets=a.datasets, reg_types=a.reg_types, seeds=a.seeds,
        units_override=a.units, sr=a.sr, lr=a.lr, input_scaling=a.input_scaling,
        learning_rate=a.learning_rate, epochs=a.epochs, reg_grid=a.reg_grid,
        csv_path=a.csv_path, use_cache=not a.no_cache, use_gpu=a.use_gpu,
        use_best_params=a.use_best_params, reservoir_source=a.reservoir_source,
    )
