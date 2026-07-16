# tsc_centralized_eval.py — centralized ESN classification: full-data train + test eval.
#
# TSC counterpart of tsr_centralized_eval.py: trains on the FULL training set
# and reports test metrics for each (dataset, reg_type) — no hyperparameter
# search, no train/val split.  Per-dataset hyperparameters come from
# configs/TSC_settings_<reg_type>.json by default (--no_best_params falls back
# to built-in defaults); promote tsc_centralized_search.py results there.
#
# Pipeline per (dataset, reg_type) — reuses tsc_centralized_search helpers:
#   1. Load dataset (cached) → standardise (stats fit on train only)
#   2. Run reservoir on every sequence → last hidden state per sequence
#   3. Train Clr_Node.fit_from_states() on ALL training states
#   4. Evaluate on the test set → acc / macro-F1 / balanced acc / Wout sparsity
#   5. Repeat over --runs reservoir seeds; report per-run + mean±std; the best
#      run (by test acc, tie-broken by macro-F1) is persisted.
#   6. Save: metrics JSON, best-run Wout .npy, confusion-matrix .png
#
# Usage examples:
#   python tsc_centralized_eval.py                             # all datasets × reg_types
#   python tsc_centralized_eval.py --datasets jpv --reg_types sl1
#   python tsc_centralized_eval.py --datasets jpv har --reg_types sl1 l2 --runs 5
#   python tsc_centralized_eval.py --datasets jpv --reg_types sl1 --no_best_params

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt

import config
from data_loader import one_hot, standardize
from tsc_centralized_search import (_load_dataset_meta, _load_dataset,
                                    _ensure_3d, _fit_eval)


# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_REG_TYPES = ["none", "l2", "sl1"]

# Conservative reservoir/readout defaults used when no search result is loaded.
# Run tsc_centralized_search.py first to find better params for your setup.
_DEFAULT_CFG = dict(
    sr            = 0.9,
    lr            = 0.3,
    input_scaling = 1.0,
    reg_param     = 1.0,
    epochs        = 500,
)


# ─── Hyperparameter loading ───────────────────────────────────────────────────

def _load_best_params(dataset: str, reg_type: str) -> dict | None:
    """Load a cfg from configs/TSC_settings_<reg_type>.json, or None if absent.

    The settings files hold the tuned per-dataset hyperparameters (promoted from
    tsc_centralized_search.py output).  Returns only the keys _fit_eval's cfg
    contract needs.
    """
    entry = config.load_settings("tsc", reg_type).get(dataset)
    if entry is None:
        return None
    print(f"  [hparams] loaded from configs/TSC_settings_{reg_type}.json")
    cfg = {k: entry[k] for k in
           ("sr", "lr", "input_scaling", "reg_param", "thres",
            "epochs", "alpha_init", "alpha_multiplier")
           if k in entry}
    return cfg


def _build_cfg(dataset: str, reg_type: str, args: argparse.Namespace) -> dict:
    """Assemble the trial cfg: defaults → best-params JSON → CLI overrides."""
    cfg = _DEFAULT_CFG.copy()
    cfg.update({
        "thres":            config.sl1_defaults.THRES,
        "alpha_init":       config.sl1_defaults.ALPHA_INIT,
        "alpha_multiplier": config.sl1_defaults.ALPHA_MULTIPLIER,
        "patience":         config.sl1_defaults.PATIENCE,
        "stag_tol":         config.sl1_defaults.STAG_TOL,
    })

    if args.use_best_params:
        best = _load_best_params(dataset, reg_type)
        if best is not None:
            cfg.update(best)
        else:
            print(f"  [hparams] no TSC_settings entry for {dataset}/{reg_type} "
                  f"— using defaults")

    cli_overrides = {k: v for k, v in {
        "sr":               args.sr,
        "lr":               args.lr,
        "input_scaling":    args.input_scaling,
        "reg_param":        args.reg_param,
        "epochs":           args.epochs,
        "thres":            args.thres,
        "alpha_init":       args.alpha_init,
        "alpha_multiplier": args.alpha_multiplier,
        "patience":         args.patience,
        "stag_tol":         args.stag_tol,
    }.items() if v is not None}
    cfg.update(cli_overrides)
    return cfg


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                   dataset: str, reg_type: str, acc: float,
                   save_dir: str) -> str:
    """Save a row-normalised confusion-matrix heatmap; return its path."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1)

    side = max(4.0, 0.45 * n_classes)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Cell counts only stay readable for small class counts.
    if n_classes <= 20:
        for i in range(n_classes):
            for j in range(n_classes):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{dataset.upper()} — {reg_type}   |   Test acc = {acc:.2f}%")

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{dataset}_{reg_type}_confusion.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(dataset: str, reg_type: str, cfg: dict, meta: dict,
                   runs: int, seed: int, save_dir: str,
                   use_cache: bool, use_gpu: bool,
                   save_model: bool = False) -> dict:
    """Full-data train + test eval over *runs* reservoir seeds; persist outputs.

    Saves:
        <save_dir>/<dataset>_<reg_type>_metrics.json    — per-run + aggregate metrics
        <save_dir>/<dataset>_<reg_type>_Wout.npy        — best-run readout weights
        <save_dir>/<dataset>_<reg_type>_confusion.png   — best-run confusion matrix

    Returns:
        Scalar metrics dict (same fields as the metrics JSON).
    """
    print(f"\n{'─'*60}")
    print(f"Dataset: {dataset}  |  reg_type: {reg_type}")
    for k, v in cfg.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'─'*60}")

    Xtr_raw, ytr_int, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
    Xtr_raw = _ensure_3d(Xtr_raw)
    Xte_raw = _ensure_3d(Xte_raw)
    Xtr_std, Xte_std = standardize(Xtr_raw, Xte_raw)

    n_classes = meta["output_dim"]
    ytr_oh    = one_hot(ytr_int, n_classes)

    t0 = time.time()
    per_run = []
    best = None          # (score, run, scores, sparsity, Wout, y_pred)
    for run in range(runs):
        scores, sparsity, Wout, _Xtr_s, Xte_s = _fit_eval(
            Xtr_std, ytr_oh, Xte_std, yte_int,
            meta, reg_type, cfg, seed=seed + run, use_gpu=use_gpu)
        y_pred = np.argmax(Xte_s @ Wout, axis=1)

        print(f"  run {run+1}/{runs}: acc={scores['acc']:.2f}%  "
              f"f1={scores['macro_f1']:.2f}%  bal={scores['balanced_acc']:.2f}%  "
              f"sparsity={sparsity:.1f}%")
        per_run.append({**scores, "sparsity": sparsity, "seed": seed + run})

        score = (scores["acc"], scores["macro_f1"])
        if best is None or score > best[0]:
            best = (score, run, scores, sparsity, Wout.copy(), y_pred.copy())
    elapsed = time.time() - t0

    _score, best_run, best_scores, best_sparsity, best_Wout, best_pred = best
    accs = np.array([r["acc"] for r in per_run])
    sps  = np.array([r["sparsity"] for r in per_run])

    print(f"\nBest run ({best_run+1}/{runs}, selected by test acc):")
    print(f"  Test acc          : {best_scores['acc']:.2f}%   "
          f"(mean±std over {runs} runs: {accs.mean():.2f}±{accs.std():.2f}%)")
    print(f"  Macro-F1 / BalAcc : {best_scores['macro_f1']:.2f}% / "
          f"{best_scores['balanced_acc']:.2f}%")
    print(f"  Wout sparsity     : {best_sparsity:.1f}%   "
          f"(mean±std: {sps.mean():.1f}±{sps.std():.1f}%)")
    print(f"  Elapsed           : {elapsed:.1f}s")

    # ── Persist outputs ──────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{dataset}_{reg_type}"

    result = {
        "dataset":           dataset,
        "reg_type":          reg_type,
        "cfg":               cfg,
        "acc_test":          best_scores["acc"],
        "macro_f1_test":     best_scores["macro_f1"],
        "balanced_acc_test": best_scores["balanced_acc"],
        "sparsity_pct":      best_sparsity,
        "acc_mean":          float(accs.mean()),
        "acc_std":           float(accs.std()),
        "sparsity_mean":     float(sps.mean()),
        "sparsity_std":      float(sps.std()),
        "best_run":          best_run,
        "total_runs":        runs,
        "per_run":           per_run,
    }
    with open(os.path.join(save_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(result, f, indent=4)
    if save_model:
        np.save(os.path.join(save_dir, f"{tag}_Wout.npy"), best_Wout)

    plot_path = plot_confusion(yte_int, best_pred, n_classes,
                               dataset, reg_type, best_scores["acc"], save_dir)
    print(f"  Plot saved → {plot_path}")

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    meta         = _load_dataset_meta()
    all_datasets = sorted(meta.keys())

    p = argparse.ArgumentParser(
        description="Centralized ESN classification — full-data train + test eval"
    )
    p.add_argument("--datasets",  nargs="+", default=all_datasets,
                   help=f"Datasets to evaluate (default: {' '.join(all_datasets)})")
    p.add_argument("--reg_types", nargs="+", default=SUPPORTED_REG_TYPES,
                   choices=SUPPORTED_REG_TYPES,
                   help="Regularization types (default: all)")
    # Hyperparameter overrides (applied on top of defaults / best_params)
    p.add_argument("--sr",            type=float, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--input_scaling", type=float, default=None)
    p.add_argument("--reg_param",     type=float, default=None)
    p.add_argument("--epochs",        type=int,   default=None)
    # SL1 sparsification schedule (per-step soft-threshold + α continuation)
    p.add_argument("--thres",            type=float, default=None,
                   help="SL1 soft-threshold magnitude (per-step pruning)")
    p.add_argument("--alpha_init",       type=float, default=None,
                   help="Initial SmoothL1 alpha (α continuation start)")
    p.add_argument("--alpha_multiplier", type=float, default=None,
                   help="Alpha growth factor (α continuation rate)")
    p.add_argument("--patience",         type=int,   default=None,
                   help="Newton stagnation early-stop window; 0 disables")
    p.add_argument("--stag_tol",         type=float, default=None,
                   help="Max non-zero-fraction change over the patience window")
    # Run control
    p.add_argument("--runs",      type=int, default=3,
                   help="Independent reservoir seeds per experiment; best test "
                        "acc persisted, mean±std reported (default: 3)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--save_dir",  default="./result/classification",
                   help="Output directory (default: ./result/classification)")
    p.add_argument("--save_model", action="store_true",
                   help="Also save the best-run Wout .npy (off by default; "
                        "metrics JSON and confusion plot are always written)")
    p.add_argument("--no_best_params", dest="use_best_params",
                   action="store_false", default=True,
                   help="Ignore configs/TSC_settings_<reg_type>.json and use "
                        "built-in defaults (best params are ON by default)")
    p.add_argument("--no_cache",  action="store_true",
                   help="Reload datasets from source instead of the .npz cache")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; auto-falls back to CPU")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_meta = _load_dataset_meta()

    all_results = []
    for ds in args.datasets:
        if ds not in dataset_meta:
            print(f"[SKIP] '{ds}' not in {config.paths.tsc_dataset_meta}")
            continue
        for rt in args.reg_types:
            cfg = _build_cfg(ds, rt, args)
            try:
                m = run_experiment(
                    dataset  = ds,
                    reg_type = rt,
                    cfg      = cfg,
                    meta     = dataset_meta[ds],
                    runs     = args.runs,
                    seed     = args.seed,
                    save_dir = args.save_dir,
                    use_cache = not args.no_cache,
                    use_gpu   = args.use_gpu,
                    save_model = args.save_model,
                )
            except Exception as exc:                              # noqa: BLE001
                print(f"  [ERROR] {ds}/{rt}: {exc}")
                continue
            all_results.append(m)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CENTRALIZED ESN CLASSIFICATION — SUMMARY (best run; mean±std in brackets)")
    print("=" * 78)
    hdr = (f"{'Dataset':<10} {'RegType':<7} {'Acc':>7} {'Acc μ±σ':>15} "
           f"{'MacroF1':>8} {'BalAcc':>7} {'Sparsity':>9}")
    print(hdr)
    print("─" * len(hdr))
    for m in all_results:
        mu = f"{m['acc_mean']:.2f}±{m['acc_std']:.2f}%"
        print(f"{m['dataset']:<10} {m['reg_type']:<7} "
              f"{m['acc_test']:>6.2f}% {mu:>15} "
              f"{m['macro_f1_test']:>7.2f}% {m['balanced_acc_test']:>6.2f}% "
              f"{m['sparsity_pct']:>8.1f}%")

    print(f"\nAll outputs saved to: {args.save_dir}/")
