# tsc_centralized_sensitivity.py — single-hyperparameter sensitivity analysis for SL1-ESN.
#
# Reads base hyperparameters from configs/TSC_settings_sl1.json, then sweeps one
# chosen parameter across a value range while keeping all others fixed.  Each
# value is repeated with n_seeds different reservoir random seeds; results are
# averaged and plotted as mean ± std for both val_acc and val_sparsity.
#
# Reuses the data/reservoir/readout helpers from tsc_centralized_search.py —
# no duplicate logic.
#
# Usage:
#   python tsc_centralized_sensitivity.py --dataset har --param reg_param
#   python tsc_centralized_sensitivity.py --dataset char --param thres \
#       --values 1e-5 1e-4 1e-3 1e-2 0.1
#   python tsc_centralized_sensitivity.py --dataset har --param sr \
#       --n_seeds 5 --epochs 300
#   python tsc_centralized_sensitivity.py --dataset har --param reg_param \
#       --out_dir ./result/pic

import argparse
import json
import os
import sys
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")   # headless backend — works without a display
import matplotlib.pyplot as plt
import numpy as np

import config
import tsc_centralized_search as chs   # reuse all data/reservoir/readout helpers
from data_loader import one_hot, standardize
from utils import parallel_map


# ─── Sweepable parameters and their default grids ─────────────────────────────

# Parameters whose natural scale is logarithmic (plot x-axis in log scale)
_LOG_PARAMS = {"reg_param", "thres", "input_scaling"}

_DEFAULT_VALUES: dict[str, list] = {
    "reg_param":        [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
    "thres":            [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1],
    "alpha_init":       [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "alpha_multiplier": [1.1, 1.25, 1.5, 2.0, 5.0, 10.0],
    "input_scaling":    [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
    "sr":               [0.3, 0.6, 0.98, 1.2, 1.5, 2.0, 2.5, 3.0],
    "lr":               [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "epochs":           [50, 100, 200, 300, 500],
    "units":            [50, 100, 200, 300, 500, 800, 1000],
}

# Parameters that take integer values and live in *meta* (reservoir construction)
# rather than *cfg* (readout training).
_INT_PARAMS  = {"epochs", "units"}
_META_PARAMS = {"units"}

SWEEPABLE_PARAMS = sorted(_DEFAULT_VALUES.keys())


# ─── Base config loader ───────────────────────────────────────────────────────

def _load_base_config(dataset: str) -> dict:
    """Return the SL1 config entry for *dataset* from TSC_settings_sl1.json."""
    path = os.path.join(config.paths.configs_path, "TSC_settings_sl1.json")
    with open(path) as f:
        entries = json.load(f)
    for e in entries:
        if e["dataset"] == dataset:
            return deepcopy(e)
    available = [e["dataset"] for e in entries]
    raise ValueError(
        f"Dataset '{dataset}' not in TSC_settings_sl1.json. "
        f"Available: {available}"
    )


def _build_trial_inputs(base: dict, param_name: str, value, epochs: int,
                        sl1_base: dict) -> tuple:
    """Return (meta, cfg) for one trial with *param_name* overridden to *value*.

    meta — dict with units/input_dim/output_dim (for reservoir construction)
    cfg  — dict with lr/sr/input_scaling/reg_param/thres/alpha_init/
           alpha_multiplier/epochs (consumed by chs._run_trial)

    The SL1 sparsification schedule (thres / alpha_init / alpha_multiplier) is
    uniform across datasets: its baseline comes from *sl1_base* (CLI), not the
    per-dataset config entry.
    """
    meta = {
        "input_dim":  base["input_dim"],
        "output_dim": base["output_dim"],
        "units":      base.get("units", 500),
    }
    cfg = {
        "lr":               base["lr"],
        "sr":               base["sr"],
        "input_scaling":    base["input_scaling"],
        "reg_param":        base["reg_param"],
        "thres":            sl1_base["thres"],
        "alpha_init":       sl1_base["alpha_init"],
        "alpha_multiplier": sl1_base["alpha_multiplier"],
        "patience":         sl1_base.get("patience", config.sl1_defaults.PATIENCE),
        "stag_tol":         sl1_base.get("stag_tol", config.sl1_defaults.STAG_TOL),
        "epochs":           epochs,
    }
    if param_name in _META_PARAMS:
        # units → reservoir construction, not readout cfg
        meta[param_name] = int(value)
    elif param_name in _INT_PARAMS:
        cfg[param_name] = int(value)
    else:
        cfg[param_name] = float(value)
    return meta, cfg


# ─── Parallel trial worker ─────────────────────────────────────────────────────

# Prepared per-sweep arrays shared with pool workers via fork (copy-on-write).
# Set in sweep() before parallel_map() so forked workers inherit the (possibly
# large) design matrices without per-task pickling.  Mirrors the _WORKER_DATA
# pattern in tsc_centralized_search.py.
_WORKER_DATA: dict = {}


def _sens_worker(meta: dict, trial_cfg: dict, seed: int, use_gpu: bool):
    """Run one (value, seed) trial in a pool worker; reads data from _WORKER_DATA.

    Returns (val_acc, val_sparsity), or ("__error__", message) so the parent can
    log the failure and record NaN without the worker touching shared state.
    """
    d = _WORKER_DATA
    try:
        return chs._run_trial(
            d["Xtr"], d["ytr_oh"], d["Xval"], d["yval_int"],
            meta, "sl1", trial_cfg, seed=seed, use_gpu=use_gpu,
        )
    except Exception as exc:                                  # noqa: BLE001
        return ("__error__", str(exc))


# ─── Sweep ────────────────────────────────────────────────────────────────────

def sweep(
    dataset: str,
    param_name: str,
    param_values: list,
    n_seeds: int     = 3,
    epochs: int | None = None,
    val_frac: float  = 0.2,
    base_seed: int   = 42,
    use_cache: bool  = True,
    sl1_base: dict | None = None,
    use_gpu: bool    = config.USE_GPU,
    n_jobs: int      = 1,
    eval_split: str  = "val",
) -> list[dict]:
    """Sweep *param_name* over *param_values*; return one result dict per value.

    Each result dict has keys:
        param_value, mean_val_acc, std_val_acc,
        mean_val_sparsity, std_val_sparsity

    eval_split selects what the reported accuracy is measured on. "val" (default)
    trains on a stratified 1-val_frac subset and evaluates on the held-out
    validation split, as used for model selection and the ablation figure.
    "test" trains on the full training set and evaluates on the official test
    set, matching the operating-point tables; use it for the sparsity-sweep
    figures so they are on the same footing as those tables.
    """
    base_cfg   = _load_base_config(dataset)
    base_epochs = epochs if epochs is not None else base_cfg.get("epochs", 500)
    n_classes   = base_cfg["output_dim"]

    if sl1_base is None:
        sl1_base = {
            "thres":            config.sl1_defaults.THRES,
            "alpha_init":       config.sl1_defaults.ALPHA_INIT,
            "alpha_multiplier": config.sl1_defaults.ALPHA_MULTIPLIER,
            "patience":         config.sl1_defaults.PATIENCE,
            "stag_tol":         config.sl1_defaults.STAG_TOL,
        }

    # ── Load and prepare data once ────────────────────────────────────────────
    print(f"Loading dataset '{dataset}' (eval on {eval_split} split) …")
    Xtr_raw, ytr_int_raw, Xte_raw, yte_int_raw = chs._load_dataset(dataset, use_cache)
    Xtr_raw = chs._ensure_3d(Xtr_raw)
    Xte_raw = chs._ensure_3d(Xte_raw)

    if eval_split == "test":
        # Train on the full training set, evaluate on the official test set —
        # the same protocol as the operating-point tables.
        Xtr_sub, Xval, _ = standardize(Xtr_raw, Xte_raw, Xte_raw)
        ytr_sub_oh = one_hot(ytr_int_raw, n_classes)
        yval_int   = yte_int_raw
    else:
        Xtr_sub, ytr_sub_int, Xval, yval_int = chs._stratified_split(
            Xtr_raw, ytr_int_raw, val_frac=val_frac, seed=base_seed
        )
        Xtr_sub, Xval, _ = standardize(Xtr_sub, Xval, Xte_raw)
        ytr_sub_oh = one_hot(ytr_sub_int, n_classes)

    # GPU + multiprocessing oversubscribe a single device; serialise in that case
    # (mirrors tsc_centralized_search.py).
    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"to avoid oversubscribing the device.")
        n_jobs = 1

    # ── Build the flat (value × seed) task list ───────────────────────────────
    total = len(param_values) * n_seeds
    # Share the design matrices with forked workers via module globals so they
    # are not pickled once per task.
    _WORKER_DATA.update({
        "Xtr":      Xtr_sub,
        "ytr_oh":   ytr_sub_oh,
        "Xval":     Xval,
        "yval_int": yval_int,
    })

    tasks = []                                     # flat (meta, cfg, seed, gpu)
    for i, v in enumerate(param_values):
        meta, trial_cfg = _build_trial_inputs(base_cfg, param_name, v, base_epochs, sl1_base)
        for s in range(n_seeds):
            seed = base_seed + i * n_seeds + s
            tasks.append((meta, trial_cfg, seed, use_gpu))

    print(f"\nBase config loaded from TSC_settings_sl1.json")
    print(f"Sweeping '{param_name}' over {len(param_values)} values "
          f"× {n_seeds} seeds = {total} trials  (n_jobs={n_jobs})\n")

    flat = parallel_map(_sens_worker, tasks, n_jobs,
                        progress=f"  {dataset}/{param_name}")

    # ── Regroup results per value (n_seeds each), in deterministic order ───────
    results = []
    for i, v in enumerate(param_values):
        accs, sparsities = [], []
        for s in range(n_seeds):
            res = flat[i * n_seeds + s]
            if isinstance(res, tuple) and res and res[0] == "__error__":
                print(f"  {param_name}={v:.4g}  seed={base_seed + i * n_seeds + s}"
                      f"  [ERROR: {res[1]}]")
                continue
            val_acc, val_sp = res
            accs.append(val_acc)
            sparsities.append(val_sp)

        if accs:
            results.append({
                "param_value":       v,
                "mean_val_acc":      float(np.mean(accs)),
                "std_val_acc":       float(np.std(accs)),
                "mean_val_sparsity": float(np.mean(sparsities)),
                "std_val_sparsity":  float(np.std(sparsities)),
            })

    return results


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_sensitivity(
    results: list[dict],
    param_name: str,
    dataset: str,
    out_dir: str,
    baseline_value: float | None = None,
) -> None:
    """Save a dual-y-axis sensitivity plot (val_acc and val_sparsity vs param).

    *baseline_value*: if given, draws a vertical dashed line at the config value
    so it is easy to see where the current setting sits in the sweep.
    """
    xs      = [r["param_value"]       for r in results]
    accs    = [r["mean_val_acc"]      for r in results]
    acc_std = [r["std_val_acc"]       for r in results]
    sps     = [r["mean_val_sparsity"] for r in results]
    sp_std  = [r["std_val_sparsity"]  for r in results]

    color_acc = "#2166ac"   # blue  — val_acc
    color_sp  = "#d6604d"   # orange-red — val_sparsity

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # ── val_acc (left y-axis) ─────────────────────────────────────────────────
    ax1.plot(xs, accs, color=color_acc, marker="o", linewidth=2.0,
             zorder=3, label="val_acc (%)")
    ax1.fill_between(
        xs,
        [a - s for a, s in zip(accs, acc_std)],
        [a + s for a, s in zip(accs, acc_std)],
        color=color_acc, alpha=0.15, zorder=2,
    )

    # ── val_sparsity (right y-axis) ───────────────────────────────────────────
    ax2.plot(xs, sps, color=color_sp, marker="s", linewidth=2.0,
             linestyle="--", zorder=3, label="val_sparsity (%)")
    ax2.fill_between(
        xs,
        [a - s for a, s in zip(sps, sp_std)],
        [a + s for a, s in zip(sps, sp_std)],
        color=color_sp, alpha=0.15, zorder=2,
    )

    # ── Baseline marker ───────────────────────────────────────────────────────
    if baseline_value is not None:
        ax1.axvline(baseline_value, color="gray", linewidth=1.2,
                    linestyle=":", zorder=1,
                    label=f"config value ({baseline_value:.4g})")

    # ── Axes formatting ───────────────────────────────────────────────────────
    if param_name in _LOG_PARAMS:
        ax1.set_xscale("log")
    if param_name in _INT_PARAMS:
        # Integer-valued params (units/epochs): tick only at the sampled integers
        # so the axis never shows fractional values like 250.5.
        ax1.set_xticks(sorted({int(x) for x in xs}))

    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel("val_acc (%)", color=color_acc, fontsize=12)
    ax2.set_ylabel("val_sparsity (%)", color=color_sp, fontsize=12)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_sp)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=10)

    plt.title(
        f"SL1-ESN (centralized) — sensitivity to '{param_name}'\n"
        f"Dataset: {dataset}",
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    stem = f"sensitivity_{dataset}_{param_name}"
    for ext in ("pdf", "png"):
        out_path = os.path.join(out_dir, f"{stem}.{ext}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close(fig)


# ─── Result export ────────────────────────────────────────────────────────────

def save_run(
    results: list[dict],
    param_name: str,
    dataset: str,
    out_dir: str,
    *,
    n_seeds: int,
    base_seed: int,
    epochs,
    val_frac: float,
    sl1_base: dict,
    baseline_value=None,
    eval_split: str = "val",
) -> str:
    """Write the full run (all hyperparameters + per-value results) to JSON.

    The schema is shared with the FL sweep and consumed by the MATLAB plotting
    scripts in ``matlab/`` (``plot_sensitivity.m``). Returns the JSON path.
    """
    payload = {
        "kind":           "centralized",
        "dataset":        dataset,
        "reg_type":       "sl1",
        "param_name":     param_name,
        "is_integer":     param_name in _INT_PARAMS,
        "is_log":         param_name in _LOG_PARAMS,
        "acc_label":      "test_acc (%)" if eval_split == "test" else "val_acc (%)",
        "sparsity_label": "test_sparsity (%)" if eval_split == "test" else "val_sparsity (%)",
        "title":          f"SL1-ESN (centralized) — sensitivity to '{param_name}'",
        "baseline_value": baseline_value,
        # ── All hyperparameters that define the run ───────────────────────────
        "settings": {
            "n_seeds":   n_seeds,
            "base_seed": base_seed,
            "epochs":    epochs,
            "val_frac":  val_frac,
            **sl1_base,
        },
        # ── Per-value swept results ───────────────────────────────────────────
        "results": results,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sensitivity_{dataset}_{param_name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Centralized SL1-ESN single-hyperparameter sensitivity analysis.\n"
            "Reads base config from configs/TSC_settings_sl1.json, sweeps one\n"
            "parameter, and outputs a chart to result/pic/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True,
        help="Dataset name — must exist in configs/TSC_settings_sl1.json",
    )
    p.add_argument(
        "--param", default="reg_param",
        choices=SWEEPABLE_PARAMS,
        metavar="PARAM",
        help=(
            f"Hyperparameter to sweep (default: reg_param). "
            f"Choices: {SWEEPABLE_PARAMS}"
        ),
    )
    p.add_argument(
        "--values", nargs="+", type=float, default=None,
        metavar="V",
        help=(
            "Explicit sweep values (space-separated floats). "
            "Omit to use a built-in default grid for the chosen param."
        ),
    )
    p.add_argument(
        "--n_seeds", type=int, default=3,
        help="Reservoir seeds per parameter value — results are averaged (default: 3)",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Readout training epochs (default: value from config, fallback 500)",
    )
    p.add_argument(
        "--val_frac", type=float, default=0.2,
        help="Validation fraction of training data (default: 0.2)",
    )
    p.add_argument(
        "--eval_split", choices=["val", "test"], default="val",
        help="Report accuracy on the held-out validation split (default) or on "
             "the official test set (train on full training data). Use 'test' "
             "for the sparsity-sweep figures so they match the tables.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    p.add_argument(
        "--no_cache", action="store_true",
        help="Reload raw data instead of using .npz cache",
    )
    p.add_argument(
        "--out_dir", default=config.paths.pics_path,
        help=f"Output directory for figures (default: {config.paths.pics_path})",
    )
    # ── SL1 sparsification schedule baselines (uniform; used for non-swept params) ─
    p.add_argument("--thres", type=float, default=config.sl1_defaults.THRES,
                   help=f"Baseline SL1 threshold (default: {config.sl1_defaults.THRES})")
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help=f"Baseline SmoothL1 alpha (default: {config.sl1_defaults.ALPHA_INIT})")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help=f"Baseline alpha growth factor (default: {config.sl1_defaults.ALPHA_MULTIPLIER})")
    p.add_argument("--patience", type=int, default=config.sl1_defaults.PATIENCE,
                   help=f"Baseline Newton stagnation early-stop window; 0 disables "
                        f"(default: {config.sl1_defaults.PATIENCE})")
    p.add_argument("--stag_tol", type=float, default=config.sl1_defaults.STAG_TOL,
                   help=f"Baseline stagnation tolerance (default: {config.sl1_defaults.STAG_TOL})")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; auto-falls back to CPU")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel worker processes over (value × seed) trials "
                        "(1=serial [default], -1=all cores; forced to 1 under --gpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    param_values = args.values if args.values else _DEFAULT_VALUES[args.param]
    if args.param in _INT_PARAMS:
        param_values = [int(v) for v in param_values]

    print("=" * 60)
    print("SL1-ESN Centralized Sensitivity Analysis")
    print("=" * 60)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Parameter : {args.param}")
    print(f"  Values    : {param_values}")
    print(f"  n_seeds   : {args.n_seeds}")
    if args.epochs:
        print(f"  Epochs    : {args.epochs}")
    print()

    sl1_base = {
        "thres":            args.thres,
        "alpha_init":       args.alpha_init,
        "alpha_multiplier": args.alpha_multiplier,
        "patience":         args.patience,
        "stag_tol":         args.stag_tol,
    }

    results = sweep(
        dataset      = args.dataset,
        param_name   = args.param,
        param_values = param_values,
        n_seeds      = args.n_seeds,
        epochs       = args.epochs,
        val_frac     = args.val_frac,
        base_seed    = args.seed,
        use_cache    = not args.no_cache,
        sl1_base     = sl1_base,
        use_gpu      = args.use_gpu,
        n_jobs       = args.n_jobs,
        eval_split   = args.eval_split,
    )

    if not results:
        print("No results collected — nothing to plot.")
        sys.exit(1)

    # Read baseline value to mark on the plot.  The SL1 schedule params are
    # uniform CLI baselines (not in the config); everything else comes from config.
    if args.param in sl1_base:
        baseline = sl1_base[args.param]
    else:
        try:
            base_cfg = _load_base_config(args.dataset)
            baseline = base_cfg.get(args.param)
        except Exception:
            baseline = None

    save_run(
        results        = results,
        param_name     = args.param,
        dataset        = args.dataset,
        out_dir        = args.out_dir,
        n_seeds        = args.n_seeds,
        base_seed      = args.seed,
        epochs         = args.epochs,
        val_frac       = args.val_frac,
        sl1_base       = sl1_base,
        baseline_value = baseline,
        eval_split     = args.eval_split,
    )

    plot_sensitivity(
        results        = results,
        param_name     = args.param,
        dataset        = args.dataset,
        out_dir        = args.out_dir,
        baseline_value = baseline,
    )

    print("\nDone.")
