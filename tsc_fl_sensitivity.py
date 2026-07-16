# tsc_fl_sensitivity.py — single-hyperparameter sensitivity analysis for FedSL1ESN (FL).
#
# Federated counterpart of tsc_centralized_sensitivity.py.  Reads base
# hyperparameters from configs/TSC_FL_settings_<reg_type>.json, then sweeps one
# chosen parameter across a value range while keeping all others fixed.  Each
# value is repeated with n_seeds different random seeds; results are averaged
# and plotted as mean ± std for both avg_acc and avg_sparsity.
#
# Each trial calls tsc_fl_eval.main() (no duplicated FL logic).  In addition to
# the reservoir / SL1 parameters shared with the centralized sweep, this script
# can also sweep FL-setting parameters:
#     n_rounds, n_clients, global_lr, local_lr
#
# Usage:
#   python tsc_fl_sensitivity.py --dataset har --param reg_param
#   python tsc_fl_sensitivity.py --dataset char --param n_clients --values 2 3 5 10 20
#   python tsc_fl_sensitivity.py --dataset har --param global_lr --n_rounds 20
#   python tsc_fl_sensitivity.py --dataset har --param sr --out_dir ./result/pic

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
from tsc_fl_eval import main as run_experiment
from utils import parallel_map

# When True, each trial reports the metrics at the final communication round
# instead of tsc_fl_eval's default best-validation-round selection. Set from the
# --report_round CLI flag before the worker pool forks, so forked workers inherit
# it. Round-50 reporting is the fixed-budget protocol used for the paper's
# federated tables and figures (the readout keeps sparsifying after val accuracy
# peaks, so best-val underreports the round-50 sparsity the text refers to).
_REPORT_FINAL_ROUND = False


# ─── Parameter groups ─────────────────────────────────────────────────────────

# Reservoir / SL1 parameters — passed to main() via param_overrides (they live
# in the per-dataset config and are merged into the flat hyperparameter dict).
_CONFIG_PARAMS = {
    "sr", "lr", "input_scaling", "reg_param", "thres",
    "alpha_init", "alpha_multiplier", "units",
}

# FL-setting parameters — passed to main() as direct keyword arguments.
_FL_RUN_PARAMS = {"n_rounds", "n_clients", "global_lr", "local_lr"}

# Parameters that must be integers.
_INT_PARAMS = {"n_rounds", "n_clients", "units"}

# Parameters whose natural scale is logarithmic (plot x-axis in log scale).
_LOG_PARAMS = {"reg_param", "thres", "input_scaling"}

_DEFAULT_VALUES: dict[str, list] = {
    # ── reservoir / SL1 ───────────────────────────────────────────────────────
    "reg_param":        [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
    "thres":            [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1],
    "input_scaling":    [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
    "sr":               [0.3, 0.6, 0.98, 1.2, 1.5, 2.0, 2.5, 3.0],
    "lr":               [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "alpha_init":       [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "alpha_multiplier": [1.1, 1.25, 1.5, 2.0, 5.0, 10.0],
    "units":            [50, 100, 200, 300, 500, 800, 1000],
    # ── FL setting ────────────────────────────────────────────────────────────
    "n_rounds":         [5, 10, 20, 30, 50],
    "n_clients":        [2, 3, 5, 10, 20, 50],
    "global_lr":        [0.1, 0.3, 0.5, 0.8, 1.0],
    "local_lr":         [0.1, 0.3, 0.5, 0.8, 1.0],
}

SWEEPABLE_PARAMS = sorted(_DEFAULT_VALUES.keys())


# ─── Base config loader ───────────────────────────────────────────────────────

def _load_base_config(dataset: str, reg_type: str) -> tuple[dict, int]:
    """Return (config entry, setting_idx) for *dataset* from TSC_FL_settings_<reg_type>.json."""
    path = os.path.join(config.paths.configs_path, f"TSC_FL_settings_{reg_type}.json")
    with open(path) as f:
        entries = json.load(f)
    for i, e in enumerate(entries):
        if e["dataset"] == dataset:
            return deepcopy(e), i
    available = [e["dataset"] for e in entries]
    raise ValueError(
        f"Dataset '{dataset}' not in TSC_FL_settings_{reg_type}.json. "
        f"Available: {available}"
    )


def _build_main_kwargs(
    base: dict,
    reg_type: str,
    setting_idx: int,
    param_name: str,
    value,
    fl_base: dict,
    seed: int,
    use_cache: bool,
    sl1_base: dict,
    use_gpu: bool = config.USE_GPU,
    use_line_search: bool = True,
) -> dict:
    """Return the kwargs for one tsc_fl_eval.main() call with *param_name*=*value*.

    Reservoir / SL1 parameters travel through ``param_overrides``; FL-setting
    parameters are passed to main() directly.  Non-swept reservoir params keep
    their config baseline; the SL1 schedule (thres / alpha_init / alpha_multiplier)
    uses the uniform *sl1_base* (CLI) baseline; FL-setting params use *fl_base*.
    """
    # ── Reservoir overrides (baseline from config) + SL1 schedule (from CLI) ────
    overrides = {
        "sr":            base["sr"],
        "lr":            base["lr"],
        "input_scaling": base["input_scaling"],
        "reg_param":     base["reg_param"],
        "thres":         sl1_base["thres"],
    }
    if reg_type == "sl1":
        overrides["alpha_init"]       = sl1_base["alpha_init"]
        overrides["alpha_multiplier"] = sl1_base["alpha_multiplier"]

    # ── FL-setting run params (baseline values from fl_base) ───────────────────
    run = dict(fl_base)   # n_rounds, n_clients, global_lr, local_lr

    # ── Apply the swept value to whichever group owns the parameter ────────────
    cast = (lambda v: int(v)) if param_name in _INT_PARAMS else (lambda v: float(v))
    if param_name in _FL_RUN_PARAMS:
        run[param_name] = cast(value)
    else:
        overrides[param_name] = cast(value)

    return dict(
        reg_type        = reg_type,
        seed            = seed,
        setting_idx     = setting_idx,
        use_cache       = use_cache,
        exp_suffix      = f"sens_{param_name}",
        param_overrides = overrides,
        patience        = 0,
        use_gpu         = use_gpu,
        use_line_search = use_line_search,
        verbose         = False,   # silence per-round trace; sweep drives many runs
        **run,
    )


# ─── Parallel trial worker ─────────────────────────────────────────────────────

def _fl_sens_worker(kwargs: dict, dataset: str):
    """Run one FL trial (one run_experiment call) in a pool worker.

    Each trial is a full, self-contained tsc_fl_eval.main() call whose kwargs are
    picklable, so no fork-shared state is needed (main() reloads data from cache).
    Returns (avg_acc, avg_sparsity), or ("__error__", message) so the parent can
    log the failure and record NaN.
    """
    try:
        res = run_experiment(**kwargs)
        rd  = res.get(dataset, {"acc": float("nan"), "sparsity": float("nan")})
        if _REPORT_FINAL_ROUND and rd.get("round_history"):
            last = rd["round_history"][-1]
            return (last["test_acc"], last["sparsity"])
        return (rd["acc"], rd["sparsity"])
    except Exception as exc:                                  # noqa: BLE001
        return ("__error__", str(exc))


# ─── Sweep ────────────────────────────────────────────────────────────────────

def sweep(
    dataset: str,
    param_name: str,
    param_values: list,
    reg_type: str   = "sl1",
    n_seeds: int    = 3,
    fl_base: dict   = None,
    base_seed: int  = 42,
    use_cache: bool = True,
    sl1_base: dict  = None,
    use_gpu: bool   = config.USE_GPU,
    use_line_search: bool = True,
    n_jobs: int     = 1,
) -> list[dict]:
    """Sweep *param_name* over *param_values*; return one result dict per value.

    Each result dict has keys:
        param_value, mean_val_acc, std_val_acc,
        mean_val_sparsity, std_val_sparsity
    """
    base_cfg, setting_idx = _load_base_config(dataset, reg_type)

    if sl1_base is None:
        sl1_base = {
            "thres":            config.sl1_defaults.THRES,
            "alpha_init":       config.sl1_defaults.ALPHA_INIT,
            "alpha_multiplier": config.sl1_defaults.ALPHA_MULTIPLIER,
        }

    # GPU + multiprocessing oversubscribe a single device; serialise in that case
    # (mirrors tsc_centralized_search.py).
    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"to avoid oversubscribing the device.")
        n_jobs = 1

    total = len(param_values) * n_seeds

    # ── Build the flat (value × seed) task list ───────────────────────────────
    tasks = []                                     # flat (kwargs, dataset)
    for i, v in enumerate(param_values):
        for s in range(n_seeds):
            seed   = base_seed + i * n_seeds + s
            kwargs = _build_main_kwargs(
                base_cfg, reg_type, setting_idx,
                param_name, v, fl_base, seed, use_cache, sl1_base,
                use_gpu=use_gpu, use_line_search=use_line_search,
            )
            tasks.append((kwargs, dataset))

    print(f"\nBase config loaded from TSC_FL_settings_{reg_type}.json")
    print(f"FL base: {fl_base}")
    print(f"Sweeping '{param_name}' over {len(param_values)} values "
          f"× {n_seeds} seeds = {total} trials  (n_jobs={n_jobs})\n")

    flat = parallel_map(_fl_sens_worker, tasks, n_jobs,
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
            avg_acc, avg_sp = res
            accs.append(avg_acc)
            sparsities.append(avg_sp)

        if accs:
            results.append({
                "param_value":       v,
                "mean_val_acc":      float(np.nanmean(accs)),
                "std_val_acc":       float(np.nanstd(accs)),
                "mean_val_sparsity": float(np.nanmean(sparsities)),
                "std_val_sparsity":  float(np.nanstd(sparsities)),
            })

    return results


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_sensitivity(
    results: list[dict],
    param_name: str,
    dataset: str,
    out_dir: str,
    reg_type: str = "sl1",
    baseline_value: float | None = None,
) -> None:
    """Save a dual-y-axis sensitivity plot (avg_acc and avg_sparsity vs param).

    *baseline_value*: if given, draws a vertical dashed line at the config value
    so it is easy to see where the current setting sits in the sweep.
    """
    xs      = [r["param_value"]       for r in results]
    accs    = [r["mean_val_acc"]      for r in results]
    acc_std = [r["std_val_acc"]       for r in results]
    sps     = [r["mean_val_sparsity"] for r in results]
    sp_std  = [r["std_val_sparsity"]  for r in results]

    color_acc = "#2166ac"   # blue  — avg_acc
    color_sp  = "#d6604d"   # orange-red — avg_sparsity

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # ── avg_acc (left y-axis) ─────────────────────────────────────────────────
    ax1.plot(xs, accs, color=color_acc, marker="o", linewidth=2.0,
             zorder=3, label="avg_acc (%)")
    ax1.fill_between(
        xs,
        [a - s for a, s in zip(accs, acc_std)],
        [a + s for a, s in zip(accs, acc_std)],
        color=color_acc, alpha=0.15, zorder=2,
    )

    # ── avg_sparsity (right y-axis) ───────────────────────────────────────────
    ax2.plot(xs, sps, color=color_sp, marker="s", linewidth=2.0,
             linestyle="--", zorder=3, label="avg_sparsity (%)")
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
        # Integer-valued params (n_rounds/n_clients/units): tick only at the
        # sampled integers so the axis never shows fractional values like 7.5.
        ax1.set_xticks(sorted({int(x) for x in xs}))

    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel("avg_acc (%)", color=color_acc, fontsize=12)
    ax2.set_ylabel("avg_sparsity (%)", color=color_sp, fontsize=12)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_sp)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=10)

    plt.title(
        f"FedSL1ESN ({reg_type}, FL) — sensitivity to '{param_name}'\n"
        f"Dataset: {dataset}",
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    stem = f"fl_sensitivity_{dataset}_{reg_type}_{param_name}"
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
    reg_type: str,
    *,
    n_seeds: int,
    base_seed: int,
    fl_base: dict,
    sl1_base: dict,
    use_line_search: bool,
    baseline_value=None,
) -> str:
    """Write the full run (all hyperparameters + per-value results) to JSON.

    The schema is shared with the centralized sweep and consumed by the MATLAB
    plotting scripts in ``matlab/`` (``plot_sensitivity.m``). Returns the path.
    """
    payload = {
        "kind":           "fl",
        "dataset":        dataset,
        "reg_type":       reg_type,
        "param_name":     param_name,
        "is_integer":     param_name in _INT_PARAMS,
        "is_log":         param_name in _LOG_PARAMS,
        "acc_label":      "avg_acc (%)",
        "sparsity_label": "avg_sparsity (%)",
        "title":          f"FedSL1ESN ({reg_type}, FL) — sensitivity to '{param_name}'",
        "baseline_value": baseline_value,
        # ── All hyperparameters that define the run ───────────────────────────
        "settings": {
            "n_seeds":         n_seeds,
            "base_seed":       base_seed,
            "use_line_search": use_line_search,
            **fl_base,
            **sl1_base,
        },
        # ── Per-value swept results ───────────────────────────────────────────
        "results": results,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"fl_sensitivity_{dataset}_{reg_type}_{param_name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Federated FedSL1ESN single-hyperparameter sensitivity analysis.\n"
            "Reads base config from configs/TSC_FL_settings_<reg_type>.json, sweeps\n"
            "one parameter (reservoir/SL1 or FL-setting), and outputs a chart\n"
            "to result/pic/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True,
        help="Dataset name — must exist in configs/TSC_FL_settings_<reg_type>.json",
    )
    p.add_argument(
        "--reg_type", default="sl1", choices=["none", "l2", "sl1"],
        help="Regularisation type (default: sl1). alpha_* params require sl1.",
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
        help="Random seeds per parameter value — results are averaged (default: 3)",
    )
    # ── FL base settings (used for every trial unless that param is swept) ─────
    p.add_argument("--n_rounds", type=int, default=10,
                   help="Base FL rounds per trial (default: 10)")
    p.add_argument("--n_clients", type=int, default=5,
                   help="Base number of FL clients (default: 5)")
    p.add_argument("--global_lr", type=float, default=1.0,
                   help="Base server learning rate for Wout update (default: 1.0)")
    p.add_argument("--local_lr", type=float, default=1.0,
                   help="Base client learning rate for Wout update (default: 1.0)")
    # ── SL1 schedule baselines (uniform; used for non-swept SL1 params) ────────
    p.add_argument("--thres", type=float, default=config.sl1_defaults.THRES,
                   help=f"Baseline SL1 threshold (default: {config.sl1_defaults.THRES})")
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help=f"Baseline SmoothL1 alpha (default: {config.sl1_defaults.ALPHA_INIT})")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help=f"Baseline alpha growth factor (default: {config.sl1_defaults.ALPHA_MULTIPLIER})")
    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument(
        "--report_round", choices=["best_val", "final"], default="best_val",
        help="Which round's metrics to report: the best-validation round "
             "(default) or the final round (fixed-budget protocol used for the "
             "paper's federated tables/figures).",
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
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; auto-falls back to CPU")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--line_search", dest="line_search", action="store_true", default=True,
                   help="Enable Armijo line search on the server Newton step (default: on)")
    p.add_argument("--no_line_search", dest="line_search", action="store_false",
                   help="Disable the server Newton-step line search (use fixed step α=1)")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel worker processes over (value × seed) trials "
                        "(1=serial [default], -1=all cores; forced to 1 under --gpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    _REPORT_FINAL_ROUND = (args.report_round == "final")

    if args.param in {"alpha_init", "alpha_multiplier"} and args.reg_type != "sl1":
        print(f"Parameter '{args.param}' is only meaningful with --reg_type sl1.")
        sys.exit(1)

    param_values = args.values if args.values else _DEFAULT_VALUES[args.param]
    if args.param in _INT_PARAMS:
        param_values = [int(v) for v in param_values]

    fl_base = {
        "n_rounds":     args.n_rounds,
        "n_clients":    args.n_clients,
        "global_lr":    args.global_lr,
        "local_lr":     args.local_lr,
    }

    sl1_base = {
        "thres":            args.thres,
        "alpha_init":       args.alpha_init,
        "alpha_multiplier": args.alpha_multiplier,
    }

    print("=" * 60)
    print("FedSL1ESN Federated Sensitivity Analysis")
    print("=" * 60)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Reg type  : {args.reg_type}")
    print(f"  Parameter : {args.param}")
    print(f"  Values    : {param_values}")
    print(f"  n_seeds   : {args.n_seeds}")
    print(f"  FL base   : {fl_base}")
    print()

    results = sweep(
        dataset      = args.dataset,
        param_name   = args.param,
        param_values = param_values,
        reg_type     = args.reg_type,
        n_seeds      = args.n_seeds,
        fl_base      = fl_base,
        base_seed    = args.seed,
        use_cache    = not args.no_cache,
        sl1_base     = sl1_base,
        use_gpu      = args.use_gpu,
        use_line_search = args.line_search,
        n_jobs       = args.n_jobs,
    )

    if not results:
        print("No results collected — nothing to plot.")
        sys.exit(1)

    # Read baseline value to mark on the plot: from fl_base for FL-setting params,
    # from sl1_base (uniform CLI) for the SL1 schedule, else from the config entry.
    try:
        if args.param in _FL_RUN_PARAMS:
            baseline = fl_base[args.param]
        elif args.param in sl1_base:
            baseline = sl1_base[args.param]
        else:
            base_cfg, _ = _load_base_config(args.dataset, args.reg_type)
            baseline = base_cfg.get(args.param)
    except Exception:
        baseline = None

    save_run(
        results        = results,
        param_name     = args.param,
        dataset        = args.dataset,
        out_dir        = args.out_dir,
        reg_type       = args.reg_type,
        n_seeds        = args.n_seeds,
        base_seed      = args.seed,
        fl_base        = fl_base,
        sl1_base       = sl1_base,
        use_line_search = args.line_search,
        baseline_value = baseline,
    )

    plot_sensitivity(
        results        = results,
        param_name     = args.param,
        dataset        = args.dataset,
        out_dir        = args.out_dir,
        reg_type       = args.reg_type,
        baseline_value = baseline,
    )

    print("\nDone.")
