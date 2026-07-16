# tsr_centralized_eval.py — centralized ESN regression on Mackey-Glass / Lorenz.
#
# Full-data training + evaluation for the regression task.  Per-dataset
# hyperparameters come from configs/TSR_settings_<reg_type>.json by default
# (--no_best_params falls back to built-in defaults); promote
# tsr_centralized_search.py results there.
#
# Pipeline (matches exp_ideal/Run_train_test_regression.py but uses the current
# reservoirpy + Reg_Node framework):
#   1. Load dataset (mg or lorenz) via data_loader.read_data()
#   2. Run reservoir on X_train → collect all hidden states
#   3. Discard warmup transient → fit Reg_Node readout (sl1 / l2 / l1 / none)
#   4. Auto-regressively generate test predictions from the final training state
#   5. Report: train RMSE, test MSE, test RMSE, Wout sparsity (%)
#   6. Save: metrics JSON, Wout .npy, prediction plot (.png)
#
# Auto-regression:
#   starting from the last reservoir state after step 2, at each step the
#   current readout prediction is fed back as the next reservoir input.
#   This reproduces esn_regression.generate() from exp_ideal/models/esn_model.py.
#
# Usage examples:
#   python tsr_centralized_eval.py                           # all datasets × all reg_types
#   python tsr_centralized_eval.py --datasets mg --reg_types sl1
#   python tsr_centralized_eval.py --datasets mg lorenz --reg_types l2 sl1 --runs 5
#   python tsr_centralized_eval.py --datasets mg --reg_types sl1 --no_best_params

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir

import config
from data_loader import read_data
from readout_node import Reg_Node
from metrics import compute_all, valid_prediction_time, short_horizon_errors


# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_DATASETS = ["mg", "lorenz", "henon", "logistic", "rossler"]
SUPPORTED_REG_TYPES = ["none", "l1", "l2", "sl1"]

# Datasets whose multi-channel inputs span very different magnitudes and so
# benefit from per-channel z-score standardization (stats fit on the training
# split only).  Single-channel, already-bounded series (e.g. "mg") are left in
# raw units.  Because the regression task is auto-regressive — X(t)=u(t),
# y(t)=u(t+1) are the *same* variable shifted by one step — X and y share the
# same per-channel statistics, so one set of train-fit stats standardizes both
# and the auto-regressive feedback loop stays in standardized space throughout.
STANDARDIZE_DATASETS = {"lorenz", "rossler", "henon"}

# Default reservoir-state noise added to the readout's training states.
# reservoirpy 0.4 removed Reservoir(noise_rc=...), so the noise is injected on
# the harvested states (see train_and_evaluate): a regulariser that makes Wout
# tolerant of small state deviations and keeps auto-regressive generation from
# collapsing.  Must match tsr_centralized_search; carried through its best-param
# JSON so --use_best_params stays consistent.  0 disables it.
DEFAULT_NOISE_RC = 1e-3

# Chaos-aware forecast metrics for auto-regressive evaluation.  Full-horizon
# free-running MSE/RMSE saturate at the attractor's climatological variance once
# the prediction decorrelates (after a few Lyapunov times on Lorenz), so they
# cannot rank models.  We additionally report short-window errors and the valid
# prediction time (VPT) — how long the closed loop tracks the true trajectory.
SHORT_HORIZONS = (25, 50, 100, 200)   # steps; short-window MSE/RMSE per horizon
VPT_THRESHOLD  = 0.4                   # normalized-error threshold for VPT
VPT_TIEBREAK_H = 50                    # short horizon used to tie-break best run


def _fit_standardizer(train_seq: np.ndarray) -> tuple:
    """Per-channel (mean, std) from a (T, F) training sequence.

    Zero-variance channels get std=1.0 so they pass through unchanged.
    """
    mean = train_seq.mean(axis=0)
    std  = train_seq.std(axis=0)
    std  = np.where(std == 0, 1.0, std)
    return mean, std

# Conservative defaults (taken from exp_ideal best-param results).
# Run tsr_centralized_search.py first to find better params for your setup.
_DEFAULT_HPARAMS = {
    "mg": {
        "none": dict(units=500, sr=0.90, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "l1":   dict(units=500, sr=0.90, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "l2":   dict(units=500, sr=0.90, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "sl1":  dict(units=500, sr=0.90, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
    },
    "lorenz": {
        "none": dict(units=500, sr=1.20, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "l1":   dict(units=500, sr=1.20, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "l2":   dict(units=500, sr=1.20, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
        "sl1":  dict(units=500, sr=1.20, lr=0.30, input_scaling=1.0,  reg_param=1e-1, warmup=100),
    },
    # Generated chaotic benchmarks (data_loader.read_data_{henon,logistic,rossler}).
    # Conservative fallbacks — run tsr_centralized_search.py to tune per dataset.
    "henon": {
        rt: dict(units=500, sr=0.90, lr=0.60, input_scaling=1.0, reg_param=1e-2, warmup=100)
        for rt in ("none", "l1", "l2", "sl1")
    },
    "logistic": {
        rt: dict(units=500, sr=0.90, lr=0.70, input_scaling=1.0, reg_param=1e-2, warmup=100)
        for rt in ("none", "l1", "l2", "sl1")
    },
    "rossler": {
        rt: dict(units=500, sr=1.10, lr=0.30, input_scaling=1.0, reg_param=1e-1, warmup=100)
        for rt in ("none", "l1", "l2", "sl1")
    },
}


# ─── Core helpers ─────────────────────────────────────────────────────────────

def _collect_states(X_train: np.ndarray, reservoir: Reservoir,
                    warmup: int) -> tuple:
    """Run reservoir on the full training sequence; return all states.

    Args:
        X_train:   Training input, shape (T, input_dim).
        reservoir: Initialized reservoirpy Reservoir node.
        warmup:    Number of initial transient steps to discard.

    Returns:
        all_states:   (T, units)         — used to seed auto-regression
        train_states: (T-warmup, units)  — used to fit the readout
    """
    reservoir.reset()
    all_states = reservoir.run(X_train)   # (T, units); state maintained for next call
    return all_states, all_states[warmup:]


def _autoregressive_generate(n_steps: int, last_state: np.ndarray,
                              reservoir: Reservoir,
                              Wout: np.ndarray) -> np.ndarray:
    """Generate *n_steps* predictions in auto-regressive mode.

    Reproduces the logic in exp_ideal/models/esn_model.py:esn_regression.generate().

    The reservoir state must already be at the end of the training sequence
    (i.e., _collect_states was just called and reservoir.reset() was NOT called
    afterwards).  At each step:
        pred(t)   = state(t-1) @ Wout
        state(t)  = reservoir.run(pred(t).reshape(1, -1))   [one-step update]

    Args:
        n_steps:     Number of steps to generate.
        last_state:  Final reservoir state from training, shape (units,).
        reservoir:   Reservoir node whose internal state continues from training.
        Wout:        Readout weight matrix, shape (units, output_dim).

    Returns:
        preds: np.ndarray of shape (n_steps, output_dim).
    """
    preds         = []
    current_state = last_state           # (units,)

    for _ in range(n_steps):
        pred = current_state @ Wout      # (output_dim,)
        preds.append(pred.copy())
        new_states    = reservoir.run(pred.reshape(1, -1))   # (1, units)
        current_state = new_states[0]

    return np.array(preds)              # (n_steps, output_dim)


# ─── Training and evaluation ──────────────────────────────────────────────────

def train_and_evaluate(
    dataset_name: str,
    reg_type: str,
    units: int   = 500,
    sr: float    = 1.0,
    lr: float    = 0.3,
    input_scaling: float = 1.0,
    reg_param: float     = 1e-1,
    warmup: int  = 100,
    noise_rc: float = DEFAULT_NOISE_RC,
    thres: float = config.sl1_defaults.THRES,
    alpha_init: float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    seed: int    = 42,
    runs: int    = 1,
    verbose: bool = True,
    use_gpu: bool = config.USE_GPU,
) -> dict:
    """Train centralized ESN regression over *runs* independent runs.

    Each run uses a different random seed for the reservoir initialisation.
    The run with the lowest test RMSE is returned.

    Args:
        dataset_name:  "mg" or "lorenz".
        reg_type:      One of "none", "l1", "l2", "sl1".
        units:         Reservoir size.
        sr:            Spectral radius.
        lr:            Leaking rate.
        input_scaling: Input weight scaling factor.
        reg_param:     Regularisation strength (ignored for reg_type="none").
        warmup:        Transient steps discarded at the start of training.
        seed:          Base random seed (run i uses seed+i).
        runs:          Number of independent runs.
        verbose:       Print per-run metrics.

    Returns:
        dict with keys: run, mse, rmse, rmse_train, sparsity,
                        preds (np.ndarray), trues (np.ndarray), Wout (np.ndarray).
    """
    X_train, y_train, X_test, y_test = read_data(dataset_name)
    output_dim = y_train.shape[1]

    # Per-channel standardization (fit on training input only).  X and y share
    # statistics because y is X shifted by one step, so the same (mean, std)
    # transforms both and the auto-regressive loop runs in standardized space.
    # Metrics, predictions and plots are de-standardized back to raw units so
    # they stay comparable to the un-standardized datasets and to past results.
    # Raw-unit training targets, kept before standardization to anchor the
    # scale-free metrics (MASE's naive baseline and DA's first anchor).
    y_train_raw = y_train.copy()

    standardize = dataset_name in STANDARDIZE_DATASETS
    if standardize:
        mean, std = _fit_standardizer(X_train)
        X_train = (X_train - mean) / std
        y_train = (y_train - mean) / std
        X_test  = (X_test  - mean) / std
        y_test  = (y_test  - mean) / std

    # Best run is chosen by valid prediction time (VPT, higher is better) — the
    # full-horizon RMSE used previously saturates at the climatological floor on
    # chaotic data and cannot rank runs.  Ties broken by short-horizon RMSE.
    best_result = None
    best_score  = (-np.inf, np.inf)   # (vpt_steps, -short_rmse) maximized lexically

    for run in range(runs):
        reservoir = Reservoir(
            units         = units,
            sr            = sr,
            lr            = lr,
            input_scaling = input_scaling,
            seed          = seed + run,
        )
        reservoir.run(np.zeros((1, X_train.shape[1])))


        all_states, train_states = _collect_states(X_train, reservoir, warmup)
        y_reg = y_train[warmup:]          # targets aligned with train_states

        # State-noise regularisation: fit Wout on slightly perturbed states so
        # the closed-loop generation tolerates the deviations it accumulates.
        # Only the fit copy is noised; the generation seed (all_states[-1]) and
        # the reported train RMSE use the clean states.
        fit_states = train_states
        if noise_rc > 0:
            fit_states = train_states + noise_rc * np.random.RandomState(
                seed + run).standard_normal(train_states.shape)

        # Fit readout
        # thres / alpha_init / alpha_multiplier drive the offline sl1 solver
        # (solve_newton_mse_l1): α continuation + per-step soft-threshold by thres,
        # matching the classification path and the FL server.
        reg_node = Reg_Node(
            reg_param        = reg_param,
            reg_type         = reg_type,
            thres            = thres,
            alpha_init       = alpha_init,
            alpha_multiplier = alpha_multiplier,
            input_dim        = units,
            output_dim       = output_dim,
            use_gpu          = use_gpu,
        )
        reg_node.fit(fit_states, y_reg, isFL=False)

        # Auto-regressive prediction on test set (in standardized space if
        # standardize=True; predictions are fed back as inputs each step)
        preds = _autoregressive_generate(
            n_steps     = len(X_test),
            last_state  = all_states[-1],
            reservoir   = reservoir,
            Wout        = reg_node.Wout,
        )

        # De-standardize back to raw units so all reported/saved quantities
        # are comparable across datasets and to prior results.
        y_pred_train = train_states @ reg_node.Wout
        if standardize:
            y_pred_train = y_pred_train * std + mean
            y_reg_raw    = y_reg * std + mean
            preds        = preds * std + mean
            trues        = y_test * std + mean
        else:
            y_reg_raw = y_reg
            trues     = y_test

        # Training RMSE (post-warmup portion only)
        e_train  = float(np.sqrt(np.mean((y_pred_train - y_reg_raw) ** 2)))

        # Full-horizon error (kept for continuity; saturates on chaotic data).
        mse      = float(np.mean((preds - trues) ** 2))
        rmse     = float(np.sqrt(mse))
        sparsity = float((reg_node.Wout == 0).mean() * 100)

        # Chaos-aware forecast metrics (all on raw-unit preds/trues).
        fc = compute_all(preds, trues, train_target=y_train_raw)
        fc["vpt_steps"] = valid_prediction_time(preds, trues,
                                                threshold=VPT_THRESHOLD)
        fc.update(short_horizon_errors(preds, trues, SHORT_HORIZONS))

        # Short-horizon RMSE used for run selection / tie-break (falls back to
        # the full-horizon RMSE if the test set is shorter than VPT_TIEBREAK_H).
        short_rmse = fc.get(f"rmse_{VPT_TIEBREAK_H}", rmse)

        if verbose:
            print(f"  run {run+1}/{runs}: train_rmse={e_train:.6f}  "
                  f"vpt={fc['vpt_steps']}  rmse_{VPT_TIEBREAK_H}={short_rmse:.4f}  "
                  f"full_rmse={rmse:.4f}  var_ratio={fc['var_ratio']:.3f}  "
                  f"sparsity={sparsity:.1f}%")

        # Maximize VPT, tie-break by lower short-horizon RMSE.
        score = (fc["vpt_steps"], -short_rmse)
        if score > best_score:
            best_score  = score
            best_result = {
                "run":        run,
                "mse":        mse,
                "rmse":       rmse,
                "rmse_train": e_train,
                "sparsity":   sparsity,
                "metrics":    fc,
                "preds":      preds.copy(),
                "trues":      trues.copy(),
                "Wout":       reg_node.Wout.copy(),
            }

    return best_result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_predictions(preds: np.ndarray, trues: np.ndarray,
                     dataset_name: str, reg_type: str,
                     save_dir: str) -> str:
    """Plot true vs predicted time series for each output dimension.

    Mackey-Glass (1-D): single panel.
    Lorenz (3-D):       three stacked panels, one per spatial component.

    Args:
        preds:        Predicted values, shape (T, output_dim).
        trues:        Ground truth values, shape (T, output_dim).
        dataset_name: Used in the figure title and filename.
        reg_type:     Used in the figure title and filename.
        save_dir:     Directory to save the .png figure.

    Returns:
        Absolute path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    T          = len(trues)
    t          = np.arange(T)
    output_dim = trues.shape[1]
    overall_rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    if output_dim == 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, trues[:, 0], color="steelblue", lw=0.9, label="Ground Truth")
        ax.plot(t, preds[:, 0], color="tomato",    lw=0.9, ls="--", label="Prediction")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.set_title(
            f"{dataset_name.upper()} — {reg_type}   |   Test RMSE = {overall_rmse:.6f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        dim_labels = ["x", "y", "z"] + [f"dim{i}" for i in range(3, output_dim)]
        fig, axes = plt.subplots(output_dim, 1,
                                 figsize=(12, 3 * output_dim), sharex=True)
        for i, ax in enumerate(axes):
            dim_rmse = float(np.sqrt(np.mean((preds[:, i] - trues[:, i]) ** 2)))
            ax.plot(t, trues[:, i], color="steelblue", lw=0.9, label="Ground Truth")
            ax.plot(t, preds[:, i], color="tomato",    lw=0.9, ls="--", label="Prediction")
            ax.set_ylabel(dim_labels[i])
            ax.set_title(f"Component {dim_labels[i]}   RMSE = {dim_rmse:.6f}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time step")
        fig.suptitle(
            f"{dataset_name.upper()} — {reg_type}   |   Overall Test RMSE = {overall_rmse:.6f}",
            fontsize=12,
        )

    plt.tight_layout()
    tag       = f"{dataset_name}_{reg_type}"
    plot_path = os.path.join(save_dir, f"{tag}_predictions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    reg_type: str,
    hparams: dict,
    save_dir: str,
    runs: int  = 5,
    seed: int  = 42,
    verbose: bool = True,
    save_model: bool = False,
) -> dict:
    """Run one (dataset, reg_type) experiment and persist all outputs.

    Saves:
        <save_dir>/<dataset>_<reg_type>_metrics.json   — scalar metrics + hparams
        <save_dir>/<dataset>_<reg_type>_Wout.npy       — trained readout weights
        <save_dir>/<dataset>_<reg_type>_preds.npy      — test predictions
        <save_dir>/<dataset>_<reg_type>_trues.npy      — test ground truth
        <save_dir>/<dataset>_<reg_type>_predictions.png — prediction plot

    Args:
        hparams:  Dict with keys: units, sr, lr, input_scaling, reg_param, warmup.

    Returns:
        Scalar metrics dict (same fields as metrics JSON).
    """
    print(f"\n{'─'*60}")
    print(f"Dataset: {dataset_name}  |  reg_type: {reg_type}")
    for k, v in hparams.items():
        print(f"  {k}: {v}")
    print(f"{'─'*60}")

    t0     = time.time()
    result = train_and_evaluate(
        dataset_name  = dataset_name,
        reg_type      = reg_type,
        units         = hparams.get("units",         500),
        sr            = hparams.get("sr",            1.0),
        lr            = hparams.get("lr",            0.3),
        input_scaling = hparams.get("input_scaling", 1.0),
        reg_param     = hparams.get("reg_param",     1e-1),
        warmup        = hparams.get("warmup",        100),
        noise_rc      = hparams.get("noise_rc",      DEFAULT_NOISE_RC),
        thres            = hparams.get("thres",            config.sl1_defaults.THRES),
        alpha_init       = hparams.get("alpha_init",       config.sl1_defaults.ALPHA_INIT),
        alpha_multiplier = hparams.get("alpha_multiplier", config.sl1_defaults.ALPHA_MULTIPLIER),
        seed          = seed,
        runs          = runs,
        verbose       = verbose,
        use_gpu       = hparams.get("use_gpu", config.USE_GPU),
    )
    elapsed = time.time() - t0

    fc = result["metrics"]
    print(f"\nBest run ({result['run']+1}/{runs}, selected by VPT):")
    print(f"  Train RMSE        : {result['rmse_train']:.6f}")
    print(f"  VPT (steps, <{VPT_THRESHOLD}) : {fc['vpt_steps']}")
    for h in SHORT_HORIZONS:
        if f"mse_{h}" in fc:
            print(f"  MSE @ {h:<4d} steps    : {fc[f'mse_{h}']:.6f}  "
                  f"(RMSE {fc[f'rmse_{h}']:.6f})")
    print(f"  Test MSE (full)   : {result['mse']:.6f}   "
          f"← saturates at climatology on chaotic data")
    print(f"  Test RMSE (full)  : {result['rmse']:.6f}")
    print(f"  nRMSE / DA        : {fc['nrmse']:.4f} / {fc['da']:.4f}")
    print(f"  var_ratio (≈1 ok) : {fc['var_ratio']:.4f}")
    print(f"  Wout sparsity     : {result['sparsity']:.1f}%")
    print(f"  Elapsed           : {elapsed:.1f}s")

    # ── Persist outputs ──────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{dataset_name}_{reg_type}"

    fc = result["metrics"]
    metrics = {
        "dataset":      dataset_name,
        "reg_type":     reg_type,
        "standardized": dataset_name in STANDARDIZE_DATASETS,
        "hparams":      hparams,
        "rmse_train":   result["rmse_train"],
        "mse_test":     result["mse"],
        "rmse_test":    result["rmse"],
        # Chaos-aware forecast metrics (the meaningful ones for auto-regression).
        "vpt_steps":      fc["vpt_steps"],
        "vpt_threshold":  VPT_THRESHOLD,
        "mse_short":      {str(h): fc[f"mse_{h}"]  for h in SHORT_HORIZONS if f"mse_{h}"  in fc},
        "rmse_short":     {str(h): fc[f"rmse_{h}"] for h in SHORT_HORIZONS if f"rmse_{h}" in fc},
        "nrmse_test":     fc["nrmse"],
        "mase_test":      fc["mase"],
        "da_test":        fc["da"],
        "var_ratio_test": fc["var_ratio"],
        "sparsity_pct": result["sparsity"],
        "best_run":     result["run"],
        "total_runs":   runs,
    }
    with open(os.path.join(save_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    if save_model:
        np.save(os.path.join(save_dir, f"{tag}_Wout.npy"),  result["Wout"])
        np.save(os.path.join(save_dir, f"{tag}_preds.npy"), result["preds"])
        np.save(os.path.join(save_dir, f"{tag}_trues.npy"), result["trues"])

    plot_path = plot_predictions(
        result["preds"], result["trues"],
        dataset_name, reg_type, save_dir,
    )
    print(f"  Plot saved → {plot_path}")

    return metrics


# ─── Hyperparameter loading ───────────────────────────────────────────────────

def _load_best_params(dataset_name: str, reg_type: str) -> dict:
    """Load tuned hparams from configs/TSR_settings_<reg_type>.json; else defaults.

    The settings files hold the tuned per-dataset hyperparameters (promoted from
    tsr_centralized_search.py's result/tsr_centralized_best_<reg_type>.json).
    """
    entry = config.load_settings("tsr", reg_type).get(dataset_name)
    if entry:
        print(f"  [hparams] loaded from configs/TSR_settings_{reg_type}.json")
        return {k: entry[k] for k in
                ("units", "sr", "lr", "input_scaling", "reg_param", "warmup", "noise_rc")
                if k in entry}
    print(f"  [hparams] no TSR_settings entry for {dataset_name}/{reg_type} "
          f"— using defaults")
    return _DEFAULT_HPARAMS[dataset_name][reg_type].copy()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centralized ESN regression — Mackey-Glass and Lorenz"
    )
    p.add_argument("--datasets",      nargs="+", default=["all"],
                   choices=SUPPORTED_DATASETS + ["all"],
                   help="Dataset name(s); 'all' expands to every supported "
                        "dataset (default: all)")
    p.add_argument("--reg_types",     nargs="+", default=["all"],
                   choices=SUPPORTED_REG_TYPES + ["all"],
                   help="Regularization type(s); 'all' expands to every "
                        "supported type (default: all)")
    # Hyperparameter overrides (applied on top of defaults / best_params)
    p.add_argument("--units",         type=int,   default=None)
    p.add_argument("--sr",            type=float, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--input_scaling", type=float, default=None)
    p.add_argument("--reg_param",     type=float, default=None)
    p.add_argument("--warmup",        type=int,   default=None)
    p.add_argument("--noise_rc",      type=float, default=None,
                   help="Gaussian noise on the readout's training states "
                        "(stabilises auto-regressive generation; 0 disables). "
                        f"Default {DEFAULT_NOISE_RC} unless set by --use_best_params")
    # SL1 sparsification schedule — drives the offline Newton solver
    # (solve_newton_mse_l1: α continuation + per-step soft-threshold by thres)
    # as well as the isFL SGD path.
    p.add_argument("--thres",            type=float, default=None,
                   help="SL1 soft-threshold magnitude (per-step pruning)")
    p.add_argument("--alpha_init",       type=float, default=None,
                   help="Initial SmoothL1 alpha (α continuation start)")
    p.add_argument("--alpha_multiplier", type=float, default=None,
                   help="Alpha growth factor (α continuation rate)")
    # Run control
    p.add_argument("--runs",          type=int,   default=5,
                   help="Independent runs per experiment; best RMSE reported (default: 5)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--save_dir",      default="./result/regression",
                   help="Output directory for all saved files (default: ./result/regression)")
    p.add_argument("--save_model",    action="store_true",
                   help="Also save Wout/preds/trues .npy (off by default; "
                        "metrics JSON and the plot are always written)")
    p.add_argument("--no_best_params", dest="use_best_params",
                   action="store_false", default=True,
                   help="Ignore configs/TSR_settings_<reg_type>.json and use "
                        "built-in defaults (best params are ON by default)")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the offline SL1 (L1General) solver; "
                        "auto-falls back to CPU if CuPy/CUDA is unavailable")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    datasets  = (SUPPORTED_DATASETS  if "all" in args.datasets
                 else list(dict.fromkeys(args.datasets)))
    reg_types = (SUPPORTED_REG_TYPES if "all" in args.reg_types
                 else list(dict.fromkeys(args.reg_types)))

    # CLI overrides (only non-None values are applied)
    cli_overrides = {k: v for k, v in {
        "units":         args.units,
        "sr":            args.sr,
        "lr":            args.lr,
        "input_scaling": args.input_scaling,
        "reg_param":     args.reg_param,
        "warmup":        args.warmup,
        "noise_rc":      args.noise_rc,
        "thres":            args.thres,
        "alpha_init":       args.alpha_init,
        "alpha_multiplier": args.alpha_multiplier,
        "use_gpu":          args.use_gpu,
    }.items() if v is not None}

    all_metrics = []
    for ds in datasets:
        for rt in reg_types:
            hparams = (
                _load_best_params(ds, rt)
                if args.use_best_params
                else _DEFAULT_HPARAMS[ds][rt].copy()
            )
            hparams.update(cli_overrides)

            m = run_experiment(
                dataset_name = ds,
                reg_type     = rt,
                hparams      = hparams,
                save_dir     = args.save_dir,
                runs         = args.runs,
                seed         = args.seed,
                save_model   = args.save_model,
            )
            all_metrics.append(m)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CENTRALIZED ESN REGRESSION — SUMMARY")
    print("=" * 72)
    hkey = str(VPT_TIEBREAK_H)
    hdr = (f"{'Dataset':<10} {'RegType':<6} "
           f"{'RMSE_train':>11} {'VPT':>6} {f'MSE@{hkey}':>11} "
           f"{'MSE_full':>11} {'var_ratio':>10} {'Sparsity':>9}")
    print(hdr)
    print("─" * len(hdr))
    for m in all_metrics:
        mse_short = m.get("mse_short", {}).get(hkey, float("nan"))
        print(f"{m['dataset']:<10} {m['reg_type']:<6} "
              f"{m['rmse_train']:>11.6f} {m['vpt_steps']:>6d} {mse_short:>11.4f} "
              f"{m['mse_test']:>11.4f} {m['var_ratio_test']:>10.4f} "
              f"{m['sparsity_pct']:>8.1f}%")

    print(f"\nAll outputs saved to: {args.save_dir}/")
