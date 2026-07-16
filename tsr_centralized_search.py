# tsr_centralized_search.py — random hyperparameter search for centralized ESN regression.
#
# Pairs with tsr_centralized_eval.py (full-data training/evaluation).
#
# Searches sr, lr, input_scaling, reg_param for each (dataset, reg_type) pair
# using windowed auto-regressive validation on the training split.  Every trial
# is scored on five forecasting metrics (see metrics.py) — rmse, nrmse, mase,
# directional accuracy (da), variance ratio — plus Wout sparsity.  Which metric
# drives selection is chosen with --select_metric; the score is:
#
#   score = -(1 - w) * loss(select_metric) + w * sparsity / 100
#         w = sparsity_weight   (0 → pure accuracy [default], 1 → pure sparsity)
#
# loss() turns any metric into a "lower is better" quantity (rmse/nrmse/mase as
# is, da → 1−da, var_ratio → |log VR|), so the negated form keeps "higher score
# is always better".  select_metric="composite" is a scale-free weighted blend
# of nrmse/mase/da/var_ratio (weights via --w_*).  With the defaults
# (select_metric=rmse, sparsity_weight=0) the score reduces to -rmse, matching
# the previous behaviour and the other two search scripts.
#
# Note: the training objective is unchanged (MSE + reg, solved by Reg_Node);
# only the trial-*selection* metric is configurable here.
#
# Results:
#   ./result/tsr_centralized_search.csv           — one row per trial (incremental)
#   ./result/tsr_centralized_best_<reg_type>.json — best configs per (dataset, reg_type)
#
# Usage:
#   python tsr_centralized_search.py
#   python tsr_centralized_search.py --datasets mg --reg_types sl1 l2 --steps 200
#   python tsr_centralized_search.py --sparsity_weight 0.3
#   python tsr_centralized_search.py --select_metric mase
#   python tsr_centralized_search.py --select_metric composite --w_da 0.4 --w_nrmse 0.3

import argparse
import json
import os
import random
import time
from copy import deepcopy

import numpy as np

from reservoirpy.nodes import Reservoir

import config
import pruning
from data_loader import read_data
from readout_node import Reg_Node
from tsr_centralized_eval import (
    _autoregressive_generate,
    _fit_standardizer,
    STANDARDIZE_DATASETS,
    VPT_THRESHOLD,
    VPT_TIEBREAK_H,
)
from utils import log_uniform, init_csv, append_csv_row, BestTracker, parallel_map
from metrics import (
    compute_all, selection_loss, SELECT_METRICS,
    valid_prediction_time, short_horizon_errors,
)


# ─── Dataset registries ───────────────────────────────────────────────────────

_TSR_DATASETS = ["mg", "lorenz", "henon", "logistic", "rossler"]
_REG_TYPES    = ["none", "l2", "l1", "sl1"]   # "none" = plain OLS (reg_param inert)

# Spectral-radius search bounds, aligned with the classification task's sr range
# (tsc_centralized_search samples sr ~ uniform(0.3, 3.0)); the random search
# samples sr ~ uniform(min, max) and the PSO uses the same min/max as box
# bounds.  The wide span lets the search explore beyond the edge-of-chaos band
# (mg≈0.9, lorenz≈1.2); the auto-regressive selection metrics reject configs
# whose closed feedback loop diverges (high sr) or over-damps (low sr).
_SR_SEARCH_RANGE = [0.30, 3.00]

# Default reservoir-state noise added to the readout's training states (see
# _eval_window).  reservoirpy 0.4 dropped Reservoir(noise_rc=...), so we inject
# the noise on the harvested states instead — a regulariser that makes Wout
# tolerate small state deviations and keeps the free-running loop from
# collapsing.  Tunable via --noise_rc; 0 disables it.
_DEFAULT_NOISE_RC = 1e-3


# ─── Single validation window ─────────────────────────────────────────────────

def _eval_window(X_tr, y_tr, X_ts, y_ts,
                 units, sr, lr, input_scaling,
                 reg_param, reg_type, warmup, seed,
                 standardize: bool = False,
                 noise_rc: float = _DEFAULT_NOISE_RC,
                 thres: float = config.sl1_defaults.THRES,
                 alpha_init: float = config.sl1_defaults.ALPHA_INIT,
                 alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
                 use_gpu: bool = config.USE_GPU) -> tuple:
    """Train on X_tr, auto-regressively predict X_ts; return (metrics, sparsity_pct).

    *metrics* is the dict from metrics.compute_all (rmse, nrmse, mase, da,
    var_ratio); MASE/DA are anchored on the de-standardized training target.

    When standardize=True, per-channel z-score stats are fit on this window's
    training portion only (X_tr) and applied to all four arrays; predictions
    are de-standardized before scoring so metrics stay in raw units — matching
    tsr_centralized_eval and avoiding validation-portion leakage.
    """
    if standardize:
        mean, std = _fit_standardizer(X_tr)
        X_tr = (X_tr - mean) / std
        y_tr = (y_tr - mean) / std
        X_ts = (X_ts - mean) / std
        y_ts = (y_ts - mean) / std

    reservoir = Reservoir(units=units, sr=sr, lr=lr,
                          input_scaling=input_scaling, seed=seed)
    # reservoir.reset()
    all_states = reservoir.run(X_tr)

    X_states   = all_states[warmup:]
    y_reg      = y_tr[warmup:]
    output_dim = y_tr.shape[1]

    # State-noise regularisation: fit Wout on slightly perturbed states so the
    # closed-loop generation tolerates the small deviations it accumulates.
    # The generation seed (all_states[-1]) stays clean — only the fit copy is
    # noised.
    if noise_rc > 0:
        X_states = X_states + noise_rc * np.random.RandomState(seed).standard_normal(X_states.shape)

    # thres / alpha_init / alpha_multiplier drive the offline sl1 solver
    # (solve_newton_mse_l1): α continuation + per-step soft-threshold by thres.
    # Only reg_param (λ) is searched here; thres is pinned (see search space).
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
    reg_node.fit(X_states, y_reg, isFL=False)

    preds    = _autoregressive_generate(len(X_ts), all_states[-1], reservoir, reg_node.Wout)
    if standardize:
        preds = preds * std + mean
        y_ts  = y_ts  * std + mean
        y_tr  = y_tr  * std + mean
    metrics  = compute_all(preds, y_ts, y_tr)
    # Chaos-aware metrics on the same auto-regressive forecast: VPT (steps
    # tracked before divergence) and short-horizon MSE (accuracy inside the
    # predictable window). These let --select_metric vpt / mse_short rank trials
    # by forecast skill instead of the climatology-saturated full-horizon rmse.
    metrics["vpt"]       = valid_prediction_time(preds, y_ts, threshold=VPT_THRESHOLD)
    # Horizon-normalized vpt ∈ [0, 1] (1 = tracked the whole forecast horizon).
    # The horizon (len(y_ts)) is constant across trials/windows, so this is a
    # pure rescaling of vpt that puts the selection loss on the same [0,1] scale
    # as sparsity/100 — see metrics._losses for why this matters to the sparsity
    # blend.  Used by --select_metric vpt with --sparsity_weight > 0.
    metrics["vpt_norm"]  = metrics["vpt"] / max(len(y_ts), 1)
    metrics["mse_short"] = short_horizon_errors(preds, y_ts, [VPT_TIEBREAK_H]).get(
        f"mse_{VPT_TIEBREAK_H}", float(metrics["rmse"] ** 2))
    sparsity = float((reg_node.Wout == 0).mean() * 100)
    return metrics, sparsity


def _eval_imp_window(X_tr, y_tr, X_ts, y_ts, cfg, target_sp,
                     units, warmup, standardize, noise_rc, seed) -> tuple:
    """Ridge + IMP baseline on one window: dense ridge fit → prune-refit to
    *target_sp* → auto-regressive forecast; returns (metrics, sparsity_pct).

    The regression analogue of the CE+IMP baseline: a dense ridge readout, then
    pruning.ridge_imp_prune to the sparsity SL1 reached, evaluated exactly like
    _eval_window so the numbers are directly comparable to the SL1 row.
    """
    if standardize:
        mean, std = _fit_standardizer(X_tr)
        X_tr = (X_tr - mean) / std
        y_tr = (y_tr - mean) / std
        X_ts = (X_ts - mean) / std
        y_ts = (y_ts - mean) / std

    reservoir  = Reservoir(units=units, sr=cfg["sr"], lr=cfg["lr"],
                           input_scaling=cfg["input_scaling"], seed=seed)
    all_states = reservoir.run(X_tr)
    X_states   = all_states[warmup:]
    y_reg      = y_tr[warmup:]
    if noise_rc > 0:
        X_states = X_states + noise_rc * np.random.RandomState(seed).standard_normal(X_states.shape)

    # Dense ridge fit (closed form), then IMP prune-refit to the SL1 sparsity.
    lam     = 1e-3
    F       = X_states.shape[1]
    W_dense = np.linalg.solve(X_states.T @ X_states + lam * np.eye(F),
                              X_states.T @ y_reg)
    W_imp   = pruning.ridge_imp_prune(X_states, y_reg, W_dense, target_sp,
                                      reg_param=lam)

    preds = _autoregressive_generate(len(X_ts), all_states[-1], reservoir, W_imp)
    if standardize:
        preds = preds * std + mean
        y_ts  = y_ts  * std + mean
        y_tr  = y_tr  * std + mean
    m = compute_all(preds, y_ts, y_tr)
    m["vpt"]       = valid_prediction_time(preds, y_ts, threshold=VPT_THRESHOLD)
    m["vpt_norm"]  = m["vpt"] / max(len(y_ts), 1)
    m["mse_short"] = short_horizon_errors(preds, y_ts, [VPT_TIEBREAK_H]).get(
        f"mse_{VPT_TIEBREAK_H}", float(m["rmse"] ** 2))
    sparsity = float((W_imp == 0).mean() * 100)
    return m, sparsity


def _imp_baseline(X, y, cfg, target_sp, len_valid, k, units, warmup,
                  standardize, noise_rc, seed) -> tuple:
    """Mean ridge+IMP metrics over the same k validation windows the search uses.

    Windows are drawn with the same seeded scheme as random_search so the
    baseline is measured on comparable slices; returns (metric_means, sparsity).
    """
    max_start    = len(X) - len_valid - 1
    valid_starts = np.random.RandomState(seed).choice(
        max_start, min(k, max_start), replace=False).tolist()
    sums = {m: 0.0 for m in ("rmse", "nrmse", "mase", "da", "var_ratio",
                             "vpt", "vpt_norm", "mse_short")}
    sp_sum = 0.0
    for idx in valid_starts:
        X_win, y_win = X[idx: idx + len_valid], y[idx: idx + len_valid]
        split = int(len_valid * 0.8)
        m, sp = _eval_imp_window(
            X_win[:split], y_win[:split], X_win[split:], y_win[split:],
            cfg, target_sp, units, warmup, standardize, noise_rc, seed)
        for key in sums:
            sums[key] += m[key]
        sp_sum += sp
    n = len(valid_starts)
    return {key: s / n for key, s in sums.items()}, sp_sum / n


# ─── Parallel combo worker ─────────────────────────────────────────────────────

# Per-dataset series shared with pool workers via fork (copy-on-write). Set in
# random_search() before each parallel_map() call so forked workers inherit X/y
# without per-task pickling of the (long) time series.
_WORKER_DATA: dict = {}


def _combo_worker(sr, lr, input_scaling, reg_param, reg_type, valid_starts,
                  len_valid, units, warmup, seed, standardize, noise_rc,
                  thres, alpha_init, alpha_multiplier, use_gpu):
    """Evaluate one (sr, lr, input_scaling, reg_param) combo over the validation
    windows; reads X/y from _WORKER_DATA. Returns (metric_means, sparsity_mean),
    or ("__error__", message) on failure so the parent records NaN and logs it.
    """
    t0 = time.time()
    try:
        X = _WORKER_DATA["X"]
        y = _WORKER_DATA["y"]
        metric_sums = {m: 0.0 for m in
                       ("rmse", "nrmse", "mase", "da", "var_ratio",
                        "vpt", "vpt_norm", "mse_short")}
        sparsity_mean = 0.0
        valid_count   = 0
        for idx in valid_starts:
            X_win = X[idx: idx + len_valid]
            y_win = y[idx: idx + len_valid]
            split = int(len_valid * 0.8)
            metrics_w, sp_w = _eval_window(
                X_win[:split], y_win[:split],
                X_win[split:], y_win[split:],
                units, sr, lr, input_scaling, reg_param,
                reg_type, warmup, seed=seed,
                standardize=standardize, noise_rc=noise_rc,
                thres=thres, alpha_init=alpha_init,
                alpha_multiplier=alpha_multiplier,
                use_gpu=use_gpu,
            )
            for m in metric_sums:
                metric_sums[m] += metrics_w[m]
            sparsity_mean += sp_w
            valid_count   += 1
        metric_means   = {m: s / valid_count for m, s in metric_sums.items()}
        sparsity_mean /= valid_count
        # Progress is shown by the parent's live bar (parallel_map(progress=...));
        # workers stay silent on success so nothing clobbers the bar.
        return metric_means, sparsity_mean
    except Exception as exc:                                  # noqa: BLE001
        print(f"    [worker] {reg_type} sr={sr:.2f} lr={lr:.3f} "
              f"is={input_scaling:.1e} rp={reg_param:.1e} ERROR in "
              f"{time.time() - t0:.1f}s: {exc}", flush=True)
        return ("__error__", str(exc))


# ─── Composite scoring and BestTracker ────────────────────────────────────────

def _selection_score(metrics: dict, sparsity: float, select_metric: str,
                     sparsity_weight: float, metric_weights: dict = None) -> float:
    """score = -(1-w)*loss(select_metric) + w*sparsity/100  (higher is better).

    loss() (metrics.selection_loss) is the "lower is better" form of the chosen
    metric.  At w=0 the score is just -loss; at w=1 it is sparsity/100.  With
    select_metric="rmse" this reproduces the previous -(1-w)*rmse formula.
    """
    loss = selection_loss(select_metric, metrics, metric_weights)
    return -(1.0 - sparsity_weight) * loss + sparsity_weight * sparsity / 100.0


def _report(tracker: BestTracker, select_metric: str = "rmse",
            sparsity_weight: float = 0.0):
    """Print the best regression config per (dataset, reg_type)."""
    print("\n" + "=" * 70)
    print("TSR CENTRALIZED ESN — HYPERPARAMETER SEARCH SUMMARY")
    print(f"  select_metric = {select_metric}")
    if sparsity_weight > 0:
        print(f"  sparsity_weight = {sparsity_weight:.2f}")
    print("=" * 70)
    for (dataset, reg_type), score, rec in tracker.items():
        m = rec["metrics"]
        print(f"\n  {dataset}  |  {reg_type}")
        print(f"    Score     : {score:.6f}   sparsity={rec['sparsity']:.1f}%")
        print(f"    rmse={m['rmse']:.6f}  nrmse={m['nrmse']:.6f}  "
              f"mase={m['mase']:.6f}  da={m['da']:.4f}  var_ratio={m['var_ratio']:.4f}")
        print(f"    vpt={m.get('vpt', float('nan')):.1f}  "
              f"mse{VPT_TIEBREAK_H}={m.get('mse_short', float('nan')):.6f}")
        for k, v in rec["cfg"].items():
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")


# ─── CSV output ───────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "step", "select_metric",
    "sr", "lr", "input_scaling", "reg_param",
    "score", "rmse", "nrmse", "mase", "da", "var_ratio",
    "vpt", "mse_short", "sparsity",
]


# ─── JSON export of best configs ─────────────────────────────────────────────

def _save_best_json(tracker: BestTracker, reg_types: list,
                    units: int, warmup: int, out_dir: str,
                    select_metric: str = "rmse",
                    noise_rc: float = _DEFAULT_NOISE_RC):
    """Save best regression configs as tsr_centralized_best_<reg_type>.json."""
    os.makedirs(out_dir, exist_ok=True)
    for reg_type in reg_types:
        entries = []
        for (ds, rt), score, rec in tracker.items():
            if rt != reg_type:
                continue
            cfg = rec["cfg"]
            m   = rec["metrics"]
            entry = {
                "dataset":              ds,
                "reg_type":             reg_type,
                "units":                units,
                "warmup":               warmup,
                "sr":                   round(cfg["sr"],            8),
                "lr":                   round(cfg["lr"],            8),
                "input_scaling":        round(cfg["input_scaling"], 8),
                "reg_param":            round(cfg["reg_param"],     8),
                "noise_rc":             noise_rc,
                "_search_select_metric": select_metric,
                "_search_rmse":         round(m["rmse"],      8),
                "_search_nrmse":        round(m["nrmse"],     8),
                "_search_mase":         round(m["mase"],      8),
                "_search_da":           round(m["da"],        6),
                "_search_var_ratio":    round(m["var_ratio"], 6),
                "_search_vpt":          round(m.get("vpt", float("nan")),       4),
                "_search_mse_short":    round(m.get("mse_short", float("nan")), 8),
                "_search_sparsity":     round(rec["sparsity"], 4),
                "_search_score":        round(score, 8),
            }
            entries.append(entry)

        if not entries:
            continue

        out_path = os.path.join(out_dir, f"tsr_centralized_best_{reg_type}.json")

        # Merge with existing file: keep the higher-scoring config per dataset,
        # so re-running with a new seed only overwrites a dataset's config when
        # it actually beats the one already on disk (e.g. a previous seed).
        def _score_of(e):
            return e.get("_search_score", float("-inf"))

        merged: dict = {}
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    for old in json.load(f):
                        merged[old["dataset"]] = old
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"  [WARN] could not read existing {out_path}: {exc} "
                      f"— overwriting")

        for entry in entries:
            ds  = entry["dataset"]
            old = merged.get(ds)
            if old is None or _score_of(entry) > _score_of(old):
                if old is not None:
                    print(f"  [{reg_type}/{ds}] new best {_score_of(entry):.6f} > "
                          f"{_score_of(old):.6f} — replaced")
                merged[ds] = entry
            else:
                print(f"  [{reg_type}/{ds}] kept existing {_score_of(old):.6f} ≥ "
                      f"{_score_of(entry):.6f}")

        out_entries = [merged[ds] for ds in sorted(merged)]
        with open(out_path, "w") as f:
            json.dump(out_entries, f, indent=4)
        print(f"Best regression configs ({reg_type}) saved → {out_path}")


# ─── Random search ────────────────────────────────────────────────────────────

def random_search(
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    reg_type: str,
    len_valid: int,
    k: int,
    steps: int,
    units: int,
    warmup: int,
    seed: int,
    sparsity_weight: float,
    csv_path: str,
    tracker: BestTracker,
    select_metric: str = "rmse",
    metric_weights: dict = None,
    params_range: dict = None,
    standardize: bool = False,
    noise_rc: float = _DEFAULT_NOISE_RC,
    thres: float = config.sl1_defaults.THRES,
    alpha_init: float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    use_gpu: bool = config.USE_GPU,
    n_jobs: int = 1,
) -> tuple:
    """Random search for (sr, lr, input_scaling, reg_param); returns (best_params, best_score).

    Samples *steps* unique parameter combos from the given ranges, evaluates each
    on *k* windowed validation splits, and records results incrementally to CSV.
    With n_jobs>1 the combo evaluations are fanned out across processes; combos
    are still sampled up front in the same order, so results match the serial run.
    """
    if params_range is None:
        params_range = {}

    rng_py  = random.Random(seed)
    rng_np  = np.random.RandomState(seed)

    # Continuous samplers, matching tsc_centralized_search._sample_config:
    # sr ~ uniform, the scale-like params (lr, input_scaling, reg_param)
    # ~ log-uniform.  A params_range entry pins that parameter to a fixed value.
    sr_fix = params_range.get("sr")
    lr_fix = params_range.get("lr")
    is_fix = params_range.get("input_scaling")
    rp_fix = params_range.get("reg_param")

    # reg_type="none" is plain OLS — reg_param has no effect, so pin it to a
    # single placeholder.  Otherwise the (sr, lr, is, reg_param) dedup would
    # treat reg_param-only variants as distinct combos and burn the step budget
    # re-running identical models.
    if reg_type == "none":
        rp_fix = 0.0

    sr_lo, sr_hi = min(_SR_SEARCH_RANGE), max(_SR_SEARCH_RANGE)
    sample_sr = (lambda: sr_fix) if sr_fix is not None else \
                (lambda: rng_py.uniform(sr_lo, sr_hi))
    sample_lr = (lambda: lr_fix) if lr_fix is not None else \
                (lambda: log_uniform(0.005, 1.0, rng_py))
    sample_is = (lambda: is_fix) if is_fix is not None else \
                (lambda: log_uniform(0.01, 100.0, rng_py))
    sample_rp = (lambda: rp_fix) if rp_fix is not None else \
                (lambda: log_uniform(1e-4, 1e2, rng_py))

    max_start    = len(X) - len_valid - 1
    valid_starts = rng_np.choice(max_start, min(k, max_start), replace=False).tolist()

    sampled, seen = [], set()
    while len(sampled) < steps:
        combo = (sample_sr(), sample_lr(), sample_is(), sample_rp())
        if combo not in seen:
            seen.add(combo)
            sampled.append(combo)

    best_score  = -np.inf
    best_params = {}

    # Share X/y with pool workers via fork (copy-on-write) instead of pickling
    # the long series into every task.
    _WORKER_DATA.clear()
    _WORKER_DATA.update({"X": X, "y": y})

    # Evaluate every combo (parallel when n_jobs>1); results come back in the
    # sampled order, so CSV rows and best-tracking stay identical to serial.
    combo_args = [
        (sr, lr, input_scaling, reg_param, reg_type, valid_starts,
         len_valid, units, warmup, seed, standardize, noise_rc,
         thres, alpha_init, alpha_multiplier, use_gpu)
        for (sr, lr, input_scaling, reg_param) in sampled
    ]
    results = parallel_map(_combo_worker, combo_args, n_jobs,
                           progress=f"  {dataset}/{reg_type}")

    for j, ((sr, lr, input_scaling, reg_param), res) in enumerate(zip(sampled, results)):
        if isinstance(res, tuple) and res and res[0] == "__error__":
            print(f"  step {j+1}/{steps}  sr={sr:.2f}  lr={lr:.3f}  "
                  f"is={input_scaling:.1e}  rp={reg_param:.1e}  [ERROR: {res[1]}]")
            continue
        metric_means, sparsity_mean = res
        score = _selection_score(metric_means, sparsity_mean,
                                 select_metric, sparsity_weight, metric_weights)

        cfg = {"sr": sr, "lr": lr, "input_scaling": input_scaling, "reg_param": reg_param}
        is_best = tracker.update((dataset, reg_type), score,
                                 {"metrics": metric_means, "sparsity": sparsity_mean,
                                  "cfg": deepcopy(cfg)})

        # Only announce a step when it sets a new best score — the full per-step
        # record still goes to the CSV below, so nothing is lost.
        if is_best:
            print(f"  step {j+1}/{steps}  [NEW BEST]  sr={sr:.2f}  lr={lr:.3f}  "
                  f"is={input_scaling:.1e}  rp={reg_param:.1e}")
            print(f"    →  rmse={metric_means['rmse']:.6f}  nrmse={metric_means['nrmse']:.6f}  "
                  f"mase={metric_means['mase']:.6f}  da={metric_means['da']:.4f}  "
                  f"var_ratio={metric_means['var_ratio']:.4f}  "
                  f"vpt={metric_means['vpt']:.1f}  mse{VPT_TIEBREAK_H}={metric_means['mse_short']:.4f}  "
                  f"sparsity={sparsity_mean:.1f}%  score={score:.6f}")

        append_csv_row(csv_path, _CSV_FIELDS, {
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":    dataset,
            "reg_type":   reg_type,
            "step":       j,
            "select_metric": select_metric,
            "sr":         f"{sr:.6g}",
            "lr":         f"{lr:.6g}",
            "input_scaling": f"{input_scaling:.6g}",
            "reg_param":  f"{reg_param:.6g}",
            "score":      f"{score:.6f}",
            "rmse":       f"{metric_means['rmse']:.6f}",
            "nrmse":      f"{metric_means['nrmse']:.6f}",
            "mase":       f"{metric_means['mase']:.6f}",
            "da":         f"{metric_means['da']:.6f}",
            "var_ratio":  f"{metric_means['var_ratio']:.6f}",
            "vpt":        f"{metric_means['vpt']:.4f}",
            "mse_short":  f"{metric_means['mse_short']:.6f}",
            "sparsity":   f"{sparsity_mean:.4f}",
        })

        if score > best_score:
            best_score  = score
            best_params = {"sr": sr, "lr": lr, "input_scaling": input_scaling,
                           "reg_param": reg_param, "units": units, "warmup": warmup}
            print(f"    ★ new best: {best_params}  score={best_score:.6f}")

    return best_params, best_score


# ─── Main search ──────────────────────────────────────────────────────────────

def main(
    datasets:        list   = None,
    reg_types_list:  list   = None,
    steps:           int    = 100,
    len_valid:       int    = 2000,
    k:               int    = 3,
    units:           int    = 500,
    warmup:          int    = 100,
    sparsity_weight: float  = 0.0,
    select_metric:   str    = "rmse",
    metric_weights:  dict   = None,
    noise_rc:        float  = _DEFAULT_NOISE_RC,
    csv_path:        str    = "./result/tsr_centralized_search.csv",
    json_dir:        str    = "./result",
    seed:            int    = 1234,
    thres:            float = config.sl1_defaults.THRES,
    alpha_init:       float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    use_gpu:          bool  = config.USE_GPU,
    n_jobs:           int   = 1,
):
    if datasets is None:
        datasets = _TSR_DATASETS
    if reg_types_list is None:
        reg_types_list = _REG_TYPES

    # GPU + multiprocessing oversubscribe a single device; serialise in that case.
    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"(parallel workers contend for one device).")
        n_jobs = 1

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    init_csv(csv_path, _CSV_FIELDS)
    tracker = BestTracker()

    for dataset_name in datasets:
        print(f"\n{'='*65}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*65}")

        X, y, _, _ = read_data(dataset_name)
        standardize = dataset_name in STANDARDIZE_DATASETS
        if standardize:
            print("  [preprocessing] per-channel standardization enabled "
                  "(stats fit per validation window)")

        for reg_type in reg_types_list:
            print(f"\n  reg_type = {reg_type}  ({steps} random trials)")

            t0 = time.time()
            best_params, best_score = random_search(
                X=X, y=y,
                dataset=dataset_name,
                reg_type=reg_type,
                len_valid=len_valid,
                k=k,
                steps=steps,
                units=units,
                warmup=warmup,
                seed=seed,
                sparsity_weight=sparsity_weight,
                csv_path=csv_path,
                tracker=tracker,
                select_metric=select_metric,
                metric_weights=metric_weights,
                standardize=standardize,
                noise_rc=noise_rc,
                thres=thres,
                alpha_init=alpha_init,
                alpha_multiplier=alpha_multiplier,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
            )
            elapsed = time.time() - t0
            print(f"\n  Best ({reg_type}): {best_params}  "
                  f"score={best_score:.6f}  ({elapsed:.1f}s)")

        # ── Ridge + IMP baseline at SL1's sparsity (regression CE+IMP analogue) ──
        # Dense ridge fit → prune-refit to the sparsity SL1 reached, over the same
        # windows, so the "does the penalty beat a prune baseline?" question is
        # answered for regression too.  Recorded as reg_type "ridge_imp".
        sl1_best = tracker.get((dataset_name, "sl1"))
        if sl1_best is not None:
            _s, rec = sl1_best
            target_sp = rec["sparsity"]
            print(f"\n  reg_type = ridge_imp  (dense ridge + IMP → {target_sp:.1f}% sparse)")
            try:
                m, sp = _imp_baseline(
                    X, y, rec["cfg"], target_sp, len_valid, k, units, warmup,
                    standardize, noise_rc, seed)
                print(f"    →  rmse={m['rmse']:.6f}  nrmse={m['nrmse']:.6f}  "
                      f"vpt={m['vpt']:.1f}  sparsity={sp:.1f}%")
                append_csv_row(csv_path, _CSV_FIELDS, {
                    "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset":    dataset_name,
                    "reg_type":   "ridge_imp",
                    "step":       "baseline",
                    "select_metric": select_metric,
                    "rmse":       f"{m['rmse']:.6f}",
                    "nrmse":      f"{m['nrmse']:.6f}",
                    "mase":       f"{m['mase']:.6f}",
                    "da":         f"{m['da']:.6f}",
                    "var_ratio":  f"{m['var_ratio']:.6f}",
                    "vpt":        f"{m['vpt']:.4f}",
                    "mse_short":  f"{m['mse_short']:.6f}",
                    "sparsity":   f"{sp:.4f}",
                })
            except Exception as exc:                             # noqa: BLE001
                print(f"    [ERROR: {exc}]")

    _report(tracker, select_metric, sparsity_weight)
    _save_best_json(tracker, reg_types_list, units, warmup, json_dir,
                    select_metric, noise_rc)
    print(f"\nAll results saved → {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Random hyperparameter search for centralized ESN regression"
    )
    p.add_argument("--datasets",  nargs="+", default=_TSR_DATASETS,
                   choices=_TSR_DATASETS)
    p.add_argument("--reg_types", nargs="+", default=_REG_TYPES,
                   choices=_REG_TYPES)
    p.add_argument("--steps",     type=int,   default=100)
    p.add_argument("--len_valid", type=int,   default=2000,
                   help="Validation window length (default: 2000). Keep the "
                        "post-warmup train portion (≈0.8·len_valid − warmup) well "
                        "above 'units', else the sl1 least-squares fit is "
                        "underdetermined and the L1General solver grinds (slow).")
    p.add_argument("--k",         type=int,   default=3,
                   help="Validation windows to average over (default: 3)")
    p.add_argument("--units",     type=int,   default=500,
                   help="Reservoir size (default: 500, matching tsr_centralized_eval)")
    p.add_argument("--warmup",    type=int,   default=100)
    p.add_argument("--noise_rc",  type=float, default=_DEFAULT_NOISE_RC,
                   help="Gaussian noise added to the readout's training states "
                        "(stabilises auto-regressive generation; 0 disables)")
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight of sparsity in the score (0=pure accuracy metric, 1=pure sparsity)")
    p.add_argument("--select_metric", default="vpt", choices=SELECT_METRICS,
                   help="Accuracy metric driving selection (default: rmse). "
                        "'vpt' = valid prediction time and 'mse_short' = "
                        f"first-{VPT_TIEBREAK_H}-step MSE are the chaos-aware "
                        "metrics (recommended for lorenz); 'composite' = "
                        "scale-free weighted blend of nrmse/mase/da/var_ratio")
    # Weights for --select_metric composite (renormalised; need not sum to 1).
    p.add_argument("--w_nrmse",     type=float, default=0.4, help="composite weight: nrmse")
    p.add_argument("--w_mase",      type=float, default=0.3, help="composite weight: mase")
    p.add_argument("--w_da",        type=float, default=0.2, help="composite weight: da")
    p.add_argument("--w_var_ratio", type=float, default=0.1, help="composite weight: var_ratio")
    p.add_argument("--csv_path",  default="./result/tsr_centralized_search.csv")
    p.add_argument("--json_dir",  default="./result",
                   help="Directory for tsr_centralized_best_<reg_type>.json files")
    p.add_argument("--seed",      type=int,   default=1234)
    # SL1 sparsification schedule — drives the offline Newton solver
    # (solve_newton_mse_l1) as well as the isFL SGD path.  Pinned here (not
    # searched); the random search sweeps reg_param (λ).
    p.add_argument("--thres",            type=float, default=config.sl1_defaults.THRES,
                   help="SL1 soft-threshold magnitude (per-step pruning)")
    p.add_argument("--alpha_init",       type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help="Initial SmoothL1 alpha (α continuation start)")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help="Alpha growth factor (α continuation rate)")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the offline SL1 (L1General) solver; "
                        "auto-falls back to CPU if CuPy/CUDA is unavailable")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel worker processes for search steps (1=serial "
                        "[default], -1=all cores). Forced to 1 under --gpu "
                        "(single-device contention).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        datasets        = args.datasets,
        reg_types_list  = args.reg_types,
        steps           = args.steps,
        len_valid       = args.len_valid,
        k               = args.k,
        units           = args.units,
        warmup          = args.warmup,
        sparsity_weight = args.sparsity_weight,
        select_metric   = args.select_metric,
        metric_weights  = {"nrmse": args.w_nrmse, "mase": args.w_mase,
                           "da": args.w_da, "var_ratio": args.w_var_ratio},
        noise_rc        = args.noise_rc,
        csv_path        = args.csv_path,
        json_dir        = args.json_dir,
        seed            = args.seed,
        thres            = args.thres,
        alpha_init       = args.alpha_init,
        alpha_multiplier = args.alpha_multiplier,
        use_gpu          = args.use_gpu,
        n_jobs           = args.n_jobs,
    )
