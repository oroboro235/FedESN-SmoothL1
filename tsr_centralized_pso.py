# tsr_centralized_pso.py — Particle Swarm Optimization for centralized ESN regression.
#
# Drop-in alternative to tsr_centralized_search.py (random search): same model,
# same windowed auto-regressive validation, same scoring (_selection_score) and
# the same best-config JSON files — only the *search strategy* differs.
#
# Why PSO fits this problem
# -------------------------
# The search space is 4 continuous hyperparameters (sr, lr, input_scaling,
# reg_param), the objective is an expensive, gradient-free black box, and the
# dimensionality is low.  That is exactly where global-best PSO shines: it needs
# no gradients, explores continuously, and steers new trials toward the best
# regions found so far (pbest / gbest), so for a comparable evaluation budget it
# typically beats uniform random search.
#
# Two domain-specific details handled below:
#   * input_scaling and reg_param are searched on a log10 scale (their natural
#     spacing), so particles move in log-space and are de-transformed (10**x)
#     only when an ESN is actually trained.
#   * Each objective evaluation trains an ESN + L1General solver over k
#     validation windows, so it is costly.  A result cache (keyed on rounded
#     params) skips re-evaluating points particles revisit.
#
# The per-trial evaluation, scoring, CSV columns and best-config JSON are reused
# verbatim from tsr_centralized_search, so PSO results compete on the same
# footing and are merged into the same tsr_centralized_best_<reg_type>.json
# (higher _search_score wins).
#
# Results:
#   ./result/tsr_centralized_pso.csv              — one row per *unique* evaluation
#   ./result/tsr_centralized_best_<reg_type>.json — best configs per (dataset, reg_type)
#
# Usage:
#   python tsr_centralized_pso.py
#   python tsr_centralized_pso.py --datasets lorenz --reg_types sl1 --select_metric vpt
#   python tsr_centralized_pso.py --n_particles 16 --n_iters 20
#   python tsr_centralized_pso.py --sparsity_weight 0.3
#   python tsr_centralized_pso.py --n_jobs -1        # evaluate the swarm in parallel
#
# Parallelism: particles within one PSO iteration are independent, so they are
# evaluated concurrently with --n_jobs worker processes (joblib/loky, inner BLAS
# threads pinned to 1 to avoid oversubscription).  Results are folded back in
# particle order, so the run is bit-for-bit deterministic regardless of n_jobs.

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np

from joblib import Parallel, delayed, parallel_config

import config
from data_loader import read_data
from utils import init_csv, append_csv_row, BestTracker
from metrics import SELECT_METRICS

# Reuse the centralized search building blocks so PSO and random search stay in
# lock-step (identical model, metrics, scoring and JSON export).
from tsr_centralized_search import (
    _eval_window,
    _selection_score,
    _save_best_json,
    _report,
    _SR_SEARCH_RANGE,
    _DEFAULT_NOISE_RC,
    _TSR_DATASETS,
    _REG_TYPES,
)
from tsr_centralized_eval import STANDARDIZE_DATASETS, VPT_TIEBREAK_H


# ─── Search-space definition ──────────────────────────────────────────────────
# Each dimension: (name, low, high, log10?).  Bounds mirror the random-search
# grids in tsr_centralized_search (sr edge-of-chaos band; is/rp log-spaced).
_DIMS = [
    ("sr",            min(_SR_SEARCH_RANGE), max(_SR_SEARCH_RANGE), False),
    ("lr",            0.01,                  0.99,                  False),
    ("input_scaling", 1e-3,                  1e1,                   True),
    ("reg_param",     1e-4,                  1e2,                   True),
]


def _to_internal(params: dict) -> np.ndarray:
    """Map a {name: value} dict to the PSO-internal coordinate vector.

    Log-scaled dimensions are stored as log10(value); linear dimensions as-is.
    """
    x = np.empty(len(_DIMS))
    for i, (name, _lo, _hi, is_log) in enumerate(_DIMS):
        v = params[name]
        x[i] = np.log10(v) if is_log else v
    return x


def _to_params(x: np.ndarray) -> dict:
    """Inverse of _to_internal: PSO coordinate vector → {name: value} dict."""
    params = {}
    for i, (name, _lo, _hi, is_log) in enumerate(_DIMS):
        params[name] = float(10 ** x[i]) if is_log else float(x[i])
    return params


def _internal_bounds() -> tuple:
    """Return (low, high) arrays of the internal (possibly log10) coordinates."""
    lo = np.array([np.log10(d[1]) if d[3] else d[1] for d in _DIMS])
    hi = np.array([np.log10(d[2]) if d[3] else d[2] for d in _DIMS])
    return lo, hi


# ─── Objective: average score over k validation windows ───────────────────────

def _evaluate_candidate(
    params: dict,
    X, y, reg_type, valid_starts, len_valid,
    units, warmup, seed, sparsity_weight,
    select_metric, metric_weights, standardize, noise_rc,
    thres, alpha_init, alpha_multiplier, use_gpu,
) -> tuple:
    """Train+score one hyperparameter combo; return (score, metric_means, sparsity).

    Mirrors the inner loop of tsr_centralized_search.random_search so PSO trials
    are scored identically to random-search trials.
    """
    metric_sums = {m: 0.0 for m in
                   ("rmse", "nrmse", "mase", "da", "var_ratio",
                    "vpt", "vpt_norm", "mse_short")}
    sparsity_mean = 0.0
    valid_count = 0
    for idx in valid_starts:
        X_win = X[idx: idx + len_valid]
        y_win = y[idx: idx + len_valid]
        split = int(len_valid * 0.8)
        metrics_w, sp_w = _eval_window(
            X_win[:split], y_win[:split],
            X_win[split:], y_win[split:],
            units, params["sr"], params["lr"],
            params["input_scaling"], params["reg_param"],
            reg_type, warmup, seed=seed,
            standardize=standardize, noise_rc=noise_rc,
            thres=thres, alpha_init=alpha_init,
            alpha_multiplier=alpha_multiplier, use_gpu=use_gpu,
        )
        for m in metric_sums:
            metric_sums[m] += metrics_w[m]
        sparsity_mean += sp_w
        valid_count += 1

    metric_means = {m: s / valid_count for m, s in metric_sums.items()}
    sparsity_mean /= valid_count
    score = _selection_score(metric_means, sparsity_mean,
                             select_metric, sparsity_weight, metric_weights)
    return score, metric_means, sparsity_mean


# ─── CSV output (one row per unique evaluation) ───────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "iter", "particle", "select_metric",
    "sr", "lr", "input_scaling", "reg_param",
    "score", "rmse", "nrmse", "mase", "da", "var_ratio",
    "vpt", "mse_short", "sparsity",
]


# ─── PSO core ─────────────────────────────────────────────────────────────────

def _print_trial(it, n_iters, p, params, metric_means, sparsity, score, tag=""):
    """Print one trial line — same metrics as tsr_centralized_search.random_search."""
    print(f"  iter {it+1}/{n_iters}  p{p:02d}  "
          f"sr={params['sr']:.2f} lr={params['lr']:.3f} "
          f"is={params['input_scaling']:.1e} rp={params['reg_param']:.1e}{tag}")
    print(f"     →  rmse={metric_means['rmse']:.6f}  nrmse={metric_means['nrmse']:.6f}  "
          f"mase={metric_means['mase']:.6f}  da={metric_means['da']:.4f}  "
          f"var_ratio={metric_means['var_ratio']:.4f}  "
          f"vpt={metric_means['vpt']:.1f}  mse{VPT_TIEBREAK_H}={metric_means['mse_short']:.4f}  "
          f"sparsity={sparsity:.1f}%  score={score:.6f}")


def pso_search(
    X, y, dataset, reg_type,
    len_valid, k, units, warmup, seed,
    sparsity_weight, csv_path, tracker,
    n_particles=12, n_iters=15,
    w_inertia_start=0.9, w_inertia_end=0.4,
    c1=1.49445, c2=1.49445, v_frac=0.2,
    select_metric="rmse", metric_weights=None,
    standardize=False, noise_rc=_DEFAULT_NOISE_RC,
    n_jobs=1,
    thres=config.sl1_defaults.THRES,
    alpha_init=config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier=config.sl1_defaults.ALPHA_MULTIPLIER,
    use_gpu=config.USE_GPU,
) -> tuple:
    """Global-best PSO over (sr, lr, log10 input_scaling, log10 reg_param).

    Returns (best_params, best_score).  Evaluations are cached on rounded
    internal coordinates so revisited points are not re-trained.

    Parallelism: within each iteration the particles are mutually independent,
    so all cache-missing particles are evaluated concurrently with *n_jobs*
    worker processes (joblib/loky).  Results are then folded back into the swarm
    in particle order, so the swarm trajectory — and therefore the outcome — is
    identical regardless of n_jobs (deterministic given the seed).
    """
    rng = np.random.RandomState(seed)
    lo, hi = _internal_bounds()

    # reg_type="none" is plain OLS — reg_param has no effect.  Collapse its
    # dimension to a single point (zero span ⇒ zero velocity) so the swarm
    # doesn't waste a dimension exploring an inert parameter, and report it as
    # 0.0 (matching random search) rather than the frozen log-bound.
    rp_idx = next(i for i, d in enumerate(_DIMS) if d[0] == "reg_param")
    if reg_type == "none":
        hi[rp_idx] = lo[rp_idx]

    def _params_of(x_internal):
        params = _to_params(x_internal)
        if reg_type == "none":
            params["reg_param"] = 0.0
        return params

    span = hi - lo
    v_max = v_frac * span

    max_start = len(X) - len_valid - 1
    valid_starts = rng.choice(max_start, min(k, max_start), replace=False).tolist()

    # Cache: rounded internal coord tuple → (score, metric_means, sparsity).
    cache: dict = {}

    def _eval_args(x_internal):
        """Positional args tuple for _evaluate_candidate (picklable for workers)."""
        return (_params_of(x_internal), X, y, reg_type, valid_starts, len_valid,
                units, warmup, seed, sparsity_weight,
                select_metric, metric_weights, standardize, noise_rc,
                thres, alpha_init, alpha_multiplier, use_gpu)

    # Initialise swarm: positions uniform in bounds, velocities small.
    pos = lo + rng.random_sample((n_particles, len(_DIMS))) * span
    vel = (rng.random_sample((n_particles, len(_DIMS))) - 0.5) * 2 * v_max

    pbest_pos   = pos.copy()
    pbest_score = np.full(n_particles, -np.inf)
    gbest_pos   = None
    gbest_score = -np.inf

    n_eval = 0
    for it in range(n_iters):
        w = w_inertia_start + (w_inertia_end - w_inertia_start) * (it / max(1, n_iters - 1))

        # ── 1. Identify cache misses for this iteration (deduplicated by key) ──
        keys = [tuple(np.round(pos[p], 4)) for p in range(n_particles)]
        todo = {}  # key → representative x_internal
        for p in range(n_particles):
            if keys[p] not in cache and keys[p] not in todo:
                todo[keys[p]] = pos[p].copy()

        # ── 2. Evaluate the misses (parallel across particles, or serial) ──
        if todo:
            todo_keys = list(todo.keys())
            todo_args = [_eval_args(todo[k_]) for k_ in todo_keys]
            if n_jobs == 1 or len(todo_keys) == 1:
                results = [_evaluate_candidate(*a) for a in todo_args]
            else:
                # Pin inner BLAS threads to 1 so the worker processes don't
                # oversubscribe the CPU against each other.
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(_evaluate_candidate)(*a) for a in todo_args)
            for k_, res in zip(todo_keys, results):
                cache[k_] = res
                n_eval += 1

        fresh = set(todo)          # keys first evaluated this iteration
        logged: set = set()        # CSV-logged keys (dedupe shared positions)

        # ── 3. Fold results back into the swarm, in particle order ──
        for p in range(n_particles):
            score, metric_means, sparsity = cache[keys[p]]
            params = _params_of(pos[p])
            is_fresh = keys[p] in fresh

            if is_fresh and keys[p] not in logged:
                logged.add(keys[p])
                append_csv_row(csv_path, _CSV_FIELDS, {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": dataset, "reg_type": reg_type,
                    "iter": it, "particle": p, "select_metric": select_metric,
                    "sr": f"{params['sr']:.6g}", "lr": f"{params['lr']:.6g}",
                    "input_scaling": f"{params['input_scaling']:.6g}",
                    "reg_param": f"{params['reg_param']:.6g}",
                    "score": f"{score:.6f}",
                    "rmse": f"{metric_means['rmse']:.6f}",
                    "nrmse": f"{metric_means['nrmse']:.6f}",
                    "mase": f"{metric_means['mase']:.6f}",
                    "da": f"{metric_means['da']:.6f}",
                    "var_ratio": f"{metric_means['var_ratio']:.6f}",
                    "vpt": f"{metric_means['vpt']:.4f}",
                    "mse_short": f"{metric_means['mse_short']:.6f}",
                    "sparsity": f"{sparsity:.4f}",
                })

            _print_trial(it, n_iters, p, params, metric_means, sparsity, score,
                         tag="" if is_fresh else "  (cached)")

            # Update personal best.
            if score > pbest_score[p]:
                pbest_score[p] = score
                pbest_pos[p]   = pos[p].copy()
            # Update global best (+ tracker / JSON record).
            if score > gbest_score:
                gbest_score = score
                gbest_pos   = pos[p].copy()
                tracker.update((dataset, reg_type), score,
                               {"metrics": metric_means, "sparsity": sparsity,
                                "cfg": deepcopy(params)})
                print(f"    ★ new gbest: {params}  score={gbest_score:.6f}")

        # ── 4. Velocity / position update (after the whole swarm is scored) ──
        r1 = rng.random_sample((n_particles, len(_DIMS)))
        r2 = rng.random_sample((n_particles, len(_DIMS)))
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -v_max, v_max)
        pos = pos + vel
        # Clamp to bounds; reflect (and damp) the velocity that hit a wall.
        below = pos < lo
        above = pos > hi
        pos = np.clip(pos, lo, hi)
        vel[below | above] *= -0.5

    best_params = _params_of(gbest_pos)
    print(f"\n  PSO done: {n_eval} unique evals "
          f"(+{n_particles * n_iters - n_eval} cache hits)")
    return best_params, gbest_score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    datasets=None, reg_types_list=None,
    n_particles=12, n_iters=15,
    len_valid=2000, k=3, units=500, warmup=100,
    sparsity_weight=0.0, select_metric="rmse", metric_weights=None,
    noise_rc=_DEFAULT_NOISE_RC, n_jobs=1,
    c1=1.49445, c2=1.49445, w_inertia_start=0.9, w_inertia_end=0.4, v_frac=0.2,
    csv_path="./result/tsr_centralized_pso.csv",
    json_dir="./result", seed=1234,
    thres=config.sl1_defaults.THRES,
    alpha_init=config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier=config.sl1_defaults.ALPHA_MULTIPLIER,
    use_gpu=config.USE_GPU,
):
    if datasets is None:
        datasets = _TSR_DATASETS
    if reg_types_list is None:
        reg_types_list = _REG_TYPES

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
            print(f"\n  reg_type = {reg_type}  "
                  f"(PSO: {n_particles} particles × {n_iters} iters)")

            t0 = time.time()
            best_params, best_score = pso_search(
                X=X, y=y, dataset=dataset_name, reg_type=reg_type,
                len_valid=len_valid, k=k, units=units, warmup=warmup, seed=seed,
                sparsity_weight=sparsity_weight, csv_path=csv_path, tracker=tracker,
                n_particles=n_particles, n_iters=n_iters,
                w_inertia_start=w_inertia_start, w_inertia_end=w_inertia_end,
                c1=c1, c2=c2, v_frac=v_frac,
                select_metric=select_metric, metric_weights=metric_weights,
                standardize=standardize, noise_rc=noise_rc, n_jobs=n_jobs,
                thres=thres, alpha_init=alpha_init,
                alpha_multiplier=alpha_multiplier, use_gpu=use_gpu,
            )
            elapsed = time.time() - t0
            print(f"\n  Best ({reg_type}): {best_params}  "
                  f"score={best_score:.6f}  ({elapsed:.1f}s)")

    _report(tracker, select_metric, sparsity_weight)
    _save_best_json(tracker, reg_types_list, units, warmup, json_dir,
                    select_metric, noise_rc)
    print(f"\nAll PSO results saved → {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Particle Swarm Optimization for centralized ESN regression")
    p.add_argument("--datasets",  nargs="+", default=_TSR_DATASETS, choices=_TSR_DATASETS)
    p.add_argument("--reg_types", nargs="+", default=_REG_TYPES,    choices=_REG_TYPES)
    p.add_argument("--n_particles", type=int, default=12, help="Swarm size (default: 12)")
    p.add_argument("--n_iters",     type=int, default=15, help="PSO iterations (default: 15)")
    p.add_argument("--c1", type=float, default=1.49445, help="Cognitive coefficient")
    p.add_argument("--c2", type=float, default=1.49445, help="Social coefficient")
    p.add_argument("--w_start", type=float, default=0.9, help="Initial inertia weight")
    p.add_argument("--w_end",   type=float, default=0.4, help="Final inertia weight")
    p.add_argument("--v_frac",  type=float, default=0.2,
                   help="Velocity clamp as a fraction of each dim's range")
    p.add_argument("--n_jobs",  type=int, default=1,
                   help="Parallel worker processes for evaluating the swarm "
                        "(1=serial; -1=all cores). Particles within an iteration "
                        "are evaluated concurrently; the result is identical to "
                        "the serial run (deterministic given --seed).")
    p.add_argument("--len_valid", type=int, default=2000)
    p.add_argument("--k",         type=int, default=3,
                   help="Validation windows to average over (default: 3)")
    p.add_argument("--units",     type=int, default=500)
    p.add_argument("--warmup",    type=int, default=100)
    p.add_argument("--noise_rc",  type=float, default=_DEFAULT_NOISE_RC)
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight of sparsity in the score (0=pure accuracy, 1=pure sparsity)")
    p.add_argument("--select_metric", default="rmse", choices=SELECT_METRICS,
                   help="Accuracy metric driving selection (default: rmse; "
                        "'vpt'/'mse_short' recommended for lorenz)")
    p.add_argument("--w_nrmse",     type=float, default=0.4, help="composite weight: nrmse")
    p.add_argument("--w_mase",      type=float, default=0.3, help="composite weight: mase")
    p.add_argument("--w_da",        type=float, default=0.2, help="composite weight: da")
    p.add_argument("--w_var_ratio", type=float, default=0.1, help="composite weight: var_ratio")
    p.add_argument("--csv_path",  default="./result/tsr_centralized_pso.csv")
    p.add_argument("--json_dir",  default="./result")
    p.add_argument("--seed",      type=int, default=1234)
    p.add_argument("--thres",            type=float, default=config.sl1_defaults.THRES)
    p.add_argument("--alpha_init",       type=float, default=config.sl1_defaults.ALPHA_INIT)
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER)
    p.add_argument("--gpu",    dest="use_gpu", action="store_true",  default=config.USE_GPU)
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        datasets=args.datasets, reg_types_list=args.reg_types,
        n_particles=args.n_particles, n_iters=args.n_iters,
        len_valid=args.len_valid, k=args.k, units=args.units, warmup=args.warmup,
        sparsity_weight=args.sparsity_weight, select_metric=args.select_metric,
        metric_weights={"nrmse": args.w_nrmse, "mase": args.w_mase,
                        "da": args.w_da, "var_ratio": args.w_var_ratio},
        noise_rc=args.noise_rc, n_jobs=args.n_jobs,
        c1=args.c1, c2=args.c2,
        w_inertia_start=args.w_start, w_inertia_end=args.w_end, v_frac=args.v_frac,
        csv_path=args.csv_path, json_dir=args.json_dir, seed=args.seed,
        thres=args.thres, alpha_init=args.alpha_init,
        alpha_multiplier=args.alpha_multiplier, use_gpu=args.use_gpu,
    )
