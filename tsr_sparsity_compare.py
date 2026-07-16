# tsr_sparsity_compare.py — λ vs threshold compression/sparsity comparison for
# the TSR (regression) readout.  The regression counterpart of the TSC
# sl1_factor_benchmark.py --pareto.
#
# TSR now uses the SAME sparsification mechanism as TSC:
#   TSC centralized  = CE  + SmoothL1, α continuation + per-step soft-threshold
#                      (newton_solver.solve_newton_l1).
#   TSR centralized  = MSE + SmoothL1, α continuation + per-step soft-threshold
#                      (newton_solver.solve_newton_mse_l1) — this is exactly what
#                      Reg_Node.fit(reg_type='sl1', isFL=False) now runs.  Both λ
#                      and `thres` are live in-loop knobs: λ is the SmoothL1 shrink
#                      baked into the objective, `thres` is the per-step proximal
#                      soft-threshold applied after every Newton step.
#   (Historically TSR used the offline L1General solver on the true-L1 objective,
#    where `thres` was ignored and only λ + L1General's final hard cut mattered.
#    That is no longer the production path.)
#
# So this script sweeps the TWO knobs that govern TSR sparsity, via the actual
# production solver (solve_newton_mse_l1):
#   λ-path         — sweep reg_param at a fixed soft-threshold `thres`.
#   threshold-path — sweep the per-step soft-threshold `thres` at fixed λ.
# Compression capability = sparsity reached at a given error.
#
# Usage:
#   python tsr_sparsity_compare.py --synthetic                       # no deps
#   python tsr_sparsity_compare.py --dataset mg --units 500
#   python tsr_sparsity_compare.py --dataset lorenz --lam_grid 1e-4 1e-3 1e-2 1e-1 1

import argparse
import time

import numpy as np

import config


# ─── Production Newton fit (MSE + SmoothL1) with a controllable soft-threshold ─

def fit_newton_mse(X, y, reg_param, threshold,
                   alpha_init=config.sl1_defaults.ALPHA_INIT,
                   alpha_multiplier=config.sl1_defaults.ALPHA_MULTIPLIER,
                   alpha_max=1e6, max_iter=100):
    """Fit Wout on MSE + SmoothL1 via the production Newton solver.

    Calls newton_solver.solve_newton_mse_l1 — exactly what
    readout_node.Reg_Node.fit(reg_type='sl1', isFL=False) now runs — with
    `threshold` wired to the per-step soft-threshold `thres`, so the
    threshold-path sweep exercises the real production knob.  alpha_max=1e6
    matches Reg_Node's regression cap; alpha_init/alpha_multiplier default to the
    shared sl1_defaults schedule.

    Returns Wout of shape (X.shape[1], y.shape[1]).
    """
    from newton_solver import solve_newton_mse_l1

    return solve_newton_mse_l1(
        X, y, np.zeros((X.shape[1], y.shape[1])), reg_param,
        alpha_init=alpha_init, alpha_max=alpha_max,
        update1=alpha_multiplier, update2=alpha_multiplier,
        thres=threshold, max_iter=max_iter)


# ─── Real TSR window eval (reservoir + auto-regressive forecast) ──────────────

def eval_window(X_tr, y_tr, X_ts, y_ts, *, units, sr, lr, input_scaling,
                warmup, noise_rc, seed, reg_param, threshold, standardize):
    """Train on X_tr, auto-regressively forecast len(X_ts); return (metrics, sp).

    Mirrors tsr_centralized_search._eval_window; fits via fit_newton_mse (the
    production solve_newton_mse_l1) so the per-step soft-threshold is controllable.
    reservoirpy and the TSR eval helpers are imported lazily so the fit/sweep logic
    can be smoke-tested (--synthetic) without them.
    """
    from reservoirpy.nodes import Reservoir
    from tsr_centralized_eval import _autoregressive_generate, _fit_standardizer
    from metrics import compute_all, valid_prediction_time

    if standardize:
        mean, std = _fit_standardizer(X_tr)
        X_tr = (X_tr - mean) / std
        y_tr = (y_tr - mean) / std
        X_ts = (X_ts - mean) / std
        y_ts = (y_ts - mean) / std

    reservoir  = Reservoir(units=units, sr=sr, lr=lr,
                           input_scaling=input_scaling, seed=seed)
    all_states = reservoir.run(X_tr)
    X_states   = all_states[warmup:]
    y_reg      = y_tr[warmup:]
    if noise_rc > 0:
        X_states = X_states + noise_rc * np.random.RandomState(seed).standard_normal(X_states.shape)

    Wout  = fit_newton_mse(X_states, y_reg, reg_param, threshold)
    preds = _autoregressive_generate(len(X_ts), all_states[-1], reservoir, Wout)
    if standardize:
        preds = preds * std + mean
        y_ts  = y_ts  * std + mean
        y_tr  = y_tr  * std + mean

    metrics = compute_all(preds, y_ts, y_tr)
    metrics["vpt"] = valid_prediction_time(preds, y_ts)
    sparsity = float((Wout == 0).mean() * 100)
    return metrics, sparsity


# ─── Synthetic smoke test (no reservoirpy / datasets) ─────────────────────────

def synthetic_regression(n=300, p=200, k=15, seed=0):
    """Correlated design + sparse ground-truth weights, split train/test.

    Lets the Newton fit + sweep tables be validated without a reservoir: a
    well-chosen λ/threshold should recover a sparse w at low held-out NRMSE."""
    rng   = np.random.RandomState(seed)
    basis = rng.randn(p, p // 8)
    X     = rng.randn(n, p // 8) @ basis.T + 0.1 * rng.randn(n, p)
    w     = np.zeros((p, 1)); idx = rng.choice(p, k, replace=False)
    w[idx, 0] = rng.randn(k) * 2.0
    y     = X @ w + 0.1 * rng.randn(n, 1)
    ntr   = int(0.7 * n)
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:]


def _nrmse(preds, trues):
    denom = np.std(trues) + 1e-12
    return float(np.sqrt(np.mean((preds - trues) ** 2)) / denom)


def eval_synth(Xtr, ytr, Xte, yte, reg_param, threshold):
    """Plain fit + held-out NRMSE/sparsity (no reservoir, no auto-regression)."""
    Wout = fit_newton_mse(Xtr, ytr, reg_param, threshold)
    nrmse = _nrmse(Xte @ Wout, yte)
    sp    = float((Wout == 0).mean() * 100)
    return {"nrmse": nrmse}, sp


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_row(fit_eval, reg_param, threshold):
    t0 = time.perf_counter()
    m, sp = fit_eval(reg_param, threshold)
    wall = time.perf_counter() - t0
    err  = m["nrmse"]
    extra = (f"  rmse={m['rmse']:.4g}" if "rmse" in m else "") + \
            (f"  vpt={m['vpt']:.0f}" if "vpt" in m else "")
    print(f"  λ={reg_param:>9.2e}  thr={threshold:>8.1e} | nrmse={err:7.4f}  "
          f"sp={sp:5.1f}%  {wall:6.2f}s{extra}")
    return dict(reg_param=reg_param, threshold=threshold, nrmse=err, sp=sp)


def compare(fit_eval, lam_grid, thr_grid, lam_fixed, thr_fixed):
    print("\nλ-PATH  (thres fixed = %.0e) — SmoothL1 shrink in the objective:" % thr_fixed)
    lam_rows = [run_row(fit_eval, lam, thr_fixed) for lam in lam_grid]

    print("\nthreshold-PATH  (λ fixed = %g) — per-step soft-threshold thres:" % lam_fixed)
    thr_rows = [run_row(fit_eval, lam_fixed, thr) for thr in thr_grid]

    print("\nHow to read (compression capability):")
    print("  • Pick a target sparsity; find that row in each path; compare nrmse.")
    print("    Lower nrmse at equal sparsity ⇒ that knob compresses better.")
    print("  • Both knobs are now in-loop (α-continuation Newton): λ shrinks via")
    print("    the SmoothL1 objective, thres via a per-step proximal soft-threshold")
    print("    the surviving weights adapt to. As in TSC, thres tends to be the")
    print("    effective sparsity knob while λ is comparatively inert — let the")
    print("    equal-sparsity nrmse gap here show which dominates for this data.")
    return lam_rows, thr_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mg")
    # Reservoir hyperparameters: None = per-dataset from configs/TSR_settings_
    # sl1.json (best params, on by default), falling back to built-in defaults.
    p.add_argument("--units", type=int, default=None)
    p.add_argument("--sr", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--input_scaling", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--noise_rc", type=float, default=None)
    p.add_argument("--no_best_params", dest="use_best_params",
                   action="store_false", default=True,
                   help="Ignore configs/TSR_settings_sl1.json and use built-in "
                        "defaults for the reservoir hyperparameters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--standardize", action="store_true",
                   help="Force per-channel z-scoring (else follow STANDARDIZE_DATASETS).")
    p.add_argument("--lam_grid", nargs="+", type=float,
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                   help="reg_param (λ) values for the λ-path sweep.")
    p.add_argument("--thr_grid", nargs="+", type=float,
                   default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                   help="per-step soft-threshold (thres) values for the threshold-path sweep.")
    p.add_argument("--lam_fixed", type=float, default=1e-2,
                   help="λ held fixed during the threshold-path sweep.")
    p.add_argument("--thr_fixed", type=float, default=config.sl1_defaults.THRES,
                   help="thres held fixed during the λ-path sweep (production default).")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    # Resolve reservoir hparams: built-in defaults ← TSR_settings_sl1 entry
    # (best params, skipped for --synthetic) ← explicit CLI values.
    hp = dict(units=500, sr=0.9, lr=0.3, input_scaling=1.0,
              warmup=100, noise_rc=1e-4)
    if args.use_best_params and not args.synthetic:
        s = config.load_settings("tsr", "sl1").get(args.dataset)
        if s is not None:
            hp.update({k: s[k] for k in hp if k in s})
            print(f"[hparams] reservoir from configs/TSR_settings_sl1.json")
    for k in hp:
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)
        setattr(args, k, hp[k])

    print("=" * 88)
    print(f"TSR λ-vs-threshold sparsity comparison  |  dataset={args.dataset}  "
          f"units={args.units}")
    print("=" * 88)

    if args.synthetic:
        print("Building SYNTHETIC sparse-truth regression (no reservoir)…")
        Xtr, ytr, Xte, yte = synthetic_regression(p=args.units, seed=args.seed)
        print(f"  train X: {Xtr.shape}   (plain fit + held-out NRMSE)")
        fit_eval = lambda lam, thr: eval_synth(Xtr, ytr, Xte, yte, lam, thr)
    else:
        from data_loader import read_data
        from tsr_centralized_eval import STANDARDIZE_DATASETS
        print(f"Loading '{args.dataset}' + building reservoir states (units={args.units})…")
        Xtr, ytr, Xte, yte = read_data(args.dataset)
        standardize = args.standardize or (args.dataset in STANDARDIZE_DATASETS)
        print(f"  train: {np.asarray(Xtr).shape}   standardize={standardize}")
        fit_eval = lambda lam, thr: eval_window(
            Xtr, ytr, Xte, yte, units=args.units, sr=args.sr, lr=args.lr,
            input_scaling=args.input_scaling, warmup=args.warmup,
            noise_rc=args.noise_rc, seed=args.seed,
            reg_param=lam, threshold=thr, standardize=standardize)

    compare(fit_eval, args.lam_grid, args.thr_grid, args.lam_fixed, args.thr_fixed)
    print("=" * 88)


if __name__ == "__main__":
    main()