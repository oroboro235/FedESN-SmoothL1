# sl1_threshold_timing_test.py — does WHEN you threshold matter?
#
# Standalone A/B test. Does NOT touch production code (newton_solver.py); it
# re-implements the SL1 Newton continuation here and only varies the *timing* of
# the threshold, reusing data/reservoir/primitives from sl1_factor_benchmark.py
# (TSC) and tsr_centralized_eval.py / tsr_sparsity_compare.py (TSR).
#
# Both variants keep EXACTLY the same twice-differentiable SmoothL1 objective and
# the same per-column Gauss-Newton Hessian — the second-order property is
# untouched.  The only difference is when exact zeros are produced:
#
#   A. per_iter   — soft-threshold every Newton step  (what newton_solver.py does;
#                   added there to mirror the FL server's per-round prox).
#   B. final_soft — NO thresholding inside the loop; the smooth Newton + α
#                   continuation runs to convergence, then the SAME soft-threshold
#                   is applied N times in a row at the end (N = iterations actually
#                   run, i.e. the same total prox budget as A; override with
#                   --final_reps).  Note k soft-thresholds ≡ one soft-threshold at
#                   k·thres, so this isolates *interleaving* from *total shrinkage*.
#   (final_only — the original L1General-style single hard cut — is still
#    implemented in solve_sl1 but no longer run by default.)
#
# Question: with the total shrinkage held equal, does interleaving the prox with
# the Newton steps matter — for convergence (opt_cond reaching opt_tol), wall
# time, task quality, and the exact-zero count?
#
# Tasks:
#   --task tsc (default) — classification: CE + SmoothL1 (ce_sl1_*), quality = acc.
#   --task tsr           — regression: MSE + SmoothL1 (mse_sl1_*, the objective
#                          solve_newton_mse_l1 minimises), quality = auto-regressive
#                          NRMSE + VPT (teacher-forced NRMSE under --synthetic).
#   --task tsc_fl        — federated classification: the training states (from the
#                          shared/global reservoir, as in FL) are row-partitioned
#                          IID across --n_clients; each "iteration" is one FL round
#                          — clients contribute local CE+SmoothL1 grad/Hessian at
#                          the global w, the server sums them and takes one damped
#                          Newton step (server.aggregate_parameters).  Summing the
#                          per-client objectives counts the λ·SmoothL1 penalty (and
#                          the α smoothing) once per client — a real property of the
#                          production FL objective, reproduced here.  per_iter then
#                          IS the server's per-round soft-threshold (server.py step
#                          5); final_soft defers all of them past the last round.
#                          Step size follows the server's optional line-search mode
#                          (Armijo on the aggregated loss; default FL uses α=1).
#
# Usage:
#   python sl1_threshold_timing_test.py --synthetic --units 200          # no deps
#   python sl1_threshold_timing_test.py --dataset jpv --units 500 --alpha_max 1e5
#   python sl1_threshold_timing_test.py --dataset jpv --units 500 --trace
#   python sl1_threshold_timing_test.py --task tsr --synthetic --units 200
#   python sl1_threshold_timing_test.py --task tsr --dataset mg --units 500 \
#       --reg_param 0.1 --alpha_max 1e6      # 1e6 = Reg_Node's regression α cap
#   python sl1_threshold_timing_test.py --task tsc_fl --dataset jpv --units 500 \
#       --n_clients 5

import argparse
import time

import numpy as np

from funcs import (ce_sl1_fval, ce_sl1_grad, ce_sl1_hess,
                   mse_sl1_fval, mse_sl1_grad, mse_sl1_hess)
from sl1_factor_benchmark import (build_states, synthetic_states, evaluate,
                                  newton_direction_counted, armijo_counted)


def make_objective(task, X, y, reg_param):
    """Return (fval, gval, hval) closures of (w, alpha) for the chosen task.

    Both tasks share shape conventions — grad (F, C), Hessian a list of C
    (F, F) blocks — so solve_sl1 is objective-agnostic.
    """
    if task == "tsc":
        return (lambda w, a: ce_sl1_fval(w, X, y, a, reg_param),
                lambda w, a: ce_sl1_grad(w, X, y, a, reg_param),
                lambda w, a: ce_sl1_hess(w, X, y, a, reg_param))
    return (lambda w, a: mse_sl1_fval(w, X, y, a, reg_param),
            lambda w, a: mse_sl1_grad(w, X, y, a, reg_param),
            lambda w, a: mse_sl1_hess(w, X, a, reg_param))


def partition_rows(X, y, n_clients, seed):
    """IID row-partition of the training states into n_clients shards."""
    idx = np.random.RandomState(seed).permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]


def make_fl_objective(shards, reg_param):
    """Aggregated FL objective: Σ_k local CE+SmoothL1 over the client shards.

    Mirrors server.aggregate_parameters: each client evaluates grad/Hessian of
    its LOCAL CE + λ·SmoothL1 at the global w (client._resolve_fns), the server
    sums them — so the penalty enters once per client, exactly as in production.
    """
    def fval(w, a):
        return sum(ce_sl1_fval(w, Xk, yk, a, reg_param) for Xk, yk in shards)

    def gval(w, a):
        return sum(ce_sl1_grad(w, Xk, yk, a, reg_param) for Xk, yk in shards)

    def hval(w, a):
        blocks = None
        for Xk, yk in shards:
            hk = ce_sl1_hess(w, Xk, yk, a, reg_param)
            blocks = hk if blocks is None else [b + h for b, h in zip(blocks, hk)]
        return blocks

    return fval, gval, hval


def solve_sl1(fval, gval, hval, Wout0, *, threshold_mode,
              alpha_init=1.0, alpha_max=1e5, update=2.0,
              thres=1e-5, max_iter=250, opt_tol=1e-6, prog_tol=1e-9,
              final_reps=None, record=False):
    """SL1 Newton + α continuation; *threshold_mode* selects the cut timing.

      'per_iter'   — soft-threshold w after every step (production behaviour).
      'final_soft' — never threshold inside the loop; apply the soft-threshold
                     *final_reps* times in a row at the very end (default: as many
                     times as the loop iterated, matching per_iter's prox budget).
      'final_only' — never threshold inside the loop; hard-cut |w|<thres once at
                     the very end (L1General UnconstrainedApx behaviour).

    Convergence test mirrors L1General: opt_cond = Σ|g| over the active set
    (|w| >= thres) must fall below opt_tol AND α must have reached alpha_max.
    For the final_* modes, the active set shrinks naturally as the continuation
    drives would-be-zero weights below thres, so opt_cond can reach opt_tol.

    Returns (w, stats). stats includes the per-iteration opt_cond trace if record.
    """
    w     = np.array(Wout0, dtype=float)
    alpha = alpha_init
    f_old = fval(w, alpha)

    converged = False
    iters = 0
    total_back = 0
    t_sum = 0.0
    trace = []

    t_start = time.perf_counter()
    for it in range(max_iter):
        iters = it + 1

        g = gval(w, alpha)
        H = hval(w, alpha)
        direction, _esc, _mu = newton_direction_counted(H, g)
        t, n_back = armijo_counted(lambda ww: fval(ww, alpha), w, direction, g)
        total_back += n_back
        t_sum += t

        w = w + t * direction
        if threshold_mode == "per_iter":
            # Soft-threshold (L1 prox) every step — perturbs the iterate off the
            # smooth Newton trajectory (the source of non-convergence).
            w = np.sign(w) * np.maximum(np.abs(w) - thres, 0.0)
        # 'final_soft' / 'final_only': no in-loop modification of w.

        old_alpha = alpha
        alpha = min(alpha * update, alpha_max)

        # Active set measured (not modified) at |w| >= thres, as in L1General.
        opt_cond = float(np.sum(np.abs(g[np.abs(w) >= thres])))
        f_new    = fval(w, alpha)
        if record:
            nz = int(np.sum(np.abs(w) >= thres))
            trace.append((iters, old_alpha, opt_cond, nz))

        if opt_cond < opt_tol and old_alpha >= alpha_max:
            converged = True
            break
        if abs(f_new - f_old) < prog_tol and old_alpha >= alpha_max:
            converged = True
            break
        f_old = f_new

    reps = 0
    if threshold_mode == "final_soft":
        # Same soft-threshold operator as per_iter, applied reps times in a row.
        # (k applications ≡ one soft-threshold at k·thres — kept literal so the
        # prox count matches per_iter exactly when final_reps is None.)
        reps = iters if final_reps is None else final_reps
        for _ in range(reps):
            w = np.sign(w) * np.maximum(np.abs(w) - thres, 0.0)
    elif threshold_mode == "final_only":
        # Single hard cut at the end — the ONLY place exact zeros are produced.
        w = np.where(np.abs(w) < thres, 0.0, w)

    wall    = time.perf_counter() - t_start
    nonzero = int(np.count_nonzero(np.abs(w) >= thres)) if threshold_mode == "per_iter" \
              else int(np.count_nonzero(w))
    stats = dict(iters=iters, converged=converged, wall=wall,
                 total_back=total_back, mean_step=t_sum / max(iters, 1),
                 opt_cond=opt_cond, nonzero=nonzero, n_weights=w.size,
                 sparsity=100.0 * (1.0 - np.count_nonzero(w) / w.size),
                 final_reps=reps, trace=trace)
    return w, stats


# ─── Task data + evaluation builders ──────────────────────────────────────────

def build_tsr_problem(args):
    """Reservoir states + targets for regression, plus an auto-regressive eval_fn.

    Mirrors tsr_centralized_eval.train_and_evaluate: standardize (where the
    dataset needs it), run the reservoir over the training series, discard the
    warmup transient, add state noise on the fit copy; evaluate by free-running
    len(test) steps from the end-of-training state and reporting NRMSE + VPT.
    """
    from reservoirpy.nodes import Reservoir
    from data_loader import read_data
    from tsr_centralized_eval import (STANDARDIZE_DATASETS,
                                      _autoregressive_generate,
                                      _fit_standardizer)
    from metrics import valid_prediction_time

    X_tr, y_tr, X_ts, y_ts = read_data(args.dataset)
    standardize = args.dataset in STANDARDIZE_DATASETS
    if standardize:
        mean, std = _fit_standardizer(X_tr)
        X_tr = (X_tr - mean) / std
        y_tr = (y_tr - mean) / std
        X_ts = (X_ts - mean) / std
        y_ts = (y_ts - mean) / std

    reservoir = Reservoir(units=args.units, sr=args.sr, lr=args.lr,
                          input_scaling=args.input_scaling, seed=args.seed)
    all_states = reservoir.run(X_tr)
    fit_states = all_states[args.warmup:]
    y_reg      = y_tr[args.warmup:]
    if args.noise_rc > 0:
        fit_states = fit_states + args.noise_rc * np.random.RandomState(
            args.seed).standard_normal(fit_states.shape)

    def eval_fn(w):
        # Generation mutates the reservoir state, so re-run the training series
        # to restore the end-of-training state before each variant's forecast.
        reservoir.reset()
        states = reservoir.run(X_tr)
        preds  = _autoregressive_generate(len(X_ts), states[-1], reservoir, w)
        trues  = y_ts
        if standardize:
            preds = preds * std + mean
            trues = trues * std + mean
        nrmse = float(np.sqrt(np.mean((preds - trues) ** 2))
                      / (np.std(trues) + 1e-12))
        vpt = valid_prediction_time(np.asarray(preds), np.asarray(trues))
        return {"nrmse": nrmse, "vpt": vpt}

    return fit_states, y_reg, eval_fn


def build_tsr_synthetic(args):
    """Sparse-ground-truth synthetic regression (no reservoirpy / datasets)."""
    from tsr_sparsity_compare import synthetic_regression

    Xtr, ytr, Xte, yte = synthetic_regression(p=args.units, seed=args.seed)

    def eval_fn(w):
        nrmse = float(np.sqrt(np.mean((Xte @ w - yte) ** 2))
                      / (np.std(yte) + 1e-12))
        return {"nrmse": nrmse, "vpt": None}    # no dynamics ⇒ no VPT

    return Xtr, ytr, eval_fn


def _fmt_quality(st):
    if "acc" in st:
        return f"acc={st['acc']:5.1f}%"
    if st.get("vpt") is not None:
        return f"nrmse={st['nrmse']:.4f} vpt={st['vpt']:4d}"
    return f"nrmse={st['nrmse']:.4f}"


def run_one(name, mode, objective, Wout0, eval_fn, args, record=False):
    fval, gval, hval = objective
    final_reps = None if args.final_reps <= 0 else args.final_reps
    w, st = solve_sl1(fval, gval, hval, Wout0, threshold_mode=mode,
                      alpha_init=args.alpha_init, alpha_max=args.alpha_max,
                      update=args.alpha_multiplier, thres=args.thres,
                      max_iter=args.max_iter, final_reps=final_reps, record=record)
    st.update(eval_fn(w))
    st["quality"] = _fmt_quality(st)
    reps = f"  reps={st['final_reps']:4d}" if mode == "final_soft" else ""
    print(f"  [{name:24s}] conv={'Y' if st['converged'] else 'N'}  "
          f"iters={st['iters']:4d}  wall={st['wall']:7.2f}s  "
          f"opt_cond={st['opt_cond']:.3e}  step={st['mean_step']:.3f}  "
          f"backtrk={st['total_back']:5d}  {st['quality']}  "
          f"sp={st['sparsity']:4.1f}%{reps}")
    return st


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="tsc", choices=["tsc", "tsr", "tsc_fl"],
                   help="tsc: CE+SmoothL1 classification (default); "
                        "tsr: MSE+SmoothL1 regression; "
                        "tsc_fl: federated classification (client-summed "
                        "objective, one Newton aggregation step per round)")
    p.add_argument("--n_clients", type=int, default=5,
                   help="[tsc_fl] number of IID client shards (default: 5, "
                        "matching tsc_fl_eval)")
    p.add_argument("--dataset", default=None,
                   help="TSC: jpv/har/… (default jpv); TSR: mg/lorenz/… (default mg)")
    p.add_argument("--units", type=int, default=500)
    p.add_argument("--sr", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reg_param", type=float, default=1.0)
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=1e5)
    p.add_argument("--alpha_multiplier", type=float, default=2.0)
    p.add_argument("--thres", type=float, default=1e-5)
    p.add_argument("--max_iter", type=int, default=250)
    p.add_argument("--final_reps", type=int, default=0,
                   help="How many soft-thresholds variant B applies at the end "
                        "(0 = match the number of Newton iterations it ran).")
    # TSR-only knobs (mirror tsr_centralized_eval defaults)
    p.add_argument("--warmup", type=int, default=100,
                   help="[tsr] transient steps discarded before the readout fit")
    p.add_argument("--noise_rc", type=float, default=1e-3,
                   help="[tsr] state noise on the fit copy (stabilises generation)")
    p.add_argument("--trace", action="store_true",
                   help="Print the per-iteration opt_cond trace for both modes.")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.dataset is None:
        args.dataset = "mg" if args.task == "tsr" else "jpv"

    print("=" * 92)
    fl_info = f"  n_clients={args.n_clients}" if args.task == "tsc_fl" else ""
    print(f"SL1 threshold-timing A/B  |  task={args.task}  dataset={args.dataset}  "
          f"units={args.units}  reg_param={args.reg_param}  "
          f"alpha_max={args.alpha_max:g}  thres={args.thres:g}{fl_info}")
    print("=" * 92)

    if args.task == "tsr":
        if args.synthetic:
            print(f"Building SYNTHETIC regression problem (p={args.units})…")
            Xtr, ytr, eval_fn = build_tsr_synthetic(args)
        else:
            print(f"Building reservoir states (units={args.units})…")
            Xtr, ytr, eval_fn = build_tsr_problem(args)
        y_target = ytr
        out_dim  = ytr.shape[1]
        print(f"  train states: {Xtr.shape}   outputs: {out_dim}\n")
    else:
        if args.synthetic:
            print(f"Building SYNTHETIC states (units={args.units})…")
            Xtr, ytr_oh, Xte, yte, n_classes = synthetic_states(args.units, args.seed)
        else:
            print(f"Building reservoir states (units={args.units})…")
            Xtr, ytr_oh, Xte, yte, n_classes = build_states(
                args.dataset, args.units, args.sr, args.lr,
                args.input_scaling, args.seed)
        y_target = ytr_oh
        out_dim  = n_classes
        eval_fn  = lambda w: {"acc": evaluate(Xte, yte, w)}   # noqa: E731
        print(f"  train states: {Xtr.shape}   classes: {n_classes}\n")

    if args.task == "tsc_fl":
        shards    = partition_rows(Xtr, y_target, args.n_clients, args.seed)
        objective = make_fl_objective(shards, args.reg_param)
        print(f"  FL shards (samples/client): {[len(x) for x, _ in shards]}\n")
    else:
        objective = make_objective(args.task, Xtr, y_target, args.reg_param)
    Wout0     = np.zeros((Xtr.shape[1], out_dim))

    print("A/B (same SmoothL1 objective + same Gauss-Newton Hessian + same soft-threshold "
          "operator; only WHEN it is applied differs):")
    a = run_one("A: per_iter soft-thr", "per_iter",
                objective, Wout0, eval_fn, args, record=args.trace)
    b = run_one("B: final multi soft-thr", "final_soft",
                objective, Wout0, eval_fn, args, record=args.trace)

    if args.trace:
        print("\nopt_cond trace (every 25 iters; does B drive it below opt_tol?):")
        print(f"  {'iter':>5} | {'A opt_cond':>11} {'A nnz':>6} | {'B opt_cond':>11} {'B nnz':>6}")
        ta = {r[0]: r for r in a["trace"]}
        tb = {r[0]: r for r in b["trace"]}
        its = sorted(set(ta) | set(tb))
        for it in its:
            if it <= 5 or it % 25 == 0 or it == its[-1]:
                ra, rb = ta.get(it), tb.get(it)
                sa = f"{ra[2]:11.3e} {ra[3]:6d}" if ra else f"{'—':>11} {'—':>6}"
                sb = f"{rb[2]:11.3e} {rb[3]:6d}" if rb else f"{'—':>11} {'—':>6}"
                print(f"  {it:5d} | {sa} | {sb}")

    print("\n" + "=" * 92)
    print("VERDICT (equal total shrinkage; only interleaving with Newton differs):")
    if b["converged"] and not a["converged"]:
        spd = a["wall"] / max(b["wall"], 1e-9)
        print(f"  Deferring ALL {b['final_reps']} soft-thresholds to the END makes "
              f"the smooth Newton CONVERGE (iter {b['iters']}) while per-iter never does.")
        print(f"  ⇒ {spd:.1f}× faster ({a['wall']:.1f}s → {b['wall']:.1f}s), "
              f"quality {a['quality']} → {b['quality']}, sparsity "
              f"{a['sparsity']:.1f}%→{b['sparsity']:.1f}%.")
        print("  Same C² objective, same Hessian, same prox budget — only the timing changed.")
    elif b["converged"] and a["converged"]:
        print("  Both converged; compare iters/wall/quality/sparsity above.")
    else:
        print("  B did not converge either — the non-convergence is NOT caused by the")
        print("  in-loop threshold. Likely thres <= 1/alpha_max, so the active set")
        print("  still contains the ~1/α 'zero' weights; try a larger thres (e.g.")
        print("  1e-4) or larger alpha_max so 1/alpha_max << thres.")
        if a["converged"]:
            print("  (A converged while B did not — interleaving actively helps here.)")
    print("=" * 92)


if __name__ == "__main__":
    main()
