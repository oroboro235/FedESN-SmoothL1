# sl1_factor_benchmark.py — isolate WHICH of two factors dominates the slow SL1
# Newton path on ESN(units=500) readout training.
#
# This is a standalone diagnostic, separate from the production solver
# (newton_solver.solve_newton_l1).  It re-implements the SAME continuation loop
# but instrumented, so it can report — per iteration — the Hessian-build time,
# the per-class solve time, the damped-Newton ridge μ escalations, the Armijo
# backtracks, and (crucially) how many iterations are spent doing nothing but
# grinding α up to alpha_max after the problem is already optimal.
#
# Two factors, toggled independently in a 2×2 design:
#
#   A. CONDITIONING  — standardise the reservoir states (z-score per column)
#                      vs. use them raw.  Correlated reservoir states make the
#                      CE Hessian ill-conditioned, which forces the μ escalation
#                      in damped_solve and makes Armijo reject the full step.
#
#   B. ALPHA GATE    — converge as soon as opt_cond < opt_tol  ("opt_only")
#                      vs. the production gate that ALSO requires the
#                      continuation to have reached alpha_max  ("alpha_max").
#                      The production gate forces the full α schedule every time,
#                      no matter how converged the iterate already is.
#
# The verdict compares the marginal wall-time saved by toggling each factor
# alone from the baseline (raw states, alpha_max gate).
#
# Usage:
#   python sl1_factor_benchmark.py
#   python sl1_factor_benchmark.py --dataset char --units 500 --reg_param 1.0
#   python sl1_factor_benchmark.py --dataset jpv --sr 0.9 --lr 0.3 --input_scaling 1.0

import argparse
import time

import numpy as np

from funcs import (ce_sl1_fval, ce_sl1_grad, ce_sl1_hess,
                   ce_l2_fval, ce_l2_grad, ce_l2_hess)


# ─── Instrumented primitives (mirror newton_solver, plus counters) ────────────

def damped_solve_counted(H, g, mu_init=1.0, factor=10.0, max_iter=50):
    """Damped Newton solve (H + μI)δ = g with adaptive ridge μ.

    Same logic as newton_solver.damped_solve but returns the number of μ
    escalations it needed (0 = the very first, un-ridged solve was accepted).
    A high escalation count is the fingerprint of an ill-conditioned Hessian.
    """
    mu  = mu_init
    n   = H.shape[0]
    eye = np.eye(n)
    for n_escalations in range(max_iter):
        try:
            delta = np.linalg.solve(H + mu * eye, g)
            if float(np.max(np.abs(delta))) < 1e2:
                return delta, mu, n_escalations
        except np.linalg.LinAlgError:
            pass
        mu *= factor
    return np.zeros_like(g), mu, max_iter


def newton_direction_counted(hessian_blocks, grad):
    """Per-class damped Newton direction; sums μ escalations over the classes."""
    direction    = np.zeros_like(grad)
    total_escal  = 0
    max_mu       = 0.0
    for k in range(grad.shape[1]):
        delta, mu, esc = damped_solve_counted(hessian_blocks[k], grad[:, k])
        direction[:, k] = -delta
        total_escal    += esc
        max_mu          = max(max_mu, mu)
    return direction, total_escal, max_mu


def armijo_counted(loss_fn, w, direction, grad, c=1e-4, rho=0.5, max_iter=20):
    """Armijo backtracking; returns (alpha, n_backtracks)."""
    dir_deriv = float(np.sum(grad * direction))
    if dir_deriv >= 0:
        return 1.0, 0
    f0    = loss_fn(w)
    alpha = 1.0
    for n_back in range(max_iter):
        if loss_fn(w + alpha * direction) <= f0 + c * alpha * dir_deriv:
            return alpha, n_back
        alpha *= rho
    return alpha, max_iter


# ─── Instrumented continuation loop ───────────────────────────────────────────

def solve_instrumented(X, y_oh, Wout0, reg_param, *,
                       alpha_init=1.0, alpha_max=5e6,
                       update1=2.0, update2=2.0,
                       thres=1e-5, max_iter=250,
                       opt_tol=1e-6, prog_tol=1e-9,
                       gate="alpha_max", trace_eval=None,
                       patience=None, stag_tol=0.002):
    """Run the SL1 Newton continuation, recording per-iteration diagnostics.

    gate ∈ {"alpha_max", "opt_only"}:
      "alpha_max" — production gate: stop only when opt_cond < opt_tol AND the
                    continuation has reached alpha_max (factor B = ON).
      "opt_only"  — stop as soon as opt_cond < opt_tol, whatever α is
                    (factor B relaxed).

    patience: if not None, enable STAGNATION early-stop — the criterion the
      traces showed is the real one (the support set / opt_cond freezes at a
      non-zero fixed point, it does NOT shrink to opt_tol).  Stop once the
      support set (fraction of non-zero weights) has changed by less than
      *stag_tol* across the last *patience* iterations AND the α continuation
      has reached alpha_max (so we never stop mid-ramp).  This is what makes the
      λ-path and thres-path comparable: each is run only until it settles.

    Returns (w, stats) where stats is a dict of aggregate diagnostics.
    """
    from collections import deque
    nz_window = deque(maxlen=patience) if patience else None
    w     = np.array(Wout0, dtype=float)
    alpha = alpha_init
    f_old = ce_sl1_fval(w, X, y_oh, alpha, reg_param)

    t_hess = t_solve = t_loss = 0.0
    total_escal = total_back = 0
    t_sum = 0.0                     # sum of accepted Armijo step sizes (→ mean step)
    gate_blocked_iters = 0          # opt_cond already < opt_tol but α < alpha_max
    converged = False
    iters = 0
    # Per-iteration trace: (iter, alpha, opt_cond, f, acc, sparsity).  trace_eval,
    # when given, is (Xte, yte_int) so we can score held-out acc each iteration
    # and locate the plateau (where extra iterations stop buying anything).
    history = []

    for it in range(max_iter):
        iters = it + 1

        t0 = time.perf_counter()
        g = ce_sl1_grad(w, X, y_oh, alpha, reg_param)
        H = ce_sl1_hess(w, X, y_oh, alpha, reg_param)
        t_hess += time.perf_counter() - t0

        t0 = time.perf_counter()
        direction, escal, _max_mu = newton_direction_counted(H, g)
        t_solve += time.perf_counter() - t0
        total_escal += escal

        t0 = time.perf_counter()
        t, n_back = armijo_counted(
            lambda ww: ce_sl1_fval(ww, X, y_oh, alpha, reg_param),
            w, direction, g,
        )
        t_loss += time.perf_counter() - t0
        total_back += n_back
        t_sum += t

        w = w + t * direction
        w = np.sign(w) * np.maximum(np.abs(w) - thres, 0.0)

        old_alpha = alpha
        alpha = min(alpha * (update2 if t == 1.0 else update1), alpha_max)

        opt_cond = float(np.sum(np.abs(g[np.abs(w) >= thres])))
        f_new    = ce_sl1_fval(w, X, y_oh, alpha, reg_param)

        if trace_eval is not None:
            Xte, yte_int = trace_eval
            acc_it = float((np.argmax(Xte @ w, axis=1) == yte_int).mean() * 100)
            sp_it  = 100.0 * (1.0 - int(np.sum(np.abs(w) >= thres)) / w.size)
            history.append((iters, old_alpha, opt_cond, f_new, acc_it, sp_it))

        # Stagnation early-stop: support set frozen across the patience window
        # while α is already at its ceiling (the genuine fixed point the traces
        # show, where opt_cond freezes at a non-zero constant).
        if nz_window is not None:
            nz_frac = int(np.sum(np.abs(w) >= thres)) / w.size
            nz_window.append(nz_frac)
            if (old_alpha >= alpha_max and len(nz_window) == nz_window.maxlen
                    and (max(nz_window) - min(nz_window)) < stag_tol):
                converged = True
                iters = it + 1
                break

        opt_ok = opt_cond < opt_tol
        # Count iterations the production gate is forced to keep running purely
        # because α has not yet reached alpha_max (this is factor B's cost).
        if opt_ok and old_alpha < alpha_max:
            gate_blocked_iters += 1

        if gate == "opt_only":
            if opt_ok:
                converged = True
                break
        else:  # "alpha_max"
            if opt_ok and old_alpha >= alpha_max:
                converged = True
                break
            if abs(f_new - f_old) < prog_tol and old_alpha >= alpha_max:
                converged = True
                break

        f_old = f_new

    nonzero = int(np.sum(np.abs(w) >= thres))
    stats = dict(
        iters=iters, converged=converged,
        t_hess=t_hess, t_solve=t_solve, t_loss=t_loss,
        total_escal=total_escal, total_back=total_back,
        gate_blocked_iters=gate_blocked_iters,
        final_alpha=alpha, opt_cond=opt_cond,
        mean_step=t_sum / max(iters, 1),
        nonzero=nonzero, n_weights=w.size,
        history=history,
    )
    return w, stats


# ─── Data / reservoir setup ───────────────────────────────────────────────────

def build_states(dataset, units, sr, lr, input_scaling, seed):
    """Load dataset, run reservoir, return (Xtr_states, ytr_oh, Xte_states, yte_int)."""
    # Imported lazily so the solver logic can be smoke-tested (--synthetic) on a
    # machine without reservoirpy / the dataset files installed.
    from reservoirpy.nodes import Reservoir
    from data_loader import read_data, one_hot, standardize

    Xtr, ytr, Xte, yte = read_data(dataset)

    def to_int(y):
        return np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)

    ytr_int, yte_int = to_int(ytr), to_int(yte)
    n_classes = int(max(ytr_int.max(), yte_int.max())) + 1

    def ensure_3d(X):
        return X[:, :, None] if X.ndim == 2 else X

    Xtr, Xte = ensure_3d(Xtr), ensure_3d(Xte)
    Xtr, Xte = standardize(Xtr, Xte)

    reservoir = Reservoir(units=units, lr=lr, sr=sr,
                          input_scaling=input_scaling, seed=seed)

    def extract(X):
        out = []
        for x in X:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            out.append(reservoir.run(x)[-1].copy())
            reservoir.reset()
        return np.vstack(out)

    Xtr_s = extract(Xtr)
    Xte_s = extract(Xte)
    return Xtr_s, one_hot(ytr_int, n_classes), Xte_s, yte_int, n_classes


def synthetic_states(units, seed, n_per_class=60, n_classes=4):
    """Correlated, off-centre synthetic states — mimics the ill-conditioned,
    non-zero-mean reservoir states that make the raw CE Hessian hard.  For
    smoke-testing only (no reservoirpy / datasets needed)."""
    rng = np.random.RandomState(seed)
    # Low-rank + offset → strong column correlation and a large mean (the two
    # things z-scoring fixes), so the raw vs standardised contrast is visible.
    basis = rng.randn(units, units // 10)
    def gen(n, c):
        z = rng.randn(n, units // 10)
        X = z @ basis.T + 5.0 + 0.5 * c
        X += 0.05 * rng.randn(n, units)
        return X
    Xtr = np.vstack([gen(n_per_class, c) for c in range(n_classes)])
    Xte = np.vstack([gen(n_per_class // 2, c) for c in range(n_classes)])
    ytr = np.repeat(np.arange(n_classes), n_per_class)
    yte = np.repeat(np.arange(n_classes), n_per_class // 2)
    eye = np.eye(n_classes)
    return Xtr, eye[ytr], Xte, yte, n_classes


def zscore_states(Xtr_s, Xte_s):
    """Standardise reservoir-state columns using train statistics (factor A)."""
    mu  = Xtr_s.mean(axis=0, keepdims=True)
    std = Xtr_s.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (Xtr_s - mu) / std, (Xte_s - mu) / std


def evaluate(Xte_s, yte_int, Wout):
    return float((np.argmax(Xte_s @ Wout, axis=1) == yte_int).mean() * 100)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_cell(label, Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args):
    """Train one (conditioning, gate) cell, timing the solve, and report."""
    Wout0 = np.zeros((Xtr_s.shape[1], n_classes))
    gate  = "opt_only" if label.endswith("opt_only") else "alpha_max"

    t0 = time.perf_counter()
    Wout, st = solve_instrumented(
        Xtr_s, ytr_oh, Wout0, args.reg_param,
        alpha_init=args.alpha_init, alpha_max=args.alpha_max,
        update1=args.alpha_multiplier, update2=args.alpha_multiplier,
        thres=args.thres, max_iter=args.max_iter, gate=gate,
    )
    wall = time.perf_counter() - t0

    acc      = evaluate(Xte_s, yte_int, Wout)
    sparsity = 100.0 * (st["nonzero"] == 0) if st["n_weights"] == 0 else \
               100.0 * (1.0 - st["nonzero"] / st["n_weights"])
    st.update(wall=wall, acc=acc, sparsity=sparsity)
    print(f"  [{label:28s}] wall={wall:7.2f}s  iters={st['iters']:4d}  "
          f"conv={'Y' if st['converged'] else 'N'}  "
          f"solve={st['t_solve']:6.2f}s hess={st['t_hess']:6.2f}s  "
          f"μ-esc={st['total_escal']:5d}  backtrk={st['total_back']:4d}  "
          f"step={st['mean_step']:.3f}  gate-wait={st['gate_blocked_iters']:4d}  "
          f"acc={acc:5.1f}%  sp={sparsity:4.1f}%")
    return st


def sweep_alpha_max(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args, ceilings):
    """Sweep the α ceiling on RAW states (opt_only gate, so a run can actually
    stop early when the smoothed problem is easy enough to converge).

    This is the lever the 2×2 design did NOT test: the 2×2 only toggled the stop
    gate while alpha_max stayed 5e6 in every cell.  If the slowness is the stiff
    late-continuation phase at α=5e6, lowering the ceiling lets Newton converge
    in few iterations (the paper's regime) at little/no accuracy cost.
    """
    print("\nα-ceiling sweep on RAW states (gate=opt_only):")
    print(f"  {'alpha_max':>10} | {'wall':>7} | {'iters':>5} | conv | "
          f"{'mean_step':>9} | {'backtrk':>7} | {'acc':>5} | {'sp':>5}")
    Wout0 = np.zeros((Xtr_s.shape[1], n_classes))
    for amax in ceilings:
        t0 = time.perf_counter()
        Wout, st = solve_instrumented(
            Xtr_s, ytr_oh, Wout0, args.reg_param,
            alpha_init=args.alpha_init, alpha_max=amax,
            update1=args.alpha_multiplier, update2=args.alpha_multiplier,
            thres=args.thres, max_iter=args.max_iter, gate="opt_only",
        )
        wall = time.perf_counter() - t0
        acc  = evaluate(Xte_s, yte_int, Wout)
        sp   = 100.0 * (1.0 - st["nonzero"] / st["n_weights"])
        print(f"  {amax:>10.0e} | {wall:6.2f}s | {st['iters']:5d} | "
              f"{'Y' if st['converged'] else 'N':^4} | {st['mean_step']:9.4f} | "
              f"{st['total_back']:7d} | {acc:4.1f}% | {sp:4.1f}%")


def trace_plateau(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args, amax):
    """Run ONE config to full max_iter and locate the plateau — the first
    iteration after which held-out acc stays within 0.5%% of its final value and
    sparsity within 1%% of its final value.  Everything past the plateau is wall
    time the (missing) early-stop is throwing away.
    """
    print(f"\nPer-iteration trace on RAW states  (alpha_max={amax:g}, gate=opt_only):")
    Wout0 = np.zeros((Xtr_s.shape[1], n_classes))
    t0 = time.perf_counter()
    _W, st = solve_instrumented(
        Xtr_s, ytr_oh, Wout0, args.reg_param,
        alpha_init=args.alpha_init, alpha_max=amax,
        update1=args.alpha_multiplier, update2=args.alpha_multiplier,
        thres=args.thres, max_iter=args.max_iter, gate="opt_only",
        trace_eval=(Xte_s, yte_int),
    )
    wall = time.perf_counter() - t0
    hist = st["history"]
    final_acc, final_sp = hist[-1][4], hist[-1][5]

    # Plateau = last point where acc/sparsity first enter and never leave their
    # final tolerance band.
    plateau = hist[-1][0]
    for (it, _a, _oc, _f, acc, sp) in hist:
        if abs(acc - final_acc) <= 0.5 and abs(sp - final_sp) <= 1.0:
            plateau = it
            break

    print(f"  {'iter':>5} | {'alpha':>9} | {'opt_cond':>10} | {'acc':>6} | {'sp':>6}")
    last = None
    for row in hist:
        it = row[0]
        # Print a sparse trace: early iters densely, then every 25.
        if it <= 15 or it % 25 == 0 or it == hist[-1][0]:
            print(f"  {it:5d} | {row[1]:9.2e} | {row[2]:10.3e} | "
                  f"{row[4]:5.1f}% | {row[5]:5.1f}%")
        last = row
    wall_per_iter = wall / max(st["iters"], 1)
    wasted = (st["iters"] - plateau) * wall_per_iter
    print(f"\n  final: acc={final_acc:.1f}%  sp={final_sp:.1f}%  "
          f"after {st['iters']} iters, {wall:.1f}s")
    print(f"  PLATEAU reached at iter ~{plateau}  "
          f"(acc within 0.5%, sparsity within 1% of final)")
    print(f"  → ~{st['iters'] - plateau} of {st['iters']} iterations are wasted "
          f"≈ {wasted:.1f}s of {wall:.1f}s ({100*wasted/wall:.0f}%) "
          f"recoverable by a proper early-stop.")


def ce_newton(X, y_oh, n_classes, l2, max_iter=60, tol=1e-7):
    """Plain damped-Newton fit of CE (+ small L2 for conditioning) — NO SmoothL1,
    NO α continuation, NO threshold.  This is the 'fit' the ablation prunes.

    Stops when the relative objective change drops below *tol* (CE+L2 is smooth
    and convex, so Newton settles in a handful of iterations).  Returns
    (w, iters, wall).
    """
    w = np.zeros((X.shape[1], n_classes))
    f_old = ce_l2_fval(w, X, y_oh, l2)
    t0 = time.perf_counter()
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        g = ce_l2_grad(w, X, y_oh, l2)
        H = ce_l2_hess(w, X, y_oh, l2)
        d, _esc, _mu = (lambda blocks, grad: newton_direction_counted(blocks, grad))(H, g)
        t = armijo_counted(lambda ww: ce_l2_fval(ww, X, y_oh, l2), w, d, g)[0]
        w = w + t * d
        f_new = ce_l2_fval(w, X, y_oh, l2)
        if abs(f_new - f_old) <= tol * (abs(f_old) + 1e-12):
            break
        f_old = f_new
    return w, iters, time.perf_counter() - t0


def prune_to_sparsity(w, target_sp_pct):
    """Global magnitude prune: zero the smallest-|w| entries until *target_sp_pct*
    of all weights are zero.  Returns (w_pruned, actual_sparsity_pct)."""
    k = int(round(target_sp_pct / 100.0 * w.size))
    if k <= 0:
        return w.copy(), 0.0
    k = min(k, w.size - 1)
    cutoff = np.partition(np.abs(w).ravel(), k)[k]
    wp = np.where(np.abs(w) < cutoff, 0.0, w)
    actual = 100.0 * (1.0 - np.count_nonzero(wp) / wp.size)
    return wp, actual


def imp_prune(X, y_oh, Xte, yte_int, w_dense, n_classes, target_sp,
              l2, n_rounds=5, n_refit=8):
    """Iterative Magnitude Pruning (prune-then-finetune) — the STRONG CE baseline.

    Starting from the dense CE fit, ramp sparsity geometrically to *target_sp*
    over *n_rounds*; after each prune, fine-tune the SURVIVING weights with
    *n_refit* reduced-system Newton steps (the pruned weights are held at exactly
    zero via a per-class support mask, so only live weights move).  This is the
    fair, prune-aware version of CE+prune, vs the one-shot prune in ablation().

    Returns (acc, actual_sparsity, wall).
    """
    w = w_dense.copy()
    cur_sp = 100.0 * (1.0 - np.count_nonzero(w) / w.size)
    schedule = np.linspace(cur_sp, target_sp, n_rounds + 1)[1:]
    t0 = time.perf_counter()
    for sp in schedule:
        w, _ = prune_to_sparsity(w, sp)
        mask = (w != 0.0)
        for _ in range(n_refit):
            g = ce_l2_grad(w, X, y_oh, l2)
            H = ce_l2_hess(w, X, y_oh, l2)
            direction = np.zeros_like(w)
            for k in range(n_classes):
                idx = np.where(mask[:, k])[0]
                if idx.size == 0:
                    continue
                d, _mu, _esc = damped_solve_counted(
                    H[k][np.ix_(idx, idx)], g[idx, k])
                direction[idx, k] = -d
            t = armijo_counted(lambda ww: ce_l2_fval(ww * mask, X, y_oh, l2),
                               w, direction, g)[0]
            w = (w + t * direction) * mask
    wall = time.perf_counter() - t0
    acc = evaluate(Xte, yte_int, w)
    actual_sp = 100.0 * (1.0 - np.count_nonzero(w) / w.size)
    return acc, actual_sp, wall


def _run_settle(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args,
                reg_param, thres, amax):
    """One run to stagnation early-stop; returns a metrics dict."""
    Wout0 = np.zeros((Xtr_s.shape[1], n_classes))
    t0 = time.perf_counter()
    Wout, st = solve_instrumented(
        Xtr_s, ytr_oh, Wout0, reg_param,
        alpha_init=args.alpha_init, alpha_max=amax,
        update1=args.alpha_multiplier, update2=args.alpha_multiplier,
        thres=thres, max_iter=args.max_iter, gate="opt_only",
        patience=args.patience, stag_tol=args.stag_tol / 100.0,
    )
    wall = time.perf_counter() - t0
    acc  = evaluate(Xte_s, yte_int, Wout)
    sp   = 100.0 * (1.0 - st["nonzero"] / st["n_weights"])
    return dict(reg_param=reg_param, thres=thres, wall=wall,
                iters=st["iters"], settled=st["converged"], acc=acc, sp=sp)


def pareto(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args, lam_grid, thres_grid):
    """Compare the two ways of buying sparsity, each with stagnation early-stop:

      λ-path     — sweep reg_param at a fixed tiny thres (principled: the penalty
                   is part of the minimised objective; weights are shrunk, not cut).
      thres-path — sweep thres at a fixed reg_param (post-hoc hard cut).

    Same alpha_max for both.  The question: at matched sparsity, which keeps more
    accuracy, and which settles in fewer iterations / less wall time?
    """
    def show(title, rows, knob):
        print(f"\n{title}  (stagnation early-stop: patience={args.patience}, "
              f"stag_tol={args.stag_tol}pp, alpha_max={args.alpha_max:g})")
        print(f"  {knob:>10} | {'iters':>5} | settle | {'wall':>7} | "
              f"{'acc':>6} | {'sp':>6}")
        for r in rows:
            kv = r["reg_param"] if knob == "reg_param" else r["thres"]
            print(f"  {kv:>10.2e} | {r['iters']:5d} | {'Y' if r['settled'] else 'N':^6} "
                  f"| {r['wall']:6.2f}s | {r['acc']:5.1f}% | {r['sp']:5.1f}%")

    print("=" * 80)
    print("PARETO — λ-path vs thres-path (sparsity via objective vs via hard cut)")
    print("=" * 80)

    lam_rows = [_run_settle(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args,
                            lam, args.thres, args.alpha_max) for lam in lam_grid]
    show("λ-PATH  (thres fixed = %.0e)" % args.thres, lam_rows, "reg_param")

    th_rows = [_run_settle(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args,
                           args.reg_param, th, args.alpha_max) for th in thres_grid]
    show("thres-PATH  (reg_param fixed = %g)" % args.reg_param, th_rows, "thres")

    print("\nHow to read:")
    print("  • Pick a target sparsity, find that row in each table, compare acc.")
    print("    Higher acc at equal sp ⇒ that knob has the better Pareto front.")
    print("  • 'iters'/'wall' = cost to SETTLE (stagnation stop). Fewer ⇒ the")
    print("    sparsity is reached without the slow high-α crawl.")
    print("  • settle=N ⇒ never stagnated within max_iter (still drifting).")
    print("=" * 80)


def ablation(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args):
    """SL1 vs CE+prune — does the SmoothL1 penalty earn its cost?

    SL1 path:  the full Newton + α continuation + threshold + stagnation stop,
               swept over thres to trace its accuracy/sparsity Pareto.
    CE+prune:  fit CE (+ tiny L2) ONCE with plain Newton (no penalty, no α), then
               magnitude-prune to each sparsity SL1 reached, for free.

    If CE+prune matches SL1's accuracy at the same sparsity (and far less time),
    the SmoothL1 penalty is not pulling its weight; if SL1 wins, the penalty is
    genuinely shaping a more prune-robust solution.
    """
    print("=" * 84)
    print("ABLATION — does the SmoothL1 penalty earn its cost?  (SL1 vs CE-only + prune)")
    print("=" * 84)

    # 1) SL1 Pareto via the thres sweep (each with stagnation early-stop).
    sl1_rows = [_run_settle(Xtr_s, ytr_oh, Xte_s, yte_int, n_classes, args,
                            args.reg_param, th, args.alpha_max)
                for th in args.thres_grid]

    # 2) CE-only fit ONCE, then prune to each SL1 sparsity (pruning is free).
    w_ce, ce_iters, ce_wall = ce_newton(Xtr_s, ytr_oh, n_classes, args.l2)
    acc_dense = evaluate(Xte_s, yte_int, w_ce)
    print(f"\nCE-only fit (l2={args.l2:g}): {ce_iters} Newton iters, {ce_wall:.2f}s, "
          f"dense acc={acc_dense:.1f}%  (prune below is post-hoc & free)")

    print(f"\n  {'target_sp':>9} | {'SL1 acc':>7} {'SL1 t':>7} "
          f"| {'1shot acc':>9} | {'IMP acc':>7} {'IMP t':>7} | winner")
    for r in sl1_rows:
        wp, _sp1 = prune_to_sparsity(w_ce, r["sp"])
        one_acc = evaluate(Xte_s, yte_int, wp)
        imp_acc, _imp_sp, imp_wall = imp_prune(
            Xtr_s, ytr_oh, Xte_s, yte_int, w_ce, n_classes, r["sp"],
            args.l2, n_rounds=args.imp_rounds, n_refit=args.imp_refit)
        best_ce = max(one_acc, imp_acc)
        win = "SL1" if r["acc"] > best_ce + 0.3 else \
              ("CE+prune" if best_ce > r["acc"] + 0.3 else "tie")
        print(f"  {r['sp']:8.1f}% | {r['acc']:6.1f}% {r['wall']:6.2f}s "
              f"| {one_acc:8.1f}% | {imp_acc:6.1f}% {imp_wall:6.2f}s | {win}")

    print("\nVerdict:")
    print("  • '1shot' = prune the dense CE fit once (free). 'IMP' = iterative")
    print("    prune + Newton refit of survivors (the strong, prune-aware CE baseline).")
    print("  • IMP cost is on top of the one-off CE fit above; SL1 t is per row.")
    print("  • SL1 winning even vs IMP, with a gap that WIDENS at high sparsity ⇒")
    print("    the SmoothL1 penalty genuinely shapes a more prune-robust solution;")
    print("    keep it (with the fast config). If IMP catches up, the penalty's")
    print("    value is marginal and a simpler prune-aware fit would do.")
    print("=" * 84)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="char")
    p.add_argument("--units", type=int, default=500)
    p.add_argument("--sr", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--input_scaling", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reg_param", type=float, default=1.0)
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=5e6)
    p.add_argument("--alpha_multiplier", type=float, default=2.0)
    p.add_argument("--thres", type=float, default=1e-5)
    p.add_argument("--max_iter", type=int, default=250)
    p.add_argument("--synthetic", action="store_true",
                   help="Skip reservoirpy/datasets; use correlated synthetic "
                        "states to smoke-test the solver and the 2×2 logic.")
    p.add_argument("--sweep_alpha_max", nargs="*", type=float, default=None,
                   help="Run an α-ceiling sweep on raw states instead of the "
                        "2×2 (e.g. --sweep_alpha_max 1e2 1e3 1e4 1e6 5e6). "
                        "Pass with no values to use a sensible default grid.")
    p.add_argument("--trace", type=float, default=None, metavar="ALPHA_MAX",
                   help="Trace one run to max_iter and report the plateau "
                        "iteration / wasted time (e.g. --trace 1e5).")
    p.add_argument("--pareto", action="store_true",
                   help="Compare λ-path vs thres-path for buying sparsity, each "
                        "with stagnation early-stop.")
    p.add_argument("--lam_grid", nargs="+", type=float,
                   default=[1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0],
                   help="reg_param values for the λ-path sweep.")
    p.add_argument("--thres_grid", nargs="+", type=float,
                   default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                   help="thres values for the thres-path sweep.")
    p.add_argument("--patience", type=int, default=8,
                   help="Stagnation early-stop window (iterations).")
    p.add_argument("--stag_tol", type=float, default=0.2,
                   help="Stagnation tolerance in percentage POINTS of sparsity "
                        "change across the patience window (default 0.2pp).")
    p.add_argument("--ablation", action="store_true",
                   help="SL1 vs CE-only+magnitude-prune at matched sparsity.")
    p.add_argument("--l2", type=float, default=1e-3,
                   help="Small L2 weight for the CE-only fit's conditioning "
                        "(ablation baseline; default 1e-3).")
    p.add_argument("--imp_rounds", type=int, default=5,
                   help="IMP prune+refit rounds to reach target sparsity (default 5).")
    p.add_argument("--imp_refit", type=int, default=8,
                   help="Newton refit steps per IMP round (default 8).")
    args = p.parse_args()

    print("=" * 96)
    print(f"SL1 factor benchmark  |  dataset={args.dataset}  units={args.units}  "
          f"reg_param={args.reg_param}  alpha_max={args.alpha_max:g}  "
          f"max_iter={args.max_iter}")
    print("=" * 96)

    if args.synthetic:
        print(f"Building SYNTHETIC correlated states (units={args.units})…")
        Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes = synthetic_states(
            args.units, args.seed
        )
    else:
        print(f"Building reservoir states (units={args.units})…")
        Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes = build_states(
            args.dataset, args.units, args.sr, args.lr, args.input_scaling, args.seed
        )
    Xtr_std, Xte_std = zscore_states(Xtr_raw, Xte_raw)
    print(f"  train states: {Xtr_raw.shape}   classes: {n_classes}")

    # Conditioning fingerprint: condition number of the raw vs standardised
    # CE-Hessian-relevant Gram matrix XᵀX (cheap, one SVD each).
    def cond(X):
        s = np.linalg.svd(X, compute_uv=False)
        return (s[0] / s[-1]) ** 2 if s[-1] > 0 else np.inf
    print(f"  cond(XᵀX) raw = {cond(Xtr_raw):.3e}   "
          f"standardised = {cond(Xtr_std):.3e}")

    # Ablation mode: SL1 vs CE-only + magnitude prune at matched sparsity.
    if args.ablation:
        ablation(Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args)
        return

    # Pareto mode: λ-path vs thres-path, both with stagnation early-stop.
    if args.pareto:
        pareto(Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args,
               args.lam_grid, args.thres_grid)
        return

    # Trace mode: locate the plateau / wasted iterations for one α ceiling.
    if args.trace is not None:
        trace_plateau(Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args, args.trace)
        return

    # α-ceiling sweep mode (the lever the 2×2 does not test) — takes priority.
    if args.sweep_alpha_max is not None:
        ceilings = args.sweep_alpha_max or [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 5e6]
        sweep_alpha_max(Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args, ceilings)
        return

    print("\n2×2 design  (conditioning × alpha-gate):")
    res = {}
    res["raw/alpha_max"]  = run_cell("raw / alpha_max (BASELINE)",
                                     Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args)
    res["raw/opt_only"]   = run_cell("raw / opt_only",
                                     Xtr_raw, ytr_oh, Xte_raw, yte_int, n_classes, args)
    res["std/alpha_max"]  = run_cell("standardised / alpha_max",
                                     Xtr_std, ytr_oh, Xte_std, yte_int, n_classes, args)
    res["std/opt_only"]   = run_cell("standardised / opt_only",
                                     Xtr_std, ytr_oh, Xte_std, yte_int, n_classes, args)

    base = res["raw/alpha_max"]["wall"]
    d_cond  = base - res["std/alpha_max"]["wall"]   # toggle A alone
    d_alpha = base - res["raw/opt_only"]["wall"]    # toggle B alone
    d_both  = base - res["std/opt_only"]["wall"]

    print("\n" + "=" * 96)
    print("VERDICT — marginal wall-time saved from the baseline (raw / alpha_max):")
    print(f"  Factor A  (standardise states)   : {d_cond:+7.2f}s "
          f"({100*d_cond/base:+5.1f}%)")
    print(f"  Factor B  (relax alpha gate)     : {d_alpha:+7.2f}s "
          f"({100*d_alpha/base:+5.1f}%)")
    print(f"  Both together                    : {d_both:+7.2f}s "
          f"({100*d_both/base:+5.1f}%)")
    dom = "A (conditioning)" if d_cond > d_alpha else "B (alpha gate)"
    print(f"  → Dominant factor: {dom}")
    print("\nReading the diagnostics:")
    print("  • Factor B dominates if baseline 'gate-wait' is a large fraction of")
    print("    'iters' and opt_only converges in far fewer iters at similar acc.")
    print("  • Factor A dominates if baseline 'μ-esc'/'backtrk' are high and drop")
    print("    sharply once standardised (cond(XᵀX) also drops by orders).")
    print("=" * 96)


if __name__ == "__main__":
    main()
