# pruning.py — magnitude pruning + Iterative Magnitude Pruning (IMP) baselines.
#
# The prune-aware "CE + IMP" (classification) and "ridge + IMP" (regression)
# baselines that the DEVLOG identifies as the honest competitor to the SL1
# penalty: SL1 only earns its cost where it beats IMP at MATCHED sparsity.
#
# Used post-hoc by the search drivers (tsc/tsr/tsc_fl): after a run reaches some
# sparsity S via SL1 (or L2 + threshold), we take the DENSE fit, IMP-prune it to
# the same S, and compare accuracy/error at equal sparsity.
#
# Reuses the shared per-class Newton primitives (funcs.ce_l2_*, newton_solver)
# rather than the private copies in sl1_factor_benchmark.py, so there is one
# authoritative IMP implementation.

import numpy as np

from newton_solver import damped_solve, armijo_line_search


# ─── Global magnitude pruning ─────────────────────────────────────────────────

def magnitude_prune(w: np.ndarray, target_sp_pct: float):
    """Zero the smallest-|w| entries until *target_sp_pct* % of w is zero.

    Global (across the whole weight matrix), matching the one-shot prune baseline.
    Returns (w_pruned, actual_sparsity_pct).
    """
    w = np.asarray(w, dtype=float)
    k = int(round(target_sp_pct / 100.0 * w.size))
    if k <= 0:
        return w.copy(), 100.0 * (1.0 - np.count_nonzero(w) / w.size)
    k = min(k, w.size - 1)
    cutoff = np.partition(np.abs(w).ravel(), k)[k]
    wp = np.where(np.abs(w) < cutoff, 0.0, w)
    actual = 100.0 * (1.0 - np.count_nonzero(wp) / wp.size)
    return wp, actual


def _prune_schedule(w: np.ndarray, target_sp: float, n_rounds: int):
    """Geometric-ish (linear here) ramp from current sparsity to *target_sp*."""
    cur_sp = 100.0 * (1.0 - np.count_nonzero(w) / w.size)
    if target_sp <= cur_sp:
        return [target_sp]
    return list(np.linspace(cur_sp, target_sp, n_rounds + 1)[1:])


# ─── Classification: CE + IMP (prune-then-finetune) ───────────────────────────

def ce_imp_prune(X: np.ndarray, y_oh: np.ndarray, w_dense: np.ndarray,
                 target_sp: float, *, l2: float = 1e-3,
                 n_rounds: int = 5, n_refit: int = 8) -> np.ndarray:
    """Iterative Magnitude Pruning of a dense CE fit — the strong CE baseline.

    Ramp sparsity to *target_sp* over *n_rounds*; after each prune, fine-tune the
    SURVIVING weights with *n_refit* masked Newton steps on the CE+L2 objective
    (pruned weights held at exactly zero via a per-class support mask).

    Args:
        X:         states (n_samples, units).
        y_oh:      one-hot labels (n_samples, n_classes).
        w_dense:   dense CE(+L2) readout (units, n_classes) to prune.
        target_sp: target sparsity in %.
        l2:        L2 weight for the refit objective.
        n_rounds:  prune rounds to ramp to target_sp.
        n_refit:   masked Newton refit steps per round.

    Returns:
        Pruned + refit weights (units, n_classes).
    """
    from funcs import ce_l2_grad, ce_l2_hess, ce_l2_fval

    w = np.asarray(w_dense, dtype=float).copy()
    n_classes = w.shape[1]
    for sp in _prune_schedule(w, target_sp, n_rounds):
        w, _ = magnitude_prune(w, sp)
        mask = (w != 0.0)
        for _ in range(n_refit):
            g = ce_l2_grad(w, X, y_oh, l2)
            H = ce_l2_hess(w, X, y_oh, l2)
            direction = np.zeros_like(w)
            for k in range(n_classes):
                idx = np.where(mask[:, k])[0]
                if idx.size == 0:
                    continue
                delta, _ = damped_solve(H[k][np.ix_(idx, idx)], g[idx, k])
                direction[idx, k] = -delta
            t = armijo_line_search(
                lambda ww: ce_l2_fval(ww * mask, X, y_oh, l2), w, direction, g)
            w = (w + t * direction) * mask
    return w


# ─── Regression: ridge + IMP (prune-then-refit) ───────────────────────────────

def ridge_imp_prune(X: np.ndarray, Y: np.ndarray, w_dense: np.ndarray,
                    target_sp: float, *, reg_param: float = 1e-3,
                    n_rounds: int = 5) -> np.ndarray:
    """Regression analogue of CE+IMP: prune, then closed-form ridge refit on the
    surviving support (per output column).

    The least-squares refit is exact in closed form, so no inner Newton loop is
    needed — each round prunes to the next sparsity on the ramp and re-solves the
    ridge system restricted to the live columns.

    Args:
        X:         states (n_samples, units).
        Y:         targets (n_samples, n_outputs).
        w_dense:   dense (ridge/OLS) readout (units, n_outputs) to prune.
        target_sp: target sparsity in %.
        reg_param: ridge λ for the support refit.
        n_rounds:  prune rounds to ramp to target_sp.

    Returns:
        Pruned + refit weights (units, n_outputs).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(w_dense, dtype=float).copy()
    n_out = w.shape[1]
    for sp in _prune_schedule(w, target_sp, n_rounds):
        w, _ = magnitude_prune(w, sp)
        for k in range(n_out):
            idx = np.where(w[:, k] != 0.0)[0]
            if idx.size == 0:
                continue
            Xs = X[:, idx]
            A  = Xs.T @ Xs + reg_param * np.eye(idx.size)
            b  = Xs.T @ Y[:, k]
            w[idx, k] = np.linalg.solve(A, b)
    return w
