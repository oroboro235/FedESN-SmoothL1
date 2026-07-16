# newton_solver.py — shared per-class Gauss-Newton machinery for the ESN
# classification readout.
#
# Used by BOTH:
#   • the centralized solver  (readout_node.Clr_Node, reg_type="sl1", solver="newton")
#   • the federated server     (server.Server_FedAvg.aggregate_parameters)
#
# The two settings run the *same* Newton-on-SmoothL1 algorithm and differ only
# in who supplies the (gradient, Hessian) pair:
#
#   centralized — a single dataset.  solve_newton_l1() runs the full L1General-
#                 style continuation loop in-process (many Newton iterations,
#                 α grown after each).
#   federated   — clients sum their local (grad, Hessian); the server takes ONE
#                 damped Newton step per round and the α continuation happens
#                 across rounds (client.update_alpha()).
#
# In both cases the Hessian is the per-class block-diagonal Gauss-Newton matrix
# from funcs.ce_sl1_hess (one (units, units) block per output class), so the
# Newton step decouples into n_classes independent linear solves.
#
# Public API:
#   damped_newton_direction(hessian_blocks, grad)   — per-class descent direction
#   armijo_line_search(loss_fn, w, direction, grad) — Armijo step size α ∈ (0, 1]
#   solve_newton_l1(...)                            — centralized continuation loop (CE + SmoothL1)
#   solve_newton_mse_l1(...)                        — centralized continuation loop (MSE + SmoothL1)
#   solve_newton_smooth(...)                        — centralized Newton loop (CE+L2 / CE only)

import functools

import numpy as np

import config


# ─── BLAS thread cap ──────────────────────────────────────────────────────────

def blas_capped(fn):
    """Run a Newton solve with BLAS pinned to one thread.

    The Newton step decouples into one np.linalg.solve() per output class on a
    (units, units) = 500x500 Hessian block.  A matrix that small is *pathological*
    for multi-threaded OpenBLAS LAPACK: on a 24-core box a 500x500 solve takes
    ~189 ms with the default thread count but ~2.5 ms pinned to one thread — a
    75x penalty, because the LU factorisation's synchronisation swamps the
    arithmetic.  (GEMM is unaffected: 0.96 ms at 24 threads.)  Since the solvers
    below do n_classes solves per iteration for up to `max_iter` iterations, the
    thread count — not the algorithm — dominated the runtime of every serial run.

    Parallel pool workers already get this cap via utils._call_star; this
    decorator makes it hold for *any* caller, in-process or not.  Nested limits
    are a no-op, and the cupy/GPU path is unaffected.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return fn(*args, **kwargs)
        with threadpool_limits(limits=1):
            return fn(*args, **kwargs)
    return wrapper


# ─── Shared primitives (used by both centralized and federated paths) ─────────

def damped_solve(H, g, mu_init: float = 1.0, factor: float = 10.0,
                 max_iter: int = 50, xp=np) -> tuple:
    """Damped Newton solve of (H + μI) δ = g with adaptive ridge μ.

    Increases μ by *factor* until the solve succeeds and the step is bounded
    (|δ|∞ < 1e2), then returns.  Falls back to the zero vector after *max_iter*
    attempts so the caller never hangs.

    Conditioning is judged by the solve itself — a singular system raises
    LinAlgError and an ill-conditioned one yields a blown-up δ that the |δ|∞
    bound rejects (NaN/Inf compare False, so μ is increased) — instead of an
    explicit np.linalg.cond() call.  cond() runs a full SVD (O(n³), ~10–20× a
    solve) on every class at every Newton iteration and was the dominant cost of
    the SL1 path; the magnitude bound is the safety net that actually mattered.

    *xp* selects the backend (numpy or cupy) so the solve can stay on the GPU.

    Returns (delta, mu_used).
    """
    mu  = mu_init
    n   = H.shape[0]
    eye = xp.eye(n)
    for _ in range(max_iter):
        try:
            delta = xp.linalg.solve(H + mu * eye, g)
            if float(xp.max(xp.abs(delta))) < 1e2:
                return delta, mu
        except np.linalg.LinAlgError:
            pass
        mu *= factor
    return xp.zeros_like(g), mu


def damped_newton_direction(hessian_blocks: list, grad, xp=np,
                            **kw):
    """Per-class damped Newton descent direction.

    D[:, k] = −(H_k + μI)⁻¹ g_k   for each output class k.

    Args:
        hessian_blocks: list of n_classes (units, units) Gauss-Newton blocks.
        grad:           (units, n_classes) aggregated gradient.
        xp:             backend module (numpy or cupy); blocks/grad must match.
        **kw:           forwarded to damped_solve (mu_init, factor, max_iter).

    Returns:
        direction: (units, n_classes) descent direction matrix.
    """
    direction = xp.zeros_like(grad)
    for k in range(grad.shape[1]):
        delta, _ = damped_solve(hessian_blocks[k], grad[:, k], xp=xp, **kw)
        direction[:, k] = -delta
    return direction


def armijo_line_search(loss_fn, w, direction, grad,
                       c: float = 1e-4, rho: float = 0.5,
                       max_iter: int = 20, xp=np) -> float:
    """Armijo backtracking line search.

    Returns the largest α in {1, ρ, ρ², …} satisfying the sufficient-decrease
    condition:
        f(w + α·d) ≤ f(w) + c·α·⟨grad, d⟩_F

    Args:
        loss_fn:   callable w -> scalar objective (sum over data or clients).
        w:         current weights, shape (units, n_classes).
        direction: descent direction d, same shape.
        grad:      gradient at w, same shape (for the directional derivative).
        c:         sufficient-decrease constant (default 1e-4).
        rho:       step shrinkage per backtrack (default 0.5).
        max_iter:  max backtracks before returning the smallest tested α.
        xp:        backend module (numpy or cupy) for w/direction/grad.  loss_fn
                   is expected to return a host float regardless.

    Returns:
        Scalar step size α ∈ (0, 1].
    """
    dir_deriv = float(xp.sum(grad * direction))
    if dir_deriv >= 0:
        # Not a descent direction (e.g. heavily regularised saddle); skip.
        return 1.0

    f0    = loss_fn(w)
    alpha = 1.0
    for _ in range(max_iter):
        if loss_fn(w + alpha * direction) <= f0 + c * alpha * dir_deriv:
            return alpha
        alpha *= rho
    return alpha   # return the smallest tested α rather than failing


# ─── Centralized continuation loop (Clr_Node, reg_type="sl1") ─────────────────

@blas_capped
def solve_newton_l1(X: np.ndarray, y_oh: np.ndarray, Wout0: np.ndarray,
                    reg_param: float, *,
                    alpha_init: float = 1.0, alpha_max: float = 5e6,
                    update1: float = 1.25, update2: float = 1.5,
                    thres: float = 1e-5, max_iter: int = 250,
                    opt_tol: float = 1e-6, prog_tol: float = 1e-9,
                    patience: int = 8, stag_tol: float = 2e-3,
                    verbose: bool = False, use_gpu: bool = False) -> np.ndarray:
    """Centralized Newton-path solver for cross-entropy + SmoothL1.

    Mirrors L1General's continuation strategy (L1GeneralUnconstrainedApx_sub)
    but uses the per-class block Gauss-Newton Hessian (funcs.ce_sl1_hess) so the
    same primitives serve the federated server.  Each iteration:

        1. evaluate gradient g and per-class Hessian blocks H at the current w
           under the current smoothing parameter α;
        2. build the damped Newton direction d (per class);
        3. Armijo line search on the current-α objective → step size t;
        4. w ← w + t·d;
        5. grow α (continuation): ×update2 if the full step was accepted, else
           ×update1, capped at alpha_max;
        6. stop once the optimality condition on the non-zero coordinates is
           below opt_tol *and* α has reached alpha_max.

    Args:
        X:         reservoir last states, shape (n_samples, units).
        y_oh:      one-hot labels, shape (n_samples, n_classes).
        Wout0:     initial readout weights, shape (units, n_classes).
        reg_param: SmoothL1 penalty weight λ.
        alpha_init/alpha_max/update1/update2: continuation schedule for α.
        thres:     hard-threshold magnitude; |w| < thres set to 0 (final + opt
                   condition support set).
        max_iter:  maximum Newton iterations.
        opt_tol:   optimality tolerance on non-zero coordinates.
        prog_tol:  minimum objective change before declaring no progress.
        patience:  stagnation early-stop — stop once the non-zero support is
                   unchanged for this many iterations after α reaches alpha_max
                   (0 disables, restoring run-to-max_iter). This is the genuine
                   convergence signal here: with per-iteration soft-thresholding
                   the summed-gradient opt_tol is unreachable (a band of weights
                   near |w|≈1/α keeps it non-zero), so without this the solver
                   always grinds the full max_iter rebuilding the Hessian.
        stag_tol:  max change in non-zero fraction across the patience window to
                   count as "frozen" (0.002 = 0.2 percentage points).
        use_gpu:   offload the per-iteration gradient/Hessian/loss evaluation to
                   the GPU via CuPy (resolved against availability; the per-class
                   linear solves stay on CPU since the blocks are already small).

    Returns:
        Trained Wout (units, n_classes) with |w| < thres zeroed.
    """
    from funcs import ce_sl1_fval, ce_sl1_grad, ce_sl1_hess, resolve_use_gpu, _get_xp

    # Resolve the GPU request once (CuPy/CUDA availability check happens here).
    gpu = resolve_use_gpu(use_gpu)
    xp  = _get_xp(gpu)

    # Upload the (constant) data once and keep the *entire* continuation loop on
    # the selected backend.  grad/Hessian are kept on the device (to_host=False),
    # the per-class solves and the soft-threshold/optimality math all run through
    # xp, so on GPU there is no host↔device round-trip per iteration — only the
    # scalar loss values and the final weights cross the boundary.
    X     = xp.asarray(X)
    y_oh  = xp.asarray(y_oh)
    w     = xp.asarray(np.array(Wout0, dtype=float))
    alpha = alpha_init
    f_old = ce_sl1_fval(w, X, y_oh, alpha, reg_param, use_cupy=gpu)   # host float

    # Sliding window of the non-zero fraction, for the stagnation early-stop.
    from collections import deque
    nz_window = deque(maxlen=patience) if patience else None

    for it in range(max_iter):
        g = ce_sl1_grad(w, X, y_oh, alpha, reg_param,
                        use_cupy=gpu, to_host=False)   # (units, n_cls) on device
        H = ce_sl1_hess(w, X, y_oh, alpha, reg_param,
                        use_cupy=gpu, to_host=False)   # list of (units, units) on device

        direction = damped_newton_direction(H, g, xp=xp)

        # Line search on the objective at the *current* smoothing parameter.
        t = armijo_line_search(
            lambda ww: ce_sl1_fval(ww, X, y_oh, alpha, reg_param, use_cupy=gpu),
            w, direction, g, xp=xp,
        )
        w = w + t * direction
        # Soft-threshold (L1 proximal operator) after every Newton step, so the
        # centralized solver applies the SAME operator per step as the FL server
        # (server.aggregate_parameters) rather than a one-off hard cut at the end.
        w = xp.sign(w) * xp.maximum(xp.abs(w) - thres, 0.0)

        # Continuation: grow α faster when the full Newton step was accepted.
        old_alpha = alpha
        alpha = min(alpha * (update2 if t == 1.0 else update1), alpha_max)

        # Optimality condition measured only on the active (non-zero) support,
        # matching L1General's convergence test.
        opt_cond = float(xp.sum(xp.abs(g[xp.abs(w) >= thres])))
        f_new    = ce_sl1_fval(w, X, y_oh, alpha, reg_param, use_cupy=gpu)

        if verbose and (it % 25 == 0):
            nz = int(xp.sum(xp.abs(w) >= thres))
            print(f"  newton it={it:4d}: f={f_new:.4f}  opt={opt_cond:.3e}  "
                  f"nonzero={nz}  alpha={old_alpha:.3e}")

        # Stagnation early-stop: the exact-zero support has frozen while α is
        # already at its ceiling — the real convergence signal for this
        # per-iteration-thresholded path (opt_cond below never reaches opt_tol).
        if nz_window is not None:
            nz_frac = float(xp.sum(xp.abs(w) >= thres)) / w.size
            nz_window.append(nz_frac)
            if (old_alpha >= alpha_max and len(nz_window) == nz_window.maxlen
                    and (max(nz_window) - min(nz_window)) < stag_tol):
                if verbose:
                    print(f"  newton support-stagnation stop at it={it}")
                break

        if opt_cond < opt_tol and old_alpha >= alpha_max:
            if verbose:
                print(f"  newton converged at it={it}")
            break
        if abs(f_new - f_old) < prog_tol and old_alpha >= alpha_max:
            break
        f_old = f_new

    # Per-iteration soft-thresholding already enforces exact zeros; the final
    # iterate is returned as-is (pulled back to host when run on the GPU).
    return w.get() if gpu else w


# ─── Centralized continuation loop (Reg_Node, reg_type="sl1") ─────────────────

@blas_capped
def solve_newton_mse_l1(X: np.ndarray, Y: np.ndarray, Wout0: np.ndarray,
                        reg_param: float, *,
                        alpha_init: float = 1.0, alpha_max: float = 1e6,
                        update1: float = 2.0, update2: float = 2.0,
                        thres: float = 1e-5, max_iter: int = 100,
                        opt_tol: float = 1e-6, prog_tol: float = 1e-9,
                        patience: int = 8, stag_tol: float = 2e-3,
                        verbose: bool = False) -> np.ndarray:
    """Centralized Newton-path solver for MSE + SmoothL1 (regression readout).

    The regression twin of :func:`solve_newton_l1`: same L1General-style α
    continuation and per-iteration soft-threshold, but with the least-squares
    objective (funcs.mse_sl1_*) instead of cross-entropy.  Each output column is
    an independent ridge-like system; the per-column Gauss-Newton Hessian
    (2·XᵀX + SmoothL1 diagonal) decouples the Newton step into one linear solve
    per column, so the same damped_newton_direction / armijo_line_search
    primitives serve both readouts.

    Unlike the L1General path this replaces, the *thres* argument is honoured:
    a soft-threshold (L1 proximal operator) is applied after every Newton step —
    the SAME operator the classification path and the FL server use — rather than
    L1General's one-off hard cut at a fixed 1e-4.

    CPU/numpy only: the MSE primitives (funcs.mse_sl1_*) have no CuPy path, and
    regression readouts are small enough that the per-iteration XᵀX rebuild is
    not the bottleneck the classification Hessian was.

    Args:
        X:         reservoir last states, shape (n_samples, units).
        Y:         targets, shape (n_samples, n_outputs).
        Wout0:     initial readout weights, shape (units, n_outputs).
        reg_param: SmoothL1 penalty weight λ.
        alpha_init/alpha_max/update1/update2: continuation schedule for α.
        thres:     soft-threshold magnitude applied every step; |w| shrunk by
                   thres and clipped at 0 (final support + opt condition set).
        max_iter:  maximum Newton iterations.
        opt_tol:   optimality tolerance on non-zero coordinates.
        prog_tol:  minimum objective change before declaring no progress.
        patience:  stagnation early-stop window (non-zero support frozen after α
                   reaches alpha_max); 0 disables. Same rationale as
                   solve_newton_l1: per-iteration thresholding keeps a band of
                   weights near |w|≈1/α non-zero, so the summed-gradient opt_tol
                   is effectively unreachable.
        stag_tol:  max change in non-zero fraction across the patience window to
                   count as "frozen".

    Returns:
        Trained Wout (units, n_outputs) with |w| < thres zeroed.
    """
    from funcs import mse_sl1_fval, mse_sl1_grad, mse_sl1_hess

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    w = np.array(Wout0, dtype=float)
    alpha = alpha_init
    f_old = mse_sl1_fval(w, X, Y, alpha, reg_param)

    from collections import deque
    nz_window = deque(maxlen=patience) if patience else None

    for it in range(max_iter):
        g = mse_sl1_grad(w, X, Y, alpha, reg_param)          # (units, n_out)
        H = mse_sl1_hess(w, X, alpha, reg_param)             # list of (units, units)

        direction = damped_newton_direction(H, g)

        t = armijo_line_search(
            lambda ww: mse_sl1_fval(ww, X, Y, alpha, reg_param),
            w, direction, g,
        )
        w = w + t * direction
        # Soft-threshold (L1 proximal operator) after every Newton step — the
        # same per-step operator the classification/FL paths apply, replacing
        # L1General's one-off hard cut so *thres* is honoured here.
        w = np.sign(w) * np.maximum(np.abs(w) - thres, 0.0)

        old_alpha = alpha
        alpha = min(alpha * (update2 if t == 1.0 else update1), alpha_max)

        opt_cond = float(np.sum(np.abs(g[np.abs(w) >= thres])))
        f_new    = mse_sl1_fval(w, X, Y, alpha, reg_param)

        if verbose and (it % 25 == 0):
            nz = int(np.sum(np.abs(w) >= thres))
            print(f"  newton-mse it={it:4d}: f={f_new:.4f}  opt={opt_cond:.3e}  "
                  f"nonzero={nz}  alpha={old_alpha:.3e}")

        # Stagnation early-stop: exact-zero support frozen while α is at its
        # ceiling — the reachable convergence signal for this thresholded path.
        if nz_window is not None:
            nz_frac = float(np.sum(np.abs(w) >= thres)) / w.size
            nz_window.append(nz_frac)
            if (old_alpha >= alpha_max and len(nz_window) == nz_window.maxlen
                    and (max(nz_window) - min(nz_window)) < stag_tol):
                if verbose:
                    print(f"  newton-mse support-stagnation stop at it={it}")
                break

        if opt_cond < opt_tol and old_alpha >= alpha_max:
            if verbose:
                print(f"  newton-mse converged at it={it}")
            break
        if abs(f_new - f_old) < prog_tol and old_alpha >= alpha_max:
            break
        f_old = f_new

    return w


# ─── Centralized Newton loop (Clr_Node, reg_type="l2" / "none") ───────────────

@blas_capped
def solve_newton_smooth(X: np.ndarray, y_oh: np.ndarray, Wout0: np.ndarray,
                        reg_param: float, reg_type: str, *,
                        max_iter: int = 100, opt_tol: float = 1e-6,
                        prog_tol: float = 1e-9,
                        patience: int = config.newton_smooth_defaults.PATIENCE,
                        rel_tol: float = config.newton_smooth_defaults.REL_TOL,
                        verbose: bool = False,
                        use_gpu: bool = False) -> np.ndarray:
    """Centralized damped-Newton solver for the *smooth* readout objectives.

    Handles reg_type in {"l2", "none"} — cross-entropy with an L2 penalty or no
    penalty.  Both are smooth and convex, so unlike solve_newton_l1 there is no
    α continuation and no per-step soft-threshold: each iteration is a plain
    damped Gauss-Newton step with an Armijo line search.

    Reuses the same per-class block machinery (damped_newton_direction,
    armijo_line_search) the FL server uses, so the centralized none/l2 path runs
    the SAME optimisation family as the federated path — removing the previous
    Adam-vs-Newton solver confound between the two settings.

    Stopping: the loop exits on any of (a) ‖g‖∞ < opt_tol, (b) |Δf| < prog_tol,
    or (c) a *relative*-objective plateau — ``patience`` consecutive iterations
    with (f_old − f_new)/max(|f_old|,1) < ``rel_tol``.  (c) is the one that
    actually fires in practice: on collinear ESN states ‖g‖∞ plateaus ~1e-4 and
    never reaches opt_tol, and the absolute prog_tol is likewise unreachable, so
    without it the solver grinds the full max_iter for a fit converged by ~iter 25.

    Args:
        X:         reservoir last states, shape (n_samples, units).
        y_oh:      one-hot labels, shape (n_samples, n_classes).
        Wout0:     initial readout weights, shape (units, n_classes).
        reg_param: L2 penalty weight λ (ignored when reg_type == "none").
        reg_type:  "l2" or "none".
        max_iter:  maximum Newton iterations.
        opt_tol:   stop once ‖g‖∞ falls below this.
        prog_tol:  stop once the absolute objective change falls below this.
        patience:  relative-plateau early-stop window (0 disables it, restoring
                   the run-to-max_iter behaviour).
        rel_tol:   relative objective-improvement floor for the plateau counter.
        use_gpu:   evaluate gradient/Hessian/loss via CuPy when available; the
                   small per-class linear solves stay on host (numpy).

    Returns:
        Trained Wout (units, n_classes).
    """
    if reg_type == "l2":
        from funcs import (ce_l2_fval as fval, ce_l2_grad as grad,
                           ce_l2_hess as hess, resolve_use_gpu)
    elif reg_type == "none":
        from funcs import (ce_none_fval as fval, ce_none_grad as grad,
                           ce_none_hess as hess, resolve_use_gpu)
    else:
        raise ValueError(f"solve_newton_smooth: unsupported reg_type '{reg_type}' "
                         f"(use solve_newton_l1 for sl1)")

    gpu = resolve_use_gpu(use_gpu)
    # grad/hess for l2/none return host (numpy) arrays, so the Newton solve runs
    # on numpy; GPU only accelerates the funcs' internal CE math.
    w     = np.array(Wout0, dtype=float)
    f_old = fval(w, X, y_oh, reg_param, use_cupy=gpu)
    stall = 0   # consecutive relative-plateau iterations, for the early-stop

    for it in range(max_iter):
        g = grad(w, X, y_oh, reg_param, use_cupy=gpu)        # (units, n_cls)
        H = hess(w, X, y_oh, reg_param, use_cupy=gpu)        # list of (units, units)

        # Start with a near-zero ridge so well-conditioned (e.g. L2-regularised)
        # Hessians take the *true* Newton step and converge quadratically; the
        # adaptive μ only grows if the solve is singular or the step blows up
        # (e.g. the separable, unregularised "none" case).
        direction = damped_newton_direction(H, g, xp=np, mu_init=1e-8)
        t = armijo_line_search(
            lambda ww: fval(ww, X, y_oh, reg_param, use_cupy=gpu),
            w, direction, g, xp=np,
        )
        w = w + t * direction

        opt_cond = float(np.max(np.abs(g)))
        f_new    = fval(w, X, y_oh, reg_param, use_cupy=gpu)

        if verbose and (it % 25 == 0):
            print(f"  newton-smooth it={it:4d}: f={f_new:.4f}  "
                  f"opt={opt_cond:.3e}  step={t:.3g}")

        if opt_cond < opt_tol:
            if verbose:
                print(f"  newton-smooth converged at it={it}")
            break
        if abs(f_new - f_old) < prog_tol:
            break

        # Relative-objective plateau early-stop: the reachable convergence signal
        # here (opt_tol/prog_tol above rarely fire on collinear ESN states). Count
        # consecutive iterations whose relative improvement is below rel_tol and
        # stop once that holds for `patience` in a row.
        if patience:
            rel_impr = (f_old - f_new) / max(abs(f_old), 1.0)
            stall = stall + 1 if rel_impr < rel_tol else 0
            if stall >= patience:
                if verbose:
                    print(f"  newton-smooth relative-plateau stop at it={it}")
                break

        f_old = f_new

    return w
