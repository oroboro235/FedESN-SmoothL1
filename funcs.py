# funcs.py — mathematical building blocks for ESN readout training.
#
# Public API (called by readout_node.py → Clr_Node.fit and Reg_Node.fit,
#             and by client.py → Client_TSC._compute_grad_hessian):
#
#   Cross-entropy + regularization (classification):
#     ce_sl1_fval  / ce_sl1_grad  / ce_sl1_hess   — CE + SmoothL1
#     ce_l2_fval   / ce_l2_grad   / ce_l2_hess    — CE + L2
#     ce_none_fval / ce_none_grad / ce_none_hess  — CE only
#
#   MSE + SmoothL1 (regression, CPU only):
#     mse_sl1_fval / mse_sl1_grad / mse_sl1_hess
#
# Hessian functions return a list of n_classes (units, units) arrays —
# the per-class block-diagonal Gauss-Newton Hessian (CE part is dense;
# regularisation part is diagonal, added in).
#
# All CE functions accept `use_cupy=True` to offload to GPU via CuPy.
# Pass use_cupy=False on machines without a CUDA GPU.
#
# GPU enablement is centralized here: every script exposes a `--gpu` flag that is
# resolved through resolve_use_gpu() — if CuPy/CUDA is not actually available the
# request is downgraded to CPU (with a one-time warning) instead of crashing, so
# the same command line works on GPU and CPU-only machines.

import warnings

import numpy as np


# ─── Backend selector ─────────────────────────────────────────────────────────

_gpu_warned = False   # ensure the CPU-fallback warning is printed at most once


def gpu_available() -> bool:
    """Return True iff CuPy is importable and at least one CUDA device is present.

    Cheap and exception-safe: a missing CuPy install, a CPU-only build, or a
    driver/runtime error all simply yield False.
    """
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_use_gpu(use_gpu: bool) -> bool:
    """Resolve a *requested* GPU flag against what the machine can actually do.

    Returns True only when GPU was requested AND a working CuPy/CUDA stack is
    available.  When GPU is requested but unavailable, emit a single warning and
    fall back to CPU.  This lets the entry scripts pass ``--gpu`` unconditionally
    without guarding for CuPy themselves.
    """
    global _gpu_warned
    if not use_gpu:
        return False
    if gpu_available():
        return True
    if not _gpu_warned:
        warnings.warn(
            "GPU acceleration was requested (--gpu / use_gpu=True) but CuPy with a "
            "CUDA device is not available; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        _gpu_warned = True
    return False


def _get_xp(use_cupy: bool = False):
    """Return cupy if *use_cupy* else numpy.

    Importing cupy is deferred to avoid crashing on CPU-only machines.  Callers
    must gate *use_cupy* through resolve_use_gpu() first (every entry script does):
    this function trusts the flag and lets a missing-CuPy ImportError surface
    loudly, so the whole public function stays on one consistent backend rather
    than silently mixing numpy arrays with cupy-only ``.get()`` calls.
    """
    if use_cupy:
        import cupy as cp
        return cp
    return np


# ─── Low-level primitives (private, accept xp as explicit namespace) ──────────

def _logsumexp(b, xp):
    """Numerically stable logsumexp across axis=1.

    Works with both numpy and cupy arrays; *xp* selects the backend.
    Equivalent to scipy.special.logsumexp(b, axis=1) but cupy-compatible.
    """
    B = xp.max(b, axis=1)
    # Tile B to match b's shape for stable subtraction
    repmat_B = xp.tile(B, (b.shape[1], 1)).T
    return xp.log(xp.sum(xp.exp(b - repmat_B), axis=1)) + B


def _softmax(z, xp):
    """Row-wise softmax; *xp* selects numpy or cupy."""
    exp_z = xp.exp(z - xp.max(z, axis=-1, keepdims=True))
    return exp_z / xp.sum(exp_z, axis=-1, keepdims=True)


def _cross_entropy_loss(w, X, y, xp, epsilon: float = 1e-12):
    """Sum cross-entropy loss.  y must be one-hot, w shape (features, classes)."""
    probs = _softmax(X @ w, xp) + epsilon
    return -xp.sum(y * xp.log(probs))


def _cross_entropy_gradient(w, X, y, xp):
    """Gradient of sum cross-entropy w.r.t. w.  Returns shape (features, classes)."""
    return X.T @ (_softmax(X @ w, xp) - y)


def _sl1_fval(w, alpha: float, _lambda: float, xp):
    """SmoothL1 regularisation value for weight matrix *w*."""
    n_feature, n_class = w.shape
    w_flat = w.reshape(-1, 1)
    zeros = xp.zeros((n_feature * n_class, 1))
    lse     = _logsumexp(xp.hstack([zeros,  alpha * w_flat]), xp)
    neg_lse = _logsumexp(xp.hstack([zeros, -alpha * w_flat]), xp)
    return (_lambda / alpha) * xp.sum(lse + neg_lse)


def _sl1_grad(w, alpha: float, _lambda: float, xp):
    """Gradient of SmoothL1 regularisation w.r.t. w.  Returns same shape as *w*."""
    n_feature, n_class = w.shape
    w_flat = w.reshape(-1, 1)
    zeros = xp.zeros((n_feature * n_class, 1))
    lse  = _logsumexp(xp.hstack([zeros, alpha * w_flat]), xp)
    grad = (_lambda * (1.0 - 2.0 * xp.exp(-lse))).reshape(-1, 1)
    return grad.reshape(n_feature, n_class)


def _l2_fval(w, _lambda: float, xp):
    """L2 regularisation value:  λ * ||w||²."""
    return _lambda * xp.sum(w ** 2)


def _l2_grad(w, _lambda: float, xp):
    """Gradient of L2 regularisation:  2λ * w."""
    return 2.0 * _lambda * w


def _cross_entropy_hessian(w, X, xp):
    """Per-class block-diagonal Gauss-Newton Hessian of cross-entropy (unnormalized).

    H_k = X^T diag(p_k · (1 - p_k)) X  for class k.

    Not divided by n, matching the _cross_entropy_gradient convention.
    Labels y are not needed (Hessian depends only on predictions P, not residuals).
    Returns a list of n_classes arrays each of shape (features, features).
    """
    P = _softmax(X @ w, xp)
    return [
        (X * (P[:, k] * (1.0 - P[:, k]))[:, None]).T @ X
        for k in range(w.shape[1])
    ]


def _sl1_hess_diag(w, alpha: float, _lambda: float, xp):
    """Element-wise diagonal of SmoothL1 Hessian: 2λα · σ(αw) · σ(−αw).

    Second derivative of (λ/α)[log(1+e^{αw}) + log(1+e^{−αw})] w.r.t. w.
    Returns an array of the same shape as w.
    """
    n_f, n_c = w.shape
    w_flat = w.reshape(-1, 1)
    zeros  = xp.zeros((n_f * n_c, 1))
    lse    = _logsumexp(xp.hstack([zeros, alpha * w_flat]), xp)  # log(1 + e^{αw})
    s      = xp.exp(alpha*w_flat.squeeze() - 2*lse)                                         # σ(−αw)
    return (2.0 * _lambda * alpha * s).reshape(n_f, n_c)


# ─── Public composite functions for classification (called from Clr_Node.fit) ─

def ce_sl1_fval(w, X, y, alpha: float, _lambda: float, use_cupy: bool = False):
    """Cross-entropy loss + SmoothL1 regularisation (scalar value)."""
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_loss(w, X, y, xp) + _sl1_fval(w, alpha, _lambda, xp)
    return float(result.get() if use_cupy else result)


def ce_sl1_grad(w, X, y, alpha: float, _lambda: float, use_cupy: bool = False,
                to_host: bool = True):
    """Gradient of CE + SmoothL1 w.r.t. w.  Returns same shape as *w*.

    With *use_cupy* and *to_host=False* the result is left on the GPU (no device→
    host copy), so a caller that keeps iterating on the GPU (e.g. the Newton
    continuation loop) avoids a round-trip every iteration.  When *use_cupy* is
    False the array is always plain numpy and *to_host* is irrelevant.
    """
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_gradient(w, X, y, xp) + _sl1_grad(w, alpha, _lambda, xp)
    if use_cupy and to_host:
        return result.get()
    return result


def ce_l2_fval(w, X, y, _lambda: float, use_cupy: bool = False):
    """Cross-entropy loss + L2 regularisation (scalar value)."""
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_loss(w, X, y, xp) + _l2_fval(w, _lambda, xp)
    return float(result.get() if use_cupy else result)


def ce_l2_grad(w, X, y, _lambda: float, use_cupy: bool = False):
    """Gradient of CE + L2 w.r.t. w.  Returns same shape as *w*."""
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_gradient(w, X, y, xp) + _l2_grad(w, _lambda, xp)
    return result.get() if use_cupy else result


def ce_none_fval(w, X, y, _lambda: float, use_cupy: bool = False):
    """Cross-entropy loss without regularisation (scalar value).

    The *_lambda* argument is accepted for API consistency but ignored.
    """
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_loss(w, X, y, xp)
    return float(result.get() if use_cupy else result)


def ce_none_grad(w, X, y, _lambda: float, use_cupy: bool = False):
    """Gradient of CE (no regularisation) w.r.t. w.  Returns same shape as *w*."""
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X, y = xp.asarray(w), xp.asarray(X), xp.asarray(y)
    result = _cross_entropy_gradient(w, X, y, xp)
    return result.get() if use_cupy else result


def ce_sl1_hess(w, X, _y, alpha: float, _lambda: float, use_cupy: bool = False,
                to_host: bool = True):
    """Per-class Hessians of CE + SmoothL1 loss.

    Returns list of n_classes (features, features) arrays.
    y is accepted for API symmetry with ce_sl1_grad but not used by the Hessian.
    Called from: client.py → _compute_grad_hessian when reg_type == 'sl1'.

    With *use_cupy* and *to_host=False* the per-class blocks stay on the GPU so
    the Newton solve can consume them without a device→host copy per iteration.
    """
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X = xp.asarray(w), xp.asarray(X)
    ce_h   = _cross_entropy_hessian(w, X, xp)           # list of (F, F)
    sl1_d  = _sl1_hess_diag(w, alpha, _lambda, xp)      # (F, n_cls)
    result = [ce_h[k] + xp.diag(sl1_d[:, k]) for k in range(w.shape[1])]
    if use_cupy and to_host:
        result = [h.get() for h in result]
    return result


def ce_l2_hess(w, X, _y, _lambda: float, use_cupy: bool = False):
    """Per-class Hessians of CE + L2 loss.

    Returns list of n_classes (features, features) arrays.
    y is accepted for API symmetry with ce_l2_grad but not used by the Hessian.
    Called from: client.py → _compute_grad_hessian when reg_type == 'l2'.
    """
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X = xp.asarray(w), xp.asarray(X)
    ce_h   = _cross_entropy_hessian(w, X, xp)           # list of (F, F)
    F      = w.shape[0]
    result = [ce_h[k] + 2.0 * _lambda * xp.eye(F) for k in range(w.shape[1])]
    if use_cupy:
        result = [h.get() for h in result]
    return result


def ce_none_hess(w, X, _y, _lambda: float, use_cupy: bool = False):
    """Per-class Hessians of CE (no regularisation).

    Returns list of n_classes (features, features) arrays.
    y and _lambda are accepted for API symmetry but not used.
    Called from: client.py → _compute_grad_hessian when reg_type == 'none'.
    """
    xp = _get_xp(use_cupy)
    if use_cupy:
        w, X = xp.asarray(w), xp.asarray(X)
    result = _cross_entropy_hessian(w, X, xp)
    if use_cupy:
        result = [h.get() for h in result]
    return result


# ─── MSE functions for regression (CPU only, called from Reg_Node.fit) ────────

def mse_fval(w, X, y):
    """Sum of squared errors:  ||Xw - y||²."""
    return np.sum((X @ w - y) ** 2)


def mse_grad(w, X, y):
    """Gradient of SSE w.r.t. w:  2 * Xᵀ(Xw - y)."""
    return 2.0 * X.T @ (X @ w - y)


def mse_sl1_fval(w, X, y, alpha: float, _lambda: float):
    """MSE + SmoothL1 value (regression, CPU only)."""
    return mse_fval(w, X, y) + _sl1_fval(w, alpha, _lambda, np)


def mse_sl1_grad(w, X, y, alpha: float, _lambda: float):
    """Gradient of MSE + SmoothL1 w.r.t. w (regression, CPU only)."""
    return mse_grad(w, X, y) + _sl1_grad(w, alpha, _lambda, np)


def mse_sl1_hess(w, X, alpha: float, _lambda: float):
    """Per-column Hessians of MSE + SmoothL1 (regression, CPU only).

    SSE Hessian is 2 * X^T X (shared across all output columns).
    SmoothL1 adds a per-element diagonal term per column.
    Returns list of n_classes (features, features) arrays.
    Called from: Reg_Node (if Newton-step training is added later).
    """
    mse_h = 2.0 * X.T @ X                           # (F, F) shared across columns
    sl1_d = _sl1_hess_diag(w, alpha, _lambda, np)   # (F, n_cls)
    return [mse_h + np.diag(sl1_d[:, k]) for k in range(w.shape[1])]














