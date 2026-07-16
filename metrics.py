# metrics.py — forecasting accuracy metrics for ESN regression evaluation.
#
# All metrics take (T, F) arrays (time steps × output channels); 1-D inputs are
# treated as a single channel.  Multi-channel results are reduced to one scalar
# (per-channel metric, then averaged) so they can be compared across the 1-D
# (mg) and 3-D (lorenz) datasets.
#
# Two layers:
#   • the individual metrics (rmse / nrmse / mase / directional_accuracy /
#     variance_ratio) and compute_all() — pure measurement, reusable by both the
#     search and the eval scripts.
#   • selection_loss() — maps any metric (or a scale-free weighted "composite")
#     to a single "lower is better" loss, which the search converts into the
#     higher-is-better trial score.

import numpy as np

_EPS = 1e-12

# Metric names accepted by selection_loss / the --select_metric CLI flag.
# "vpt" (valid prediction time) and "mse_short" (short-horizon MSE) are the
# chaos-aware metrics — only usable when the caller measured them on an
# auto-regressive forecast and added them to the metrics dict (see
# valid_prediction_time / short_horizon_errors).
SELECT_METRICS = ["rmse", "nrmse", "mase", "da", "var_ratio",
                  "vpt", "mse_short", "composite"]

# Default weights for the "composite" selection metric.  Only scale-free terms
# take part (rmse is excluded — it is scale-dependent and offered standalone).
COMPOSITE_DEFAULT_WEIGHTS = {"nrmse": 0.4, "mase": 0.3, "da": 0.2, "var_ratio": 0.1}


def _as2d(a: np.ndarray) -> np.ndarray:
    """Coerce to a float (T, F) array; a 1-D series becomes a single column."""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


# ─── Individual metrics ───────────────────────────────────────────────────────

def rmse(preds: np.ndarray, trues: np.ndarray) -> float:
    """Root mean squared error in raw units (lower is better)."""
    preds, trues = _as2d(preds), _as2d(trues)
    return float(np.sqrt(np.mean((preds - trues) ** 2)))


def nrmse(preds: np.ndarray, trues: np.ndarray) -> float:
    """RMSE normalised by the ground-truth std, per channel then averaged.

    Scale-free (lower is better): 0 = perfect, ≈1 = no better than predicting
    the channel mean.  Equivalent to sqrt(MSE / var(true)).
    """
    preds, trues = _as2d(preds), _as2d(trues)
    num = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
    den = np.where(trues.std(axis=0) < _EPS, _EPS, trues.std(axis=0))
    return float(np.mean(num / den))


def mase(preds: np.ndarray, trues: np.ndarray, train_target: np.ndarray) -> float:
    """Mean Absolute Scaled Error (lower is better; <1 beats the naive model).

    Forecast MAE divided by the in-sample naive one-step (random-walk) MAE,
    where the naive scale is computed on *train_target* (the training portion of
    the target series) to avoid normalising by the quantity being evaluated.
    Per channel, then averaged.
    """
    preds, trues, train_target = _as2d(preds), _as2d(trues), _as2d(train_target)
    mae   = np.mean(np.abs(preds - trues), axis=0)
    naive = np.mean(np.abs(np.diff(train_target, axis=0)), axis=0)
    naive = np.where(naive < _EPS, _EPS, naive)
    return float(np.mean(mae / naive))


def directional_accuracy(preds: np.ndarray, trues: np.ndarray,
                         anchor: np.ndarray) -> float:
    """Hit rate of the predicted direction of change (higher is better, [0, 1]).

    For each step compares sign(pred_t − true_{t-1}) with sign(true_t − true_{t-1}).
    *anchor* is the last known true value before the first forecast step
    (shape (F,)), used as true_{-1}.  Pooled over all steps and channels.
    """
    preds, trues = _as2d(preds), _as2d(trues)
    anchor = np.asarray(anchor, dtype=float).reshape(1, -1)
    prev = np.vstack([anchor, trues[:-1]])          # true value at t-1
    return float(np.mean(np.sign(trues - prev) == np.sign(preds - prev)))


def variance_ratio(preds: np.ndarray, trues: np.ndarray) -> float:
    """var(pred) / var(true), per channel then averaged (ideal = 1.0).

    <1 ⇒ under-dispersed (predictions decaying toward the mean — a common
    auto-regressive failure mode); >1 ⇒ over-dispersed / unstable.
    """
    preds, trues = _as2d(preds), _as2d(trues)
    vt = np.where(trues.var(axis=0) < _EPS, _EPS, trues.var(axis=0))
    return float(np.mean(preds.var(axis=0) / vt))


def compute_all(preds: np.ndarray, trues: np.ndarray,
                train_target: np.ndarray) -> dict:
    """All five metrics as a dict; *train_target* anchors MASE and DA."""
    return {
        "rmse":      rmse(preds, trues),
        "nrmse":     nrmse(preds, trues),
        "mase":      mase(preds, trues, train_target),
        "da":        directional_accuracy(preds, trues, _as2d(train_target)[-1]),
        "var_ratio": variance_ratio(preds, trues),
    }


# ─── Classification metrics (TSC / TSC-FL readouts) ───────────────────────────
# accuracy alone is misleading on the imbalanced datasets in the suite
# (ECG5000, DistalPhalanxOutlineCorrect, …): a classifier that ignores the
# minority class can still score high accuracy.  macro-F1 (unweighted mean of
# per-class F1) and balanced accuracy (unweighted mean of per-class recall) both
# weight every class equally, so they expose that failure.  Pure-numpy (no
# sklearn) so the FL client workers can call them without extra imports.

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int) -> np.ndarray:
    """Integer confusion matrix C[i, j] = #(true=i, pred=j), shape (C, C)."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray,
                          n_classes: int) -> dict:
    """accuracy / macro-F1 / balanced accuracy from integer labels (all in %).

    Per-class precision/recall with a 0/0 → 0 convention.  macro-F1 averages F1
    over ALL classes; balanced accuracy averages recall over classes that have
    support (rows with at least one true sample), matching sklearn.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)          # true count per class
    pred_cnt = cm.sum(axis=0).astype(float)         # predicted count per class

    with np.errstate(divide="ignore", invalid="ignore"):
        recall    = np.where(support  > 0, tp / support,  0.0)
        precision = np.where(pred_cnt > 0, tp / pred_cnt, 0.0)
        denom     = precision + recall
        f1        = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    total = cm.sum()
    acc          = float(tp.sum() / total * 100.0) if total > 0 else 0.0
    macro_f1     = float(f1.mean() * 100.0)
    present      = support > 0
    balanced_acc = float(recall[present].mean() * 100.0) if present.any() else 0.0
    return {"acc": acc, "macro_f1": macro_f1, "balanced_acc": balanced_acc}


# ─── Chaos-aware horizon metrics ──────────────────────────────────────────────
# Free-running auto-regression of a chaotic system (e.g. Lorenz) diverges from
# the true trajectory exponentially fast: even a perfect model decorrelates
# after a few Lyapunov times, after which full-horizon MSE/RMSE saturate at the
# climatological variance (≈ 2·σ²) regardless of model quality.  The two metrics
# below instead measure skill *inside* the predictable window.

def valid_prediction_time(preds: np.ndarray, trues: np.ndarray,
                          threshold: float = 0.4, norm: str = "std") -> int:
    """Valid prediction time (VPT): steps tracked before divergence (higher is better).

    The first step index at which the normalized forecast error exceeds
    *threshold*; if it never does, the full horizon length is returned.  This is
    the standard chaotic-forecast skill measure (cf. Pathak et al. 2018).

    Normalized error at step t is
        e(t) = ||pred(t) − true(t)|| / D
    with D the RMS magnitude of the true trajectory about its mean
    (norm="std", == sqrt of the summed-channel variance; e ≈ √2 ⇒ fully
    decorrelated) or the RMS of the raw true trajectory (norm="raw").
    """
    preds, trues = _as2d(preds), _as2d(trues)
    if norm == "raw":
        D = np.sqrt(np.mean(np.sum(trues ** 2, axis=1)))
    else:
        D = np.sqrt(np.mean(np.sum((trues - trues.mean(axis=0)) ** 2, axis=1)))
    D = D if D > _EPS else _EPS
    err  = np.sqrt(np.sum((preds - trues) ** 2, axis=1)) / D
    over = np.where(err > threshold)[0]
    return int(over[0]) if over.size else int(len(trues))


def short_horizon_errors(preds: np.ndarray, trues: np.ndarray,
                         horizons) -> dict:
    """MSE and RMSE over the first H steps, for each H in *horizons*.

    Errors confined to the predictable window, before chaotic divergence
    inflates the full-horizon error to the climatological floor.  Returns
    {"mse_<H>": ..., "rmse_<H>": ...} for every H that fits the horizon.
    """
    preds, trues = _as2d(preds), _as2d(trues)
    n   = len(trues)
    out = {}
    for h in horizons:
        if 0 < h <= n:
            seg = (preds[:h] - trues[:h]) ** 2
            out[f"mse_{h}"]  = float(np.mean(seg))
            out[f"rmse_{h}"] = float(np.sqrt(np.mean(seg)))
    return out


# ─── Selection loss (lower is better) ─────────────────────────────────────────

def _losses(m: dict) -> dict:
    """Map a metrics dict to "lower is better" loss terms.

    rmse/nrmse/mase already decrease with quality; DA is inverted (1 − DA) and
    the variance ratio becomes |log(VR)| so VR = 0.5 and VR = 2 are penalised
    equally and the ideal VR = 1 gives 0.
    """
    losses = {
        "rmse":      m["rmse"],
        "nrmse":     m["nrmse"],
        "mase":      m["mase"],
        "da":        1.0 - m["da"],
        "var_ratio": abs(np.log(max(m["var_ratio"], _EPS))),
    }
    # Chaos-aware terms, present only when the caller measured them on an
    # auto-regressive forecast.  VPT is higher-is-better, so its loss is negated
    # (more tracked steps ⇒ lower loss); mse_short is already lower-is-better.
    if "vpt" in m:
        # Prefer the horizon-normalized vpt (∈[0,1], 1 = tracked the whole
        # horizon) so that in the sparsity blend
        #   score = -(1-w)*loss + w*sparsity/100
        # the accuracy term and the sparsity term share the same [0,1] scale and
        # sparsity_weight is a genuine 0–1 trade-off.  The normaliser (horizon)
        # is constant across trials, so this is a pure rescaling — it does NOT
        # change vpt-only rankings (sparsity_weight=0), only the blended score.
        # Falls back to the raw step count if vpt_norm was not provided.
        losses["vpt"] = -float(m.get("vpt_norm", m["vpt"]))
    if "mse_short" in m:
        losses["mse_short"] = float(m["mse_short"])
    return losses


def composite_loss(m: dict, weights: dict = None) -> float:
    """Weighted sum of the scale-free loss terms (nrmse, mase, 1−DA, |log VR|).

    Weights need not sum to 1 — they are renormalised here.  rmse is not part of
    the composite (scale-dependent); request it explicitly via select_metric.
    """
    weights = COMPOSITE_DEFAULT_WEIGHTS if weights is None else weights
    losses  = _losses(m)
    total_w = sum(weights.values()) or 1.0
    return sum(w * losses[k] for k, w in weights.items()) / total_w


def selection_loss(metric: str, m: dict, composite_weights: dict = None) -> float:
    """Single "lower is better" loss for the requested selection *metric*."""
    if metric == "composite":
        return composite_loss(m, composite_weights)
    losses = _losses(m)
    if metric not in losses:
        raise ValueError(f"Unknown select_metric '{metric}'; "
                         f"choose from {SELECT_METRICS}")
    return losses[metric]
