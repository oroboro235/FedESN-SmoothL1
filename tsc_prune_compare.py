# tsc_prune_compare.py — does the SL1 penalty beat plain L2 + magnitude pruning?
#
# Question (the one that decides whether SL1 is worth keeping for TSC):
#   The readout is sparsified by a soft-threshold operator
#       prune_t(w) = sign(w) · max(|w| − t, 0)
#   In the SL1 path that operator is applied *during* training (every Newton
#   step), on top of a SmoothL1 penalty that actively pushes weights toward zero.
#   The obvious cheaper alternative is: train a plain ridge (CE + L2) readout and
#   apply the SAME soft-threshold operator *post-hoc* to magnitude-prune it.
#
#   So — at a MATCHED sparsity level — does the SL1 penalty arrange the weights so
#   that pruning keeps the "right" ones (higher accuracy), or does a dense L2 fit
#   pruned to the same sparsity do just as well?  If L2+prune matches SL1, the L1
#   penalty (and its slow Newton continuation) is not earning its cost.
#
# This is the SL1-vs-L2 counterpart of the SL1-vs-(CE+IMP) study in DEVLOG.md.
# The operator (soft-threshold) is identical in both arms here, so the ONLY
# variable is the training-time regulariser (L2 vs SmoothL1).
#
# Fair-comparison design (one variable = the regulariser):
#   • ONE reservoir per (dataset, seed) — both arms are trained on the SAME
#     extracted last-states, so any gap is the regulariser, not the reservoir.
#   • L2 arm : two selection modes, --l2_select.
#       dense (default, WEAK): tune λ on validation for best DENSE accuracy, refit
#         on train+val, then sweep the soft-threshold t.  On collinear reservoir
#         states this lands near OLS, whose compensating weight pairs magnitude
#         thresholding shatters — so the SL1 gap it reports is an UPPER BOUND.
#       prune (STRONG, use for reported results): re-select λ at EVERY sparsity
#         level by POST-PRUNE validation accuracy (an upper envelope over λ).  This
#         is the honest "best L2+prune vs SL1" question.  Note it grants the L2 arm
#         two knobs (λ per level, plus the threshold) while SL1 keeps one (thres,
#         with λ fixed at its accuracy-selected value) — deliberately generous to
#         the baseline.
#   • SL1 arm: fix λ (SmoothL1's λ is a poor sparsity knob — see DEVLOG; thres is
#     the real one), sweep `thres` → each value trains one readout whose
#     per-Newton-step soft-threshold yields an (sparsity, test_acc) point.
#   • Both curves are interpolated onto a common sparsity grid; we report
#     Δacc = SL1 − L2 at each level and the crossover sparsity.
#   • Averaged over reservoir seeds.
#
# Results:
#   ./result/tsc_prune_compare.csv       — one row per (dataset, method, seed, knob)
#   ./result/tsc_prune_compare_summary.csv — Δ(SL1−L2) on the common sparsity grid
#   ./result/pic/tsc_prune_<dataset>.png — accuracy-vs-sparsity curve (if --plot)
#
# Usage:
#   python tsc_prune_compare.py
#   python tsc_prune_compare.py --datasets jpv --plot
#   # the reported (fair) run — strong L2 baseline, all datasets, 3 seeds:
#   python tsc_prune_compare.py --l2_select prune --l2_refit --seeds 1 2 3 \
#       --n_jobs -1 --no_gpu --plot

import argparse
import json
import os
import time
import warnings

import numpy as np
from reservoirpy.nodes import Reservoir

import config
from data_loader import one_hot, standardize
from readout_node import Clr_Node
from utils import init_csv, append_csv_row, parallel_map
from tsc_centralized_search import (
    _load_dataset_meta, _load_dataset, _stratified_split,
    _extract_states, _ensure_3d,
)


# ─── Reservoir hyperparameters (shared by both arms per dataset) ───────────────

def _load_reservoir_hparams(dataset: str, source: str) -> dict:
    """Per-dataset reservoir hyperparameters from configs/TSC_settings_<source>.json.

    Both arms use the SAME reservoir per (dataset, seed) so the comparison
    isolates the readout regulariser.  We just need a sensible, per-dataset
    operating point; `source` (default "sl1") picks which tuned config supplies
    it.  Falls back to generic defaults for anything not in the file.
    """
    defaults = {"units": 500, "sr": 0.9, "lr": 0.3, "input_scaling": 1.0}
    s = config.load_settings("tsc", source).get(dataset)
    if s is not None:
        return {
            "units":         s.get("units", 500),
            "sr":            s.get("sr", 0.9),
            "lr":            s.get("lr", 0.3),
            "input_scaling": s.get("input_scaling", 1.0),
        }
    return defaults


# ─── Soft-threshold pruning primitive (identical operator for both arms) ───────

def _soft_threshold(W: np.ndarray, t: float) -> np.ndarray:
    """Proximal-L1 soft-threshold: sign(w)·max(|w|−t, 0).  t=0 is a no-op."""
    return np.sign(W) * np.maximum(np.abs(W) - t, 0.0)


def _sparsity(W: np.ndarray) -> float:
    """Percentage of exactly-zero weights."""
    return float((W == 0).mean() * 100.0)


def _acc(W: np.ndarray, X: np.ndarray, y_int: np.ndarray) -> float:
    return float((np.argmax(X @ W, axis=1) == y_int).mean() * 100.0)


# ─── L2 arm: dense ridge fit + post-hoc soft-threshold sweep ───────────────────

def _fit_l2(S: np.ndarray, y_oh: np.ndarray, reg_param: float,
            epochs: int, use_gpu: bool) -> np.ndarray:
    """Dense CE + L2 readout via the shared Gauss-Newton smooth solver."""
    node = Clr_Node(reg_param=reg_param, reg_type="l2", solver="newton",
                    epochs=epochs, use_gpu=use_gpu)
    return node.fit_from_states(S, y_oh)


def _l2_refit_support(S: np.ndarray, y_oh: np.ndarray, mask: np.ndarray,
                      reg_param: float, epochs: int, use_gpu: bool) -> np.ndarray:
    """Refit a ridge readout on the surviving support only (IMP-style).

    `mask` is the (units, n_classes) boolean keep-pattern from the prune step.
    Refitting the survivors is the difference between "prune once" and iterative
    magnitude pruning — a strictly stronger L2 baseline.  Each class refits over
    its own kept rows (the reservoir columns whose weight survived for that class).

    Each class is refit as a genuine ONE-VS-REST problem with a two-column target
    [1-y_k, y_k], and the score is the logit difference.  Passing the single column
    y_oh[:, k:k+1] instead would be degenerate: a softmax over one logit is
    identically 1, so the cross-entropy is constant, the gradient collapses to
    X^T(1-y), and the "refit" returns near-zero weights (|w| ~ 1e-4) that merely
    encode a class-conditional mean.  That silently *weakened* the arm it was
    supposed to strengthen.
    """
    W = np.zeros_like(mask, dtype=float)
    for k in range(y_oh.shape[1]):
        keep = np.where(mask[:, k])[0]
        if keep.size == 0:
            continue
        y_bin = np.column_stack([1.0 - y_oh[:, k], y_oh[:, k]])
        sub = _fit_l2(S[:, keep], y_bin, reg_param, epochs, use_gpu)
        W[keep, k] = sub[:, 1] - sub[:, 0]
    return W


def _l2_prune_curve(S_fit, y_fit_oh, S_te, y_te_int, reg_param, thresholds,
                    epochs, use_gpu, refit):
    """Train one dense ridge, then sweep the soft-threshold → curve points.

    thresholds are magnitudes; here they are chosen per-fit from the |w|
    quantiles (see _adaptive_thresholds) so the sweep spans a useful sparsity
    range regardless of the ridge's weight scale.  Returns list of dicts
    {knob, sparsity, test_acc}.  With refit=True each pruned support is refit.

    This is the `--l2_select dense` arm: one λ, chosen for DENSE accuracy.  On
    collinear reservoir states that lands near OLS, whose large compensating
    weight pairs magnitude thresholding shatters — so read it as a WEAK baseline
    (an upper bound on SL1's advantage).  See _l2_prune_curve_select.
    """
    W_dense = _fit_l2(S_fit, y_fit_oh, reg_param, epochs, use_gpu)
    pts = []
    for t in thresholds:
        Wp = _soft_threshold(W_dense, t)
        if refit and t > 0:
            Wp = _l2_refit_support(S_fit, y_fit_oh, Wp != 0, reg_param,
                                   epochs, use_gpu)
        pts.append({"knob": t, "sparsity": _sparsity(Wp),
                    "test_acc": _acc(Wp, S_te, y_te_int),
                    "reg_param": reg_param})
    return pts


def _l2_prune_curve_select(S_tr, y_tr_oh, S_val, yval_int, S_fit, y_fit_oh,
                           S_te, y_te_int, reg_grid, sparsity_targets,
                           epochs, use_gpu, refit):
    """Strong L2 baseline: re-select λ at EVERY sparsity level (`--l2_select prune`).

    The `dense` arm above tunes λ once, for dense accuracy, which under-regularises
    on collinear states and makes the subsequent prune collapse.  The fair question
    is "best L2+prune vs SL1", so here λ is chosen per sparsity level q by
    POST-PRUNE validation accuracy — an upper envelope over λ:

        for each q:  λ*(q) = argmax_λ  val_acc( prune_q( fit_λ(S_tr) ) )
                     then refit at λ*(q) on train+val, prune to q, score on test.

    Because a soft-threshold zeroes |w| ≤ t, the t hitting a target zero-fraction q
    is the q-quantile of |w| (_adaptive_thresholds) — so every λ can be driven to
    the SAME sparsity, which is what makes a per-q λ selection well defined.

    Selection uses train-only fits scored on validation; the reported number comes
    from a train+val refit scored on test.  No test information enters selection.

    Selection MUST run the same prune(+refit) pipeline it is selecting for.  An
    earlier version skipped the refit during selection to save sub-fits; that is
    invalid whenever the prune-once curve is degenerate, which is exactly what
    happens on the binary datasets: the two readout columns of a 2-class softmax
    are exact mirror images (corr = −1), so the soft-threshold preserves the
    antisymmetry while destroying the decision offset (the readout carries no
    bias term), and every sample collapses onto one class — accuracy pinned at the
    minority-class rate regardless of λ.  Selecting λ on *those* numbers is
    selecting on noise.  With the BLAS fix a refit costs ~0.1 s, so the saving was
    worthless anyway.
    """
    W_sel = {rp: _fit_l2(S_tr,  y_tr_oh,  rp, epochs, use_gpu) for rp in reg_grid}
    W_fin = {rp: _fit_l2(S_fit, y_fit_oh, rp, epochs, use_gpu) for rp in reg_grid}

    def _prune_at(W, S, y_oh, rp, q):
        """Threshold W to zero-fraction q, refitting the survivors if requested."""
        t  = float(_adaptive_thresholds(W, [q])[0])
        Wp = _soft_threshold(W, t)
        if refit and q > 0:
            Wp = _l2_refit_support(S, y_oh, Wp != 0, rp, epochs, use_gpu)
        return t, Wp

    pts = []
    for q in sparsity_targets:
        best_rp, best_va = None, -1.0
        for rp in reg_grid:
            _, Wp = _prune_at(W_sel[rp], S_tr, y_tr_oh, rp, q)
            va = _acc(Wp, S_val, yval_int)
            if va > best_va:
                best_rp, best_va = rp, va

        t, Wp = _prune_at(W_fin[best_rp], S_fit, y_fit_oh, best_rp, q)
        pts.append({"knob": t, "sparsity": _sparsity(Wp),
                    "test_acc": _acc(Wp, S_te, y_te_int),
                    "reg_param": best_rp, "val_acc": best_va})
    return pts


def _adaptive_thresholds(W: np.ndarray, sparsity_targets) -> np.ndarray:
    """Soft-threshold magnitudes hitting the requested sparsity fractions.

    For a soft-threshold, a weight is zeroed iff |w| ≤ t, so the t that zeroes a
    target fraction q of the weights is the q-quantile of |w|.  Building the
    sweep from the actual weight distribution (rather than a fixed t grid) makes
    the L2 curve span the same sparsity band as the SL1 curve for any λ scale.
    """
    a = np.abs(W).ravel()
    return np.quantile(a, np.asarray(sparsity_targets, dtype=float))


# ─── SL1 arm: SmoothL1 Newton fit with a soft-threshold sweep ──────────────────

def _fit_sl1(S, y_oh, reg_param, thres, epochs, use_gpu):
    """CE + SmoothL1 readout; the per-Newton-step soft-threshold uses `thres`."""
    node = Clr_Node(
        reg_param        = reg_param,
        reg_type         = "sl1",
        solver           = "newton",
        thres            = thres,
        alpha_init       = config.sl1_defaults.ALPHA_INIT,
        alpha_multiplier = config.sl1_defaults.ALPHA_MULTIPLIER,
        patience         = config.sl1_defaults.PATIENCE,
        stag_tol         = config.sl1_defaults.STAG_TOL,
        epochs           = epochs,
        use_gpu          = use_gpu,
    )
    return node.fit_from_states(S, y_oh)


def _sl1_prune_curve(S_fit, y_fit_oh, S_te, y_te_int, reg_param, thres_grid,
                     epochs, use_gpu, refit=False):
    """Sweep `thres`; each value is a full SL1 fit → one curve point.

    With refit=True, the surviving support is refit once with a plain ridge
    (same `_l2_refit_support` machinery as the L2 arm) before scoring.  This is
    the standard prune-then-refit debiasing: the per-step soft-threshold shrinks
    every SURVIVING weight by `thres` too (sign(w)·max(|w|−t,0)), a bias that
    grows with the threshold and that the L2 arm's refit removes but the raw SL1
    readout carries.  Refitting makes the two arms symmetric, so the comparison
    isolates WHICH support was selected rather than mixing in the τ-shrinkage
    bias.  The support itself is untouched — sparsity is identical either way.
    """
    pts = []
    for t in thres_grid:
        W = _fit_sl1(S_fit, y_fit_oh, reg_param, t, epochs, use_gpu)
        # Refit only where the threshold actually bit: the shrinkage bias on
        # surviving weights is ∝ t, so on a near-dense readout (tiny t) there is
        # nothing to debias — and refitting a ~full support is a full one-vs-rest
        # ridge per class, the most expensive point of the sweep for no gain.
        if refit and _sparsity(W) >= 10.0:
            W = _l2_refit_support(S_fit, y_fit_oh, W != 0, reg_param,
                                  epochs, use_gpu)
        pts.append({"knob": t, "sparsity": _sparsity(W),
                    "test_acc": _acc(W, S_te, y_te_int)})
    return pts


# ─── Curve interpolation onto a common sparsity grid ───────────────────────────

def _interp_curve(spars, accs, grid):
    """Interpolate acc(sparsity) onto `grid`, over the curve's own span only.

    Points are sorted by sparsity and de-duplicated (accuracy averaged over ties)
    so np.interp gets a monotone x.  Grid points outside the measured span are
    returned as NaN (no extrapolation — we only compare where both arms have data).
    """
    spars = np.asarray(spars, dtype=float)
    accs  = np.asarray(accs,  dtype=float)
    order = np.argsort(spars)
    spars, accs = spars[order], accs[order]
    uniq, inv = np.unique(spars, return_inverse=True)
    mean_acc = np.array([accs[inv == i].mean() for i in range(len(uniq))])
    grid = np.asarray(grid, dtype=float)
    out = np.interp(grid, uniq, mean_acc, left=np.nan, right=np.nan)
    return out


# ─── Per-dataset comparison ────────────────────────────────────────────────────

_CSV_FIELDS = [
    "dataset", "method", "seed", "select", "knob", "reg_param",
    "sparsity", "test_acc", "fit_secs",
]

_SUMMARY_FIELDS = [
    "dataset", "target_sparsity", "sl1_acc", "l2_acc", "delta_sl1_minus_l2", "n_seeds",
]


def _prune_worker(dataset, seed, n_classes, rc, sl1_lam, l2_reg_grid, thres_grid,
                  l2_sparsity_targets, epochs, use_gpu, l2_refit, l2_select,
                  sl1_refit, use_cache):
    """One (dataset, seed) unit — the parallel unit.  Both arms, one reservoir.

    Loads its own dataset from the npz cache rather than receiving the arrays in
    the task tuple: keeps the pickled payload tiny and works under spawn (Windows)
    as well as fork.  The reservoir and the train/val split are built from `seed`,
    so both arms see byte-identical states and any gap is the regulariser alone.
    """
    Xtr_raw, ytr_int, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
    Xtr_raw, Xte_raw = _ensure_3d(Xtr_raw), _ensure_3d(Xte_raw)

    Xtr_sub, ytr_sub_int, Xval, yval_int = _stratified_split(
        Xtr_raw, ytr_int, val_frac=0.2, seed=seed)
    Xtr_sub, Xval, Xte = standardize(Xtr_sub, Xval, Xte_raw)

    reservoir = Reservoir(units=rc["units"], sr=rc["sr"], lr=rc["lr"],
                          input_scaling=rc["input_scaling"], seed=seed)
    S_tr  = _extract_states(Xtr_sub, reservoir)
    S_val = _extract_states(Xval,    reservoir)
    S_te  = _extract_states(Xte,     reservoir)

    S_fit    = np.vstack([S_tr, S_val])
    yfit_int = np.concatenate([ytr_sub_int, yval_int])
    y_fit_oh = one_hot(yfit_int,    n_classes)
    y_tr_oh  = one_hot(ytr_sub_int, n_classes)

    # ── L2 arm ────────────────────────────────────────────────────────────────
    t0 = time.time()
    if l2_select == "prune":
        # Strong baseline: λ re-selected per sparsity level by post-prune val acc.
        l2_pts = _l2_prune_curve_select(
            S_tr, y_tr_oh, S_val, yval_int, S_fit, y_fit_oh, S_te, yte_int,
            l2_reg_grid, l2_sparsity_targets, epochs, use_gpu, l2_refit)
        l2_desc = "λ*∈{" + ",".join(f"{p['reg_param']:g}" for p in l2_pts) + "}"
    else:
        # Weak baseline: one λ, tuned for DENSE val accuracy.
        best_l2 = (-1.0, None)
        for rp in l2_reg_grid:
            Wd = _fit_l2(S_tr, y_tr_oh, rp, epochs, use_gpu)
            va = _acc(Wd, S_val, yval_int)
            if va > best_l2[0]:
                best_l2 = (va, rp)
        l2_rp   = best_l2[1]
        W_dense = _fit_l2(S_fit, y_fit_oh, l2_rp, epochs, use_gpu)
        l2_thr  = _adaptive_thresholds(W_dense, l2_sparsity_targets)
        l2_pts  = _l2_prune_curve(S_fit, y_fit_oh, S_te, yte_int, l2_rp,
                                  l2_thr, epochs, use_gpu, l2_refit)
        l2_desc = f"λ={l2_rp:g}, dense val={best_l2[0]:.1f}%"
    l2_secs = time.time() - t0

    # ── SL1 arm: fixed λ, sweep thres ─────────────────────────────────────────
    t0 = time.time()
    sl1_pts = _sl1_prune_curve(S_fit, y_fit_oh, S_te, yte_int, sl1_lam,
                               thres_grid, epochs, use_gpu, refit=sl1_refit)
    sl1_secs = time.time() - t0

    return {"dataset": dataset, "seed": seed, "sl1_lam": sl1_lam,
            "l2_pts": l2_pts, "sl1_pts": sl1_pts, "l2_desc": l2_desc,
            "l2_secs": l2_secs, "sl1_secs": sl1_secs}


def compare(datasets, seeds, reservoir_source, l2_reg_grid, sl1_reg_param,
            thres_grid, l2_sparsity_targets, sparsity_grid, epochs,
            csv_path, summary_path, use_cache, use_gpu, l2_refit, l2_select,
            sl1_refit, n_jobs, plot):
    meta = _load_dataset_meta()
    init_csv(csv_path, _CSV_FIELDS)
    init_csv(summary_path, _SUMMARY_FIELDS)

    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"(one CUDA context per process).")
        n_jobs = 1

    # Fan out over (dataset, seed) — every unit is independent.
    known = []
    for d in datasets:
        if d in meta:
            known.append(d)
        else:
            print(f"[SKIP] '{d}' not in {config.paths.tsc_dataset_meta}")
    datasets = known

    tasks = []
    for dataset in datasets:
        n_classes = meta[dataset]["output_dim"]
        rc = _load_reservoir_hparams(dataset, reservoir_source)
        sl1_lam = (sl1_reg_param if sl1_reg_param is not None
                   else _sl1_default_lambda(dataset))
        print(f"{dataset:14s} classes={n_classes:2d}  units={rc['units']}  "
              f"sr={rc['sr']:.3g}  lr={rc['lr']:.3g}  "
              f"in_scale={rc['input_scaling']:.3g}  SL1 λ={sl1_lam:.3g}")
        for seed in seeds:
            tasks.append((dataset, seed, n_classes, rc, sl1_lam, l2_reg_grid,
                          thres_grid, l2_sparsity_targets, epochs, use_gpu,
                          l2_refit, l2_select, sl1_refit, use_cache))

    print(f"\nL2 λ-grid={l2_reg_grid}   thres-grid={thres_grid}   "
          f"l2_select={l2_select}   l2_refit={l2_refit}")
    print(f"{len(datasets)} dataset(s) × {len(seeds)} seed(s) = {len(tasks)} "
          f"units  (n_jobs={n_jobs})\n")

    results = parallel_map(_prune_worker, tasks, n_jobs, progress="prune-compare")

    # ── Collect in submit order → deterministic CSV regardless of completion ──
    curves = {d: {"sl1": {}, "l2": {}} for d in datasets}
    for r in results:
        d, seed = r["dataset"], r["seed"]
        curves[d]["l2"][seed]  = ([p["sparsity"] for p in r["l2_pts"]],
                                  [p["test_acc"] for p in r["l2_pts"]])
        curves[d]["sl1"][seed] = ([p["sparsity"] for p in r["sl1_pts"]],
                                  [p["test_acc"] for p in r["sl1_pts"]])

        for method, pts, rp, secs, sel in (
            ("l2",  r["l2_pts"],  None,          r["l2_secs"],  l2_select),
            ("sl1", r["sl1_pts"], r["sl1_lam"],  r["sl1_secs"], "-"),
        ):
            for p in pts:
                append_csv_row(csv_path, _CSV_FIELDS, {
                    "dataset": d, "method": method, "seed": seed,
                    "select": sel, "knob": f"{p['knob']:.6g}",
                    "reg_param": f"{p.get('reg_param', rp):.6g}",
                    "sparsity": f"{p['sparsity']:.3f}",
                    "test_acc": f"{p['test_acc']:.4f}",
                    "fit_secs": f"{secs:.3f}",
                })

        print(f"  {d:14s} seed={seed}  L2[{l2_select}] ({r['l2_desc']}): "
              f"sparsity {_span(r['l2_pts'])}  [{r['l2_secs']:.1f}s]   |   "
              f"SL1: sparsity {_span(r['sl1_pts'])}  [{r['sl1_secs']:.1f}s]")

    for dataset in datasets:
        print(f"\n{'='*72}\nDataset: {dataset}\n{'='*72}")
        _report_dataset(dataset, curves[dataset], sparsity_grid, seeds,
                        summary_path)
        if plot:
            _plot_dataset(dataset, curves[dataset], seeds)

    print(f"\nPer-point rows → {csv_path}\nSummary grid → {summary_path}")


def _span(pts):
    s = [p["sparsity"] for p in pts]
    return f"{min(s):.1f}–{max(s):.1f}%"


def _sl1_default_lambda(dataset: str) -> float:
    """SL1 λ from the tuned sl1 config if present, else 1.0 (DEVLOG: λ≈1)."""
    s = config.load_settings("tsc", "sl1").get(dataset)
    if s is not None and "reg_param" in s:
        return float(s["reg_param"])
    return 1.0


# ─── Reporting ─────────────────────────────────────────────────────────────────

def _report_dataset(dataset, ds_curves, grid, seeds, summary_path):
    """Interpolate both arms per seed onto the common grid, average, print Δ."""
    sl1_mat, l2_mat = [], []
    for seed in seeds:
        if seed in ds_curves["sl1"]:
            sl1_mat.append(_interp_curve(*ds_curves["sl1"][seed], grid))
        if seed in ds_curves["l2"]:
            l2_mat.append(_interp_curve(*ds_curves["l2"][seed], grid))
    if not sl1_mat or not l2_mat:
        return
    # Columns outside a curve's measured span are NaN; a grid level neither arm
    # reached is an all-NaN column → nanmean would warn. Silence just that case;
    # the print/CSV loop below already skips any level that stayed NaN.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sl1_mean = np.nanmean(np.vstack(sl1_mat), axis=0)
        l2_mean  = np.nanmean(np.vstack(l2_mat),  axis=0)

    print(f"\n  {'sparsity':>9s} {'SL1 acc':>9s} {'L2 acc':>9s} "
          f"{'Δ(SL1−L2)':>11s}   winner")
    crossover = None
    prev_sign = None
    for g, s, l in zip(grid, sl1_mean, l2_mean):
        if np.isnan(s) or np.isnan(l):
            continue
        d = s - l
        win = "SL1" if d > 0.3 else ("L2" if d < -0.3 else "tie")
        print(f"  {g:8.1f}% {s:8.2f}% {l:8.2f}% {d:+10.2f}    {win}")
        append_csv_row(summary_path, _SUMMARY_FIELDS, {
            "dataset": dataset, "target_sparsity": f"{g:.1f}",
            "sl1_acc": f"{s:.4f}", "l2_acc": f"{l:.4f}",
            "delta_sl1_minus_l2": f"{d:.4f}", "n_seeds": len(sl1_mat),
        })
        sign = np.sign(d)
        if prev_sign is not None and sign != prev_sign and sign != 0:
            crossover = g
        prev_sign = sign if sign != 0 else prev_sign
    if crossover is not None:
        print(f"  → crossover near {crossover:.0f}% sparsity")


def _plot_dataset(dataset, ds_curves, seeds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:            # noqa: BLE001
        print(f"  [plot skipped: {e}]")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in (("l2", "tab:orange"), ("sl1", "tab:blue")):
        for seed in seeds:
            if seed not in ds_curves[method]:
                continue
            sp, ac = ds_curves[method][seed]
            order = np.argsort(sp)
            ax.plot(np.asarray(sp)[order], np.asarray(ac)[order],
                    "-o", color=color, alpha=0.5, ms=3,
                    label=method.upper() if seed == seeds[0] else None)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{dataset}: SL1 vs L2 + soft-threshold pruning")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(config.paths.pics_path, exist_ok=True)
    out = os.path.join(config.paths.pics_path, f"tsc_prune_{dataset}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  plot → {out}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    meta = _load_dataset_meta()
    p = argparse.ArgumentParser(
        description="SL1 vs L2 + soft-threshold pruning for ESN classification")
    p.add_argument("--datasets", nargs="+", default=sorted(meta.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                   help="Reservoir/split seeds to average over (default: 1 2 3)")
    p.add_argument("--reservoir_source", default="sl1", choices=["sl1", "l2", "none"],
                   help="Which configs/TSC_settings_*.json supplies the (shared) "
                        "reservoir hyperparameters per dataset (default: sl1)")
    p.add_argument("--l2_reg_grid", nargs="+", type=float,
                   default=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
                   help="λ grid the L2 arm tunes over on validation (dense acc)")
    p.add_argument("--sl1_reg_param", type=float, default=None,
                   help="Fixed SmoothL1 λ (default: per-dataset from "
                        "configs/TSC_settings_sl1.json, else 1.0). λ is a poor "
                        "sparsity knob — thres is swept instead.")
    p.add_argument("--thres_grid", nargs="+", type=float,
                   default=[1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                   help="SL1 soft-threshold magnitudes to sweep (sparsity knob)")
    p.add_argument("--l2_sparsity_targets", nargs="+", type=float,
                   default=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99],
                   help="Target zero-fractions for the L2 post-hoc prune sweep "
                        "(mapped to |w| quantiles per fit)")
    p.add_argument("--sparsity_grid", nargs="+", type=float,
                   default=[50, 60, 70, 80, 85, 90, 93, 95, 97, 98, 99],
                   help="Common sparsity levels (%%) to report Δ(SL1−L2) at")
    p.add_argument("--epochs", type=int, default=200,
                   help="Max Newton iterations per fit (early-stops kick in earlier)")
    p.add_argument("--l2_refit", action="store_true",
                   help="Refit the surviving support after each L2 prune (IMP-style, "
                        "a strictly stronger L2 baseline than prune-once)")
    p.add_argument("--sl1_refit", action="store_true",
                   help="Refit the SL1 surviving support once with a plain ridge "
                        "before scoring (prune-then-refit debiasing). Removes the "
                        "tau-shrinkage bias on surviving weights, making the SL1 "
                        "arm symmetric with an --l2_refit L2 arm: the comparison "
                        "then isolates WHICH support each penalty selects.")
    p.add_argument("--l2_select", default="dense", choices=["dense", "prune"],
                   help="How the L2 arm picks λ. 'dense' (default): one λ, tuned "
                        "for dense val accuracy — near-OLS on collinear reservoir "
                        "states, so its compensating weight pairs shatter under "
                        "thresholding; a WEAK baseline kept for reference. 'prune': "
                        "re-select λ at every sparsity level by POST-PRUNE val "
                        "accuracy (upper envelope over λ) — the fair strong baseline, "
                        "use this for reported results.")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel workers over (dataset, seed) units. "
                        "-1 = all cores. Forced to 1 under --gpu.")
    p.add_argument("--csv_path", default="./result/tsc_prune_compare.csv")
    p.add_argument("--summary_path", default="./result/tsc_prune_compare_summary.csv")
    p.add_argument("--plot", action="store_true", help="Save acc-vs-sparsity plots")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU)
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compare(
        datasets=a.datasets, seeds=a.seeds, reservoir_source=a.reservoir_source,
        l2_reg_grid=a.l2_reg_grid, sl1_reg_param=a.sl1_reg_param,
        thres_grid=a.thres_grid, l2_sparsity_targets=a.l2_sparsity_targets,
        sparsity_grid=a.sparsity_grid, epochs=a.epochs,
        csv_path=a.csv_path, summary_path=a.summary_path,
        use_cache=not a.no_cache, use_gpu=a.use_gpu,
        l2_refit=a.l2_refit, l2_select=a.l2_select, sl1_refit=a.sl1_refit,
        n_jobs=a.n_jobs, plot=a.plot,
    )
