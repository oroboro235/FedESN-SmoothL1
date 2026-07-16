# tsc_centralized_search.py — random hyperparameter search for centralized ESN classification.
#
# Pairs with tsc_centralized_sensitivity.py (single-parameter sweeps, which reuses
# this module's data/reservoir/readout helpers).
#
# Pipeline per trial:
#   1. Load dataset → standardise → stratified train/val split (80/20 by default)
#   2. Run reservoir on every sequence → collect last hidden state
#   3. Train Clr_Node.fit_from_states() with Adam + hand-computed gradients
#   4. Evaluate on the validation split → val_acc, val_sparsity
#   5. After all trials: re-train best config on full train set → test_acc, test_sparsity
#
# NOTE: Clr_Node.fit_from_states() is used instead of the reservoirpy pipeline because
# model.fit() passes label arrays as the readout's X input, which fails check_node_input()
# for sequence-classification settings.
#
# Results are written incrementally to result/tsc_centralized_search.csv (preserves
# partial runs) and best configs to result/tsc_centralized_best_<reg_type>.json.
#
# Usage examples:
#   python tsc_centralized_search.py
#   python tsc_centralized_search.py --datasets har char --reg_types sl1 l2
#   python tsc_centralized_search.py --n_trials 30 --epochs 500

import argparse
import hashlib
import json
import os
import random
import time
from copy import deepcopy

import numpy as np
from reservoirpy.nodes import Reservoir

import config
import metrics
import pruning
from data_loader import read_data, one_hot, standardize
from readout_node import Clr_Node
from utils import log_uniform, init_csv, append_csv_row, BestTracker, parallel_map


# ─── Dataset meta-information ─────────────────────────────────────────────────

def _load_dataset_meta() -> dict:
    """Return {dataset_name: {input_dim, output_dim, units}} for all TSC datasets.

    Thin alias kept for the scripts that import it from here (tsc_centralized_
    {eval,noniid_eval}.py, tsc_loss_compare.py, tsc_prune_compare.py); the actual
    source is the configs/datasets_tsc.json registry (config.load_tsc_dataset_meta).
    """
    return config.load_tsc_dataset_meta()


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _to_int_labels(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y.astype(int)


def _ensure_3d(X: np.ndarray) -> np.ndarray:
    """Ensure X has shape (n_samples, timesteps, n_features)."""
    if X.ndim == 2:
        return X[:, :, np.newaxis]
    return X


def _load_dataset(name: str, use_cache: bool) -> tuple:
    """Load and cache (Xtr, ytr_int, Xte, yte_int) with integer labels."""
    cache_file = config.paths.cache_path + name + ".npz"
    if use_cache and os.path.exists(cache_file):
        d = np.load(cache_file)
        Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    else:
        Xtr, ytr, Xte, yte = read_data(name)
        np.savez(cache_file, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, _to_int_labels(ytr), Xte, _to_int_labels(yte)


def _stratified_split(X: np.ndarray, y: np.ndarray,
                      val_frac: float = 0.2,
                      seed: int = 0) -> tuple:
    """Stratified train/val split ensuring every class appears in both sets."""
    rng = np.random.RandomState(seed)
    tr_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.extend(idx[:n_val])
        tr_idx.extend(idx[n_val:])
    return X[np.array(tr_idx)], y[np.array(tr_idx)], X[np.array(val_idx)], y[np.array(val_idx)]


# ─── Reservoir state extraction ───────────────────────────────────────────────

def _extract_states(X: np.ndarray, reservoir: Reservoir) -> np.ndarray:
    """Run reservoir on each sequence; return the last hidden state per sequence."""
    states = []
    for x in X:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = reservoir.run(x)
        states.append(out[-1].copy())
        reservoir.reset()
    return np.vstack(states)


# ─── Single trial ─────────────────────────────────────────────────────────────

def _run_trial(
    Xtr: np.ndarray, ytr_oh: np.ndarray,
    Xeval: np.ndarray, yeval_int: np.ndarray,
    meta: dict, reg_type: str, cfg: dict, seed: int,
    use_gpu: bool = config.USE_GPU,
) -> tuple:
    """Build reservoir, extract states, train Clr_Node, evaluate on a held-out set.

    Used for both the validation step (Xeval=Xval) and the final test step
    (Xeval=Xte, Xtr=full train set) — the procedure is identical, only the data
    splits differ.

    Thin wrapper over _fit_eval that returns only (acc, sparsity), the pair the
    parallel validation worker and the trial cache expect.  The richer metric
    set (macro-F1 / balanced accuracy) and the reservoir states are computed only
    at final-test time via _fit_eval directly, so the hot validation loop and the
    on-disk trial cache stay unchanged.

    Returns:
        (acc, sparsity) both in percent [0, 100].
    """
    scores, sparsity, _Wout, _Xtr_s, _Xeval_s = _fit_eval(
        Xtr, ytr_oh, Xeval, yeval_int, meta, reg_type, cfg, seed, use_gpu)
    return scores["acc"], sparsity


def _fit_eval(
    Xtr: np.ndarray, ytr_oh: np.ndarray,
    Xeval: np.ndarray, yeval_int: np.ndarray,
    meta: dict, reg_type: str, cfg: dict, seed: int,
    use_gpu: bool = config.USE_GPU,
) -> tuple:
    """Build reservoir, extract states, train Clr_Node, evaluate on a held-out set.

    Returns:
        (scores, sparsity, Wout, Xtr_states, Xeval_states) where *scores* is the
        metrics.classification_scores dict (acc / macro_f1 / balanced_acc, %).
        The states and Wout are returned so the final phase can reuse them for the
        CE+IMP baseline without re-running the reservoir.
    """
    reservoir = Reservoir(
        units         = meta["units"],
        lr            = cfg["lr"],
        sr            = cfg["sr"],
        input_scaling = cfg["input_scaling"],
        seed          = seed,
    )

    Xtr_states   = _extract_states(Xtr,   reservoir)
    Xeval_states = _extract_states(Xeval, reservoir)

    readout = Clr_Node(
        reg_param        = cfg["reg_param"],
        reg_type         = reg_type,
        thres            = cfg["thres"],
        alpha_init       = cfg["alpha_init"],
        alpha_multiplier = cfg["alpha_multiplier"],
        patience         = cfg.get("patience", config.sl1_defaults.PATIENCE),
        stag_tol         = cfg.get("stag_tol", config.sl1_defaults.STAG_TOL),
        epochs           = cfg["epochs"],
        use_gpu          = use_gpu,
    )
    Wout     = readout.fit_from_states(Xtr_states, ytr_oh)
    y_pred   = np.argmax(Xeval_states @ Wout, axis=1)
    scores   = metrics.classification_scores(yeval_int, y_pred, meta["output_dim"])
    sparsity = float((Wout == 0).mean() * 100)
    return scores, sparsity, Wout, Xtr_states, Xeval_states


# ─── Parallel trial worker ─────────────────────────────────────────────────────

# Per-dataset arrays shared with pool workers via fork (copy-on-write). Set in
# search() before each parallel_map() call so the forked workers inherit them
# without per-task pickling of the (potentially large) state matrices.
_WORKER_DATA: dict = {}


def _trial_worker(reg_type: str, cfg: dict, trial_seed: int, use_gpu: bool):
    """Run one validation trial in a pool worker; reads data from _WORKER_DATA.

    Returns (val_acc, val_sparsity), or ("__error__", message) so the parent can
    log the failure and record NaN without the worker touching shared state.
    """
    d = _WORKER_DATA
    t0 = time.time()
    try:
        # Progress is shown by the parent's live bar (parallel_map(progress=...));
        # workers stay silent on success so nothing clobbers the bar.
        return _run_trial(
            d["Xtr"], d["ytr_oh"], d["Xval"], d["yval_int"],
            d["meta"], reg_type, cfg, seed=trial_seed, use_gpu=use_gpu,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"    [worker] {reg_type} seed={trial_seed} ERROR in "
              f"{time.time() - t0:.1f}s: {exc}", flush=True)
        return ("__error__", str(exc))


# ─── Search space ─────────────────────────────────────────────────────────────

def _trial_key(dataset: str, reg_type: str, cfg: dict, trial_seed: int) -> str:
    """Stable 16-char hash uniquely identifying a trial's inputs."""
    payload = {
        "dataset":       dataset,
        "reg_type":      reg_type,
        "trial_seed":    trial_seed,
        "sr":            round(cfg["sr"],            8),
        "lr":            round(cfg["lr"],            8),
        "input_scaling": round(cfg["input_scaling"], 8),
        "reg_param":     round(cfg["reg_param"],     8),
        "epochs":        cfg["epochs"],
        "thres":         cfg["thres"],
        "alpha_init":       cfg["alpha_init"],
        "alpha_multiplier": cfg["alpha_multiplier"],
        "patience":         cfg.get("patience", config.sl1_defaults.PATIENCE),
        "stag_tol":         cfg.get("stag_tol", config.sl1_defaults.STAG_TOL),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class TrialCache:
    """Persistent JSON store; prevents re-running identical trials."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)
            print(f"[TrialCache] Loaded {len(self._data)} cached trials from {path}")

    def get(self, key: str):
        v = self._data.get(key)
        return (v["val_acc"], v["val_sparsity"]) if v else None

    def set(self, key: str, val_acc: float, val_sparsity: float):
        self._data[key] = {"val_acc": val_acc, "val_sparsity": val_sparsity}
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f)


def _sample_config(rng: random.Random, epochs: int) -> dict:
    return {
        "sr":            rng.uniform(0.3, 3.0),
        "lr":            log_uniform(0.005, 1.0,   rng),
        "input_scaling": log_uniform(0.01,  100.0, rng),
        "reg_param":     log_uniform(1e-4,  1e2,   rng),
        "epochs":        epochs,
    }


# ─── Composite scoring and BestTracker ────────────────────────────────────────

def _composite_score(val_acc: float, val_sparsity: float,
                     sparsity_weight: float) -> float:
    """score = (1 - w) * val_acc + w * val_sparsity  (both in %, higher is better)."""
    return (1.0 - sparsity_weight) * val_acc + sparsity_weight * val_sparsity


def _report(tracker: BestTracker, test_results: dict,
            sparsity_weight: float = 0.0, min_sparsity: float = 0.0):
    """Print the best config per (dataset, reg_type) alongside its test result."""
    print("\n" + "=" * 70)
    print("TSC CENTRALIZED ESN — HYPERPARAMETER SEARCH SUMMARY")
    if sparsity_weight > 0 or min_sparsity > 0:
        print(f"  sparsity_weight={sparsity_weight:.2f}  min_sparsity={min_sparsity:.1f}%")
    print("=" * 70)
    _nan = {"acc": float("nan"), "macro_f1": float("nan"),
            "balanced_acc": float("nan"), "sparsity": float("nan")}
    seen_datasets = []
    for (dataset, reg_type), score, rec in tracker.items():
        if dataset not in seen_datasets:
            seen_datasets.append(dataset)
        t = test_results.get((dataset, reg_type), _nan)
        print(f"\n  {dataset}  |  {reg_type}")
        print(f"    Best trial   : {rec['trial']}")
        print(f"    Score        : {score:.2f}   "
              f"val_acc={rec['val_acc']:.2f}%   val_sparsity={rec['val_sparsity']:.1f}%")
        print(f"    Test result  : acc={t['acc']:.2f}%   f1={t['macro_f1']:.2f}%   "
              f"bal_acc={t['balanced_acc']:.2f}%   sparsity={t['sparsity']:.1f}%")
        for k, v in rec["cfg"].items():
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")

    # CE+IMP baselines (not in the tracker — a post-hoc matched-sparsity compare).
    for dataset in seen_datasets:
        t = test_results.get((dataset, "ce_imp"))
        if t is None:
            continue
        print(f"\n  {dataset}  |  ce_imp  (CE dense + IMP @ SL1 sparsity)")
        print(f"    Test result  : acc={t['acc']:.2f}%   f1={t['macro_f1']:.2f}%   "
              f"bal_acc={t['balanced_acc']:.2f}%   sparsity={t['sparsity']:.1f}%")


# ─── CSV output ───────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "trial", "split",
    "sr", "lr", "input_scaling", "reg_param", "thres",
    "alpha_init", "alpha_multiplier",
    "epochs",
    "score", "val_acc", "val_sparsity", "test_acc", "test_sparsity",
    "test_macro_f1", "test_balanced_acc",
]


# ─── JSON export of best configs ─────────────────────────────────────────────

def _entry_score(entry: dict, sparsity_weight: float) -> float:
    """Score of a JSON entry, for cross-run comparison.

    Prefers the stored ``_search_score`` (written by recent runs); falls back to
    recomputing the composite score from the stored val metrics for legacy
    entries that predate ``_search_score``.
    """
    if "_search_score" in entry:
        return entry["_search_score"]
    return _composite_score(
        entry.get("_search_val_acc", float("-inf")),
        entry.get("_search_val_sparsity", 0.0),
        sparsity_weight,
    )


def _save_best_json(tracker: BestTracker, dataset_meta: dict,
                    reg_types: list, out_dir: str,
                    sparsity_weight: float = 0.0):
    """Save best configs as tsc_centralized_best_<reg_type>.json (one per reg_type).

    Merges with any existing file: for each dataset the higher-scoring config
    (this run vs. the one already on disk, e.g. a previous seed) is kept, so
    re-running with a new seed only overwrites a dataset's config when it
    actually beats the stored one.
    """
    os.makedirs(out_dir, exist_ok=True)
    for reg_type in reg_types:
        entries = []
        for (ds, rt), score, rec in tracker.items():
            if rt != reg_type:
                continue
            val_acc, val_sp, cfg, trial = (
                rec["val_acc"], rec["val_sparsity"], rec["cfg"], rec["trial"]
            )
            meta  = dataset_meta.get(ds, {})
            entry = {
                "dataset":              ds,
                "input_dim":            meta.get("input_dim"),
                "output_dim":           meta.get("output_dim"),
                "units":                meta.get("units", 500),
                "lr":                   round(cfg["lr"],            8),
                "sr":                   round(cfg["sr"],            8),
                "input_scaling":        round(cfg["input_scaling"], 8),
                "input_connectivity":   0.1,
                "rc_connectivity":      0.1,
                "reg_param":            round(cfg["reg_param"],     8),
                "thres":                cfg["thres"],
                "epochs":               cfg["epochs"],
                "_search_score":        round(score, 4),
                "_search_val_acc":      round(val_acc, 4),
                "_search_val_sparsity": round(val_sp,  4),
                "_search_trial":        trial,
            }
            if reg_type == "sl1":
                entry["alpha_init"]       = cfg["alpha_init"]
                entry["alpha_multiplier"] = cfg["alpha_multiplier"]
            entries.append(entry)

        if not entries:
            continue

        out_path = os.path.join(out_dir, f"tsc_centralized_best_{reg_type}.json")

        # Merge with existing file: keep the higher-scoring config per dataset.
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
            if old is None or _entry_score(entry, sparsity_weight) > \
                              _entry_score(old, sparsity_weight):
                if old is not None:
                    print(f"  [{reg_type}/{ds}] new best "
                          f"{_entry_score(entry, sparsity_weight):.4f} > "
                          f"{_entry_score(old, sparsity_weight):.4f} — replaced")
                merged[ds] = entry
            else:
                print(f"  [{reg_type}/{ds}] kept existing "
                      f"{_entry_score(old, sparsity_weight):.4f} ≥ "
                      f"{_entry_score(entry, sparsity_weight):.4f}")

        out_entries = [merged[ds] for ds in sorted(merged)]
        with open(out_path, "w") as f:
            json.dump(out_entries, f, indent=4)
        print(f"Best configs ({reg_type}) saved → {out_path}")


# ─── Main search ──────────────────────────────────────────────────────────────

def search(
    datasets:         list,
    reg_types:        list,
    n_trials:         int,
    epochs:           int,
    val_frac:         float,
    seed:             int,
    use_cache:        bool,
    csv_path:         str,
    sparsity_weight:  float = 0.0,
    min_sparsity:     float = 0.0,
    json_dir:         str   = "./result",
    thres:            float = config.sl1_defaults.THRES,
    alpha_init:       float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    patience:         int   = config.sl1_defaults.PATIENCE,
    stag_tol:         float = config.sl1_defaults.STAG_TOL,
    use_trial_cache:  bool  = True,
    use_gpu:          bool  = config.USE_GPU,
    n_jobs:           int   = 1,
):
    dataset_meta = _load_dataset_meta()
    tracker      = BestTracker()
    init_csv(csv_path, _CSV_FIELDS)

    # GPU + multiprocessing oversubscribe a single device; serialise in that case.
    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"(parallel workers contend for one device).")
        n_jobs = 1

    cache_json_path = os.path.splitext(csv_path)[0] + "_trial_cache.json"
    trial_cache     = TrialCache(cache_json_path) if use_trial_cache else None

    for dataset in datasets:
        if dataset not in dataset_meta:
            print(f"[SKIP] '{dataset}' not in {config.paths.tsc_dataset_meta}")
            continue

        meta = dataset_meta[dataset]
        print(f"\n{'='*65}")
        print(f"Dataset: {dataset}  "
              f"(input={meta['input_dim']}, classes={meta['output_dim']}, "
              f"units={meta['units']})")
        print(f"{'='*65}")

        try:
            Xtr_raw, ytr_int_raw, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
        except Exception as exc:
            print(f"  [ERROR] loading {dataset}: {exc}")
            continue

        Xtr_raw = _ensure_3d(Xtr_raw)
        Xte_raw = _ensure_3d(Xte_raw)

        Xtr_sub, ytr_sub_int, Xval, yval_int = _stratified_split(
            Xtr_raw, ytr_int_raw, val_frac=val_frac, seed=seed
        )
        Xtr_sub, Xval, Xte = standardize(Xtr_sub, Xval, Xte_raw)

        n_classes   = meta["output_dim"]
        ytr_sub_oh  = one_hot(ytr_sub_int, n_classes)
        ytr_full_oh = one_hot(ytr_int_raw,  n_classes)

        # Share this dataset's arrays with pool workers via fork (copy-on-write):
        # set the module global before parallel_map() forks, so workers inherit
        # the state matrices without per-task pickling.
        _WORKER_DATA.clear()
        _WORKER_DATA.update({
            "Xtr": Xtr_sub, "ytr_oh": ytr_sub_oh,
            "Xval": Xval, "yval_int": yval_int, "meta": meta,
        })

        for reg_type in reg_types:
            print(f"\n  reg_type = {reg_type}  ({n_trials} trials × {epochs} epochs)")

            # ── Phase 1: sample every trial config up front (sequential) ────────
            # Deterministic per-(seed, dataset, reg_type, trial) sampling: a
            # trial's cfg and reservoir seed no longer depend on which OTHER
            # datasets / reg_types / n_trials are in the run (the old shared RNG
            # stream + global counter meant any change to the command line shifted
            # every key, so the trial cache could only ever hit on an identical
            # re-run).  Now subset runs, extended n_trials, and added reg_types
            # all reuse previously cached trials.
            specs = []   # (trial, cfg, trial_seed, key, cached)
            for trial in range(n_trials):
                trial_rng = random.Random(f"{seed}/{dataset}/{reg_type}/{trial}")
                cfg       = _sample_config(trial_rng, epochs)
                # Uniform SL1 schedule (not searched) — identical for every trial/dataset.
                cfg["thres"]            = thres
                cfg["alpha_init"]       = alpha_init
                cfg["alpha_multiplier"] = alpha_multiplier
                cfg["patience"]         = patience
                cfg["stag_tol"]         = stag_tol
                trial_seed = seed + trial

                key    = _trial_key(dataset, reg_type, cfg, trial_seed)
                cached = trial_cache.get(key) if trial_cache else None
                specs.append((trial, cfg, trial_seed, key, cached))

            if trial_cache:
                n_hit = sum(1 for s in specs if s[4] is not None)
                print(f"  [TrialCache] {n_hit}/{n_trials} trials served from cache")

            # ── Phase 2: evaluate the uncached trials (parallel when n_jobs>1) ──
            pending_args = [(reg_type, cfg, trial_seed, use_gpu)
                            for (_, cfg, trial_seed, _, cached) in specs
                            if cached is None]
            pending_res  = iter(parallel_map(
                _trial_worker, pending_args, n_jobs,
                progress=f"  {dataset}/{reg_type}"))

            # ── Phase 3: record results in trial order (deterministic) ──────────
            for (trial, cfg, trial_seed, key, cached) in specs:
                is_cached = cached is not None
                if is_cached:
                    val_acc, val_sparsity = cached
                else:
                    res = next(pending_res)
                    if isinstance(res, tuple) and res and res[0] == "__error__":
                        print(f"  trial {trial+1}/{n_trials} — [ERROR: {res[1]}]")
                        val_acc = val_sparsity = float("nan")
                    else:
                        val_acc, val_sparsity = res
                    if trial_cache and not (val_acc != val_acc):
                        trial_cache.set(key, val_acc, val_sparsity)

                score = _composite_score(val_acc, val_sparsity, sparsity_weight)
                # Reject configs below the minimum-sparsity floor before tracking.
                is_best = False
                if val_sparsity >= min_sparsity:
                    is_best = tracker.update((dataset, reg_type), score, {
                        "val_acc":      val_acc,
                        "val_sparsity": val_sparsity,
                        "cfg":          deepcopy(cfg),
                        "trial":        trial,
                    })

                # Only announce a trial when it sets a new best score — the full
                # per-trial record still goes to the CSV below, so nothing is lost.
                if is_best:
                    print(f"  trial {trial+1}/{n_trials} — [NEW BEST]"
                          f"{' [CACHED]' if is_cached else ''}  "
                          f"sr={cfg['sr']:.3f}  lr={cfg['lr']:.4f}  "
                          f"is={cfg['input_scaling']:.3f}  reg={cfg['reg_param']:.3g}  "
                          f"thres={cfg['thres']:.0e}")
                    print(f"    →  val_acc={val_acc:.2f}%  "
                          f"val_sparsity={val_sparsity:.1f}%  score={score:.2f}")

                append_csv_row(csv_path, _CSV_FIELDS, {
                    "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset":       dataset,
                    "reg_type":      reg_type,
                    "trial":         trial,
                    "split":         "val",
                    "score":         f"{score:.4f}",
                    "val_acc":       f"{val_acc:.4f}",
                    "val_sparsity":  f"{val_sparsity:.4f}",
                    "test_acc":      "",
                    "test_sparsity": "",
                    **{k: (f"{v:.6g}" if isinstance(v, float) else v)
                       for k, v in cfg.items()},
                })

    # ── Final evaluation ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Final evaluation: re-training best configs on full training set…")
    test_results = {}

    for dataset in datasets:
        if dataset not in dataset_meta:
            continue
        meta = dataset_meta[dataset]

        try:
            Xtr_raw, ytr_int_raw, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
        except Exception:
            continue

        Xtr_raw = _ensure_3d(Xtr_raw)
        Xte_raw = _ensure_3d(Xte_raw)
        Xtr_std, Xte_std = standardize(Xtr_raw, Xte_raw)

        n_classes   = meta["output_dim"]
        ytr_full_oh = one_hot(ytr_int_raw, n_classes)

        # Stash of (states, dense CE Wout) for the CE+IMP baseline below, keyed by
        # the reservoir config so it is reused rather than re-run.
        sl1_result = None    # (target_sparsity, best_cfg) to match IMP against

        for reg_type in reg_types:
            best = tracker.get((dataset, reg_type))
            if best is None:
                continue
            _score, rec = best
            best_val_acc, best_val_sp, best_cfg, best_trial = (
                rec["val_acc"], rec["val_sparsity"], rec["cfg"], rec["trial"]
            )

            print(f"  {dataset}/{reg_type}  (trial={best_trial}  "
                  f"val_acc={best_val_acc:.2f}%  val_sparsity={best_val_sp:.1f}%)",
                  end="", flush=True)
            try:
                scores, test_sparsity, _W, _Xs, _Xes = _fit_eval(
                    Xtr_std, ytr_full_oh, Xte_std, yte_int,
                    meta, reg_type, best_cfg, seed=seed,
                    use_gpu=use_gpu,
                )
            except Exception as exc:
                print(f"  [ERROR: {exc}]")
                scores = {"acc": float("nan"), "macro_f1": float("nan"),
                          "balanced_acc": float("nan")}
                test_sparsity = float("nan")

            print(f"  →  test_acc={scores['acc']:.2f}%  "
                  f"f1={scores['macro_f1']:.2f}%  bal={scores['balanced_acc']:.2f}%  "
                  f"test_sparsity={test_sparsity:.1f}%")
            test_results[(dataset, reg_type)] = {**scores, "sparsity": test_sparsity}
            if reg_type == "sl1" and test_sparsity == test_sparsity:  # not NaN
                sl1_result = (test_sparsity, best_cfg)

            append_csv_row(csv_path, _CSV_FIELDS, {
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset":       dataset,
                "reg_type":      reg_type,
                "trial":         f"best({best_trial})",
                "split":         "test",
                "score":         "",
                "val_acc":       f"{best_val_acc:.4f}",
                "val_sparsity":  f"{best_val_sp:.4f}",
                "test_acc":         f"{scores['acc']:.4f}",
                "test_sparsity":    f"{test_sparsity:.4f}",
                "test_macro_f1":    f"{scores['macro_f1']:.4f}",
                "test_balanced_acc": f"{scores['balanced_acc']:.4f}",
                **{k: (f"{v:.6g}" if isinstance(v, float) else v)
                   for k, v in best_cfg.items()},
            })

        # ── CE + IMP baseline at SL1's sparsity (the honest competitor) ─────────
        # Fit a DENSE cross-entropy readout on SL1's own reservoir, then IMP-prune
        # it to the exact sparsity SL1 reached, and evaluate at matched sparsity —
        # the DEVLOG "does SL1 beat CE+IMP?" comparison, on the test set.
        if sl1_result is not None:
            target_sp, sl1_cfg = sl1_result
            print(f"  {dataset}/ce_imp  (CE dense + IMP → {target_sp:.1f}% sparse)",
                  end="", flush=True)
            try:
                # Dense CE fit on SL1's reservoir (reg_type="none" → no penalty).
                _s, _sp, W_dense, Xtr_s, Xte_s = _fit_eval(
                    Xtr_std, ytr_full_oh, Xte_std, yte_int,
                    meta, "none", sl1_cfg, seed=seed, use_gpu=use_gpu)
                W_imp = pruning.ce_imp_prune(
                    Xtr_s, ytr_full_oh, W_dense, target_sp,
                    l2=1e-3, n_rounds=5, n_refit=8)
                y_pred = np.argmax(Xte_s @ W_imp, axis=1)
                imp_scores = metrics.classification_scores(yte_int, y_pred, n_classes)
                imp_sp = float((W_imp == 0).mean() * 100)
            except Exception as exc:
                print(f"  [ERROR: {exc}]")
                imp_scores = {"acc": float("nan"), "macro_f1": float("nan"),
                              "balanced_acc": float("nan")}
                imp_sp = float("nan")

            print(f"  →  test_acc={imp_scores['acc']:.2f}%  "
                  f"f1={imp_scores['macro_f1']:.2f}%  "
                  f"bal={imp_scores['balanced_acc']:.2f}%  sparsity={imp_sp:.1f}%")
            test_results[(dataset, "ce_imp")] = {**imp_scores, "sparsity": imp_sp}
            append_csv_row(csv_path, _CSV_FIELDS, {
                "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset":           dataset,
                "reg_type":          "ce_imp",
                "trial":             "baseline",
                "split":             "test",
                "test_acc":          f"{imp_scores['acc']:.4f}",
                "test_sparsity":     f"{imp_sp:.4f}",
                "test_macro_f1":     f"{imp_scores['macro_f1']:.4f}",
                "test_balanced_acc": f"{imp_scores['balanced_acc']:.4f}",
            })

    _report(tracker, test_results, sparsity_weight, min_sparsity)
    _save_best_json(tracker, dataset_meta, reg_types, json_dir, sparsity_weight)
    print(f"\nAll results saved → {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    meta         = _load_dataset_meta()
    all_datasets = sorted(meta.keys())

    p = argparse.ArgumentParser(
        description="Centralized ESN classification hyperparameter search"
    )
    p.add_argument("--datasets",  nargs="+", default=all_datasets)
    p.add_argument("--reg_types", nargs="+", default=["none", "l2", "sl1"],
                   choices=["none", "l2", "sl1"])
    p.add_argument("--n_trials",  type=int,   default=20)
    p.add_argument("--epochs",    type=int,   default=500)
    p.add_argument("--val_frac",  type=float, default=0.2)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no_cache",  action="store_true")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; auto-falls back to CPU")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel worker processes for trials (1=serial [default], "
                        "-1=all cores). Forced to 1 under --gpu (single-device "
                        "contention).")
    p.add_argument("--csv_path",  default="./result/tsc_centralized_search.csv")
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight of sparsity in composite score (0=pure acc, 1=pure sparsity)")
    p.add_argument("--min_sparsity",    type=float, default=0.0,
                   help="Hard lower bound on val_sparsity (%%) for a config to qualify")
    p.add_argument("--json_dir",  default="./result",
                   help="Directory for tsc_centralized_best_<reg_type>.json files")
    # ── SL1 sparsification schedule (uniform across datasets; NOT searched) ──────
    p.add_argument("--thres",     type=float, default=config.sl1_defaults.THRES,
                   help="Soft-threshold magnitude zeroing |w| < thres "
                        f"(default: {config.sl1_defaults.THRES})")
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help=f"Initial SmoothL1 alpha (default: {config.sl1_defaults.ALPHA_INIT})")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help="Per-iteration alpha growth factor for the Newton/Adam "
                        f"continuation (default: {config.sl1_defaults.ALPHA_MULTIPLIER})")
    p.add_argument("--patience", type=int, default=config.sl1_defaults.PATIENCE,
                   help="Newton stagnation early-stop window; 0 disables "
                        f"(default: {config.sl1_defaults.PATIENCE})")
    p.add_argument("--stag_tol", type=float, default=config.sl1_defaults.STAG_TOL,
                   help="Max non-zero-fraction change over the patience window to "
                        f"count as converged (default: {config.sl1_defaults.STAG_TOL})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    search(
        datasets        = args.datasets,
        reg_types       = args.reg_types,
        n_trials        = args.n_trials,
        epochs          = args.epochs,
        val_frac        = args.val_frac,
        seed            = args.seed,
        use_cache       = not args.no_cache,
        csv_path        = args.csv_path,
        sparsity_weight = args.sparsity_weight,
        min_sparsity    = args.min_sparsity,
        json_dir        = args.json_dir,
        thres            = args.thres,
        alpha_init       = args.alpha_init,
        alpha_multiplier = args.alpha_multiplier,
        patience         = args.patience,
        stag_tol         = args.stag_tol,
        use_gpu          = args.use_gpu,
        n_jobs           = args.n_jobs,
    )
