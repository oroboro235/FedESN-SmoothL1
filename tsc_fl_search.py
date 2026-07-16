# tsc_fl_search.py — random hyperparameter search for FedSL1ESN (FL classification).
#
# Each trial calls tsc_fl_eval.main() with one sampled config and records both
# accuracy AND sparsity.  A composite score (identical to tsc_centralized_search.py)
# is used for best-config selection:
#
#   score = (1 - w) * val_acc + w * avg_sparsity   (w = --sparsity_weight)
#
# val_acc is the pooled-validation accuracy (val_frac of every client's local
# train shard) — the TEST accuracy is recorded but never used for selection, and
# the client partition is seeded once per search (data_seed = seed) so trials
# compare on the same split.  Trial configs are derived deterministically from
# (seed, dataset, reg_type, trial), so the persistent trial cache
# (<csv_path>_trial_cache.json) reuses results across subset runs, extended
# --n_trials, and added reg_types.
#
# Results:
#   ./result/tsc_fl_search.csv          — one row per trial (incremental)
#   ./result/tsc_fl_best_<reg>.json     — best configs per (dataset, reg_type)
#
# Usage:
#   python tsc_fl_search.py
#   python tsc_fl_search.py --datasets har char --reg_types sl1
#   python tsc_fl_search.py --n_trials 30 --sparsity_weight 0.3

import argparse
import hashlib
import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np

import config
from tsc_fl_eval import main as run_experiment
from utils import log_uniform, init_csv, append_csv_row, BestTracker, parallel_map


# ─── Search space ─────────────────────────────────────────────────────────────

def _sample_config(reg_type: str, rng: random.Random) -> dict:
    # NOTE: thres / alpha_init / alpha_multiplier are NOT searched — they are
    # uniform CLI arguments (identical across datasets), passed to run_experiment
    # directly in search() rather than sampled here.
    return {
        "sr":            rng.uniform(0.3, 3.0),
        "lr":            log_uniform(0.005, 1.0,   rng),
        "input_scaling": log_uniform(0.01,  100.0, rng),
        "reg_param":     log_uniform(1e-4,  1e2,   rng),
    }


# ─── Dataset → setting index map ──────────────────────────────────────────────

def _build_index_map(reg_type: str) -> dict:
    json_path = os.path.join(config.paths.configs_path, f"TSC_FL_settings_{reg_type}.json")
    with open(json_path) as f:
        settings = json.load(f)
    return {s["dataset"]: i for i, s in enumerate(settings)}


# ─── Fixed reservoir params from centralized search ──────────────────────────

_RESERVOIR_KEYS = ("sr", "lr", "input_scaling")

# Run-level values the search trials actually execute with — tsc_fl_eval.main()
# defaults, which the search never overrides.  Recorded in the best-config JSON
# (instead of whatever the base settings file happens to contain) so a saved
# entry reproduces its trial.
_RUN_DEFAULTS = {"local_lr": 1.0, "global_lr": 1.0, "local_epochs": 100}


def _load_fixed_reservoir(reg_type: str, json_dir: str) -> dict:
    """Load {dataset: {sr, lr, input_scaling}} from the tuned centralized configs.

    Used by --fix_reservoir: the reservoir hyperparameters found by
    tsc_centralized_search.py transfer to FL (all clients share the same frozen
    reservoir), so we freeze them and let the FL search focus on reg_param —
    cheaper, and the right thing to tune under a non-iid partition.

    Prefers the promoted configs/TSC_settings_<reg_type>.json; falls back to the
    raw search output <json_dir>/tsc_centralized_best_<reg_type>.json.
    """
    entries = config.load_settings("tsc", reg_type)
    src = f"configs/TSC_settings_{reg_type}.json"
    if not entries:
        path = os.path.join(json_dir, f"tsc_centralized_best_{reg_type}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--fix_reservoir needs configs/TSC_settings_{reg_type}.json or "
                f"{path}; run tsc_centralized_search.py for reg_type="
                f"'{reg_type}' first (or pass --centralized_json_dir)")
        with open(path) as f:
            entries = {s["dataset"]: s for s in json.load(f)}
        src = path
    print(f"  [fix_reservoir] {reg_type}: reservoir params from {src}")
    return {ds: {k: s[k] for k in _RESERVOIR_KEYS} for ds, s in entries.items()}


# ─── Parallel trial worker ─────────────────────────────────────────────────────

def _fl_trial_worker(dataset, reg_type, n_rounds, n_clients, trial_seed,
                     setting_idx, partition, dirichlet_alpha, use_cache,
                     exp_suffix, use_line_search, patience, use_gpu,
                     thres, alpha_init, alpha_multiplier, val_frac,
                     data_seed, cfg):
    """Run one FL trial (one run_experiment call) in a pool worker.

    Returns (val_acc, avg_acc, avg_sparsity, macro_f1, balanced_acc) for
    *dataset* — val_acc is the selection metric, avg_acc the report-only test
    accuracy at the best-val round — or ("__error__", message) so the parent
    records NaN and logs the failure. data_seed is FIXED across trials so every
    trial sees the same client partition (config quality is not confounded with
    partition luck); trial_seed varies only the reservoir/model randomness.
    """
    t0 = time.time()
    try:
        results = run_experiment(
            reg_type        = reg_type,
            n_rounds        = n_rounds,
            n_clients       = n_clients,
            seed            = trial_seed,
            setting_idx     = setting_idx,
            partition       = partition,
            dirichlet_alpha = dirichlet_alpha,
            val_frac        = val_frac,
            data_seed       = data_seed,
            use_cache       = use_cache,
            exp_suffix      = exp_suffix,
            use_line_search = use_line_search,
            patience        = patience,
            use_gpu         = use_gpu,
            thres           = thres,
            alpha_init      = alpha_init,
            alpha_multiplier= alpha_multiplier,
            param_overrides = cfg,
            verbose         = False,   # silence per-round trace; parent shows a bar
        )
        rd = results.get(dataset, {})
        # Progress is shown by the parent's live bar (parallel_map(progress=...));
        # workers stay silent on success so nothing clobbers the bar.
        return (rd.get("val_acc", float("nan")),
                rd.get("acc", float("nan")), rd.get("sparsity", float("nan")),
                rd.get("macro_f1", float("nan")), rd.get("balanced_acc", float("nan")))
    except Exception as exc:                                  # noqa: BLE001
        print(f"    [worker] {dataset}/{reg_type} seed={trial_seed} ERROR in "
              f"{time.time() - t0:.1f}s: {exc}", flush=True)
        return ("__error__", str(exc))


# ─── Trial cache (deterministic per-trial keys) ───────────────────────────────

def _trial_key(dataset: str, reg_type: str, cfg: dict, trial_seed: int,
               run_ctx: dict) -> str:
    """Stable 16-char hash of everything that determines a trial's result."""
    payload = {
        "dataset":    dataset,
        "reg_type":   reg_type,
        "trial_seed": trial_seed,
        **{k: round(cfg[k], 8)
           for k in ("sr", "lr", "input_scaling", "reg_param")},
        **run_ctx,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class _TrialCache:
    """Persistent JSON store keyed by _trial_key; value = 5-metric tuple.

    Works because trial configs are derived deterministically from
    (seed, dataset, reg_type, trial) — see search() phase 1 — so subset runs,
    extended --n_trials, and added reg_types reuse previously computed trials.
    """

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)
            print(f"[TrialCache] Loaded {len(self._data)} cached FL trials "
                  f"from {path}")

    def get(self, key: str):
        v = self._data.get(key)
        return tuple(v) if v else None

    def set(self, key: str, metrics: tuple):
        self._data[key] = list(metrics)
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f)


# ─── Composite scoring and BestTracker ────────────────────────────────────────

def _composite_score(avg_acc: float, avg_sparsity: float,
                     sparsity_weight: float) -> float:
    """score = (1 - w) * avg_acc + w * avg_sparsity  (both in %, higher is better)."""
    return (1.0 - sparsity_weight) * avg_acc + sparsity_weight * avg_sparsity


def _report(tracker: BestTracker, sparsity_weight: float = 0.0):
    """Print the best FL config per (dataset, reg_type)."""
    print("\n" + "=" * 70)
    print("TSC FL ESN — HYPERPARAMETER SEARCH SUMMARY")
    if sparsity_weight > 0:
        print(f"  sparsity_weight = {sparsity_weight:.2f}")
    print("=" * 70)
    for (dataset, reg_type), score, rec in tracker.items():
        print(f"\n  {dataset}  |  {reg_type}")
        print(f"    Best trial   : {rec['trial']}")
        print(f"    Score        : {score:.2f}   "
              f"val_acc={rec.get('val_acc', float('nan')):.2f}%   "
              f"test_acc={rec['avg_acc']:.2f}%   "
              f"avg_sparsity={rec['avg_sparsity']:.1f}%")
        print(f"    avg_macro_f1={rec.get('avg_macro_f1', float('nan')):.2f}%   "
              f"avg_balanced_acc={rec.get('avg_balanced_acc', float('nan')):.2f}%")
        for k, v in rec["cfg"].items():
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")


# ─── CSV output ───────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "trial",
    "sr", "lr", "input_scaling", "reg_param", "thres",
    "alpha_init", "alpha_multiplier",
    "score", "val_acc", "avg_acc", "avg_sparsity",
    "avg_macro_f1", "avg_balanced_acc",
]


# ─── JSON export of best configs ─────────────────────────────────────────────

def _entry_score(entry: dict, sparsity_weight: float) -> float:
    """Score of a JSON entry, for cross-run comparison.

    Recomputed from the stored metrics under the CURRENT sparsity_weight, so
    entries written by runs with a different weight compare on the same
    objective (a stored _search_score baked in its own run's weight). Prefers
    the validation accuracy (new runs); falls back to the test-based avg_acc
    for legacy entries, and to the stored _search_score as a last resort.
    """
    acc = entry.get("_search_val_acc", entry.get("_search_avg_acc"))
    if acc is not None:
        return _composite_score(acc, entry.get("_search_avg_sparsity", 0.0),
                                sparsity_weight)
    return entry.get("_search_score", float("-inf"))


def _save_best_json(tracker: BestTracker, reg_types: list,
                    index_maps: dict, out_dir: str,
                    thres: float, alpha_init: float, alpha_multiplier: float,
                    sparsity_weight: float = 0.0):
    """Save best FL configs as tsc_fl_best_<reg_type>.json (one per reg_type).

    Format mirrors configs/TSC_FL_settings_<reg_type>.json so the file can be used
    as an updated config directly.

    Merges with any existing file: for each dataset the higher-scoring config
    (this run vs. the one already on disk, e.g. a previous seed) is kept, so
    re-running with a new seed only overwrites a dataset's config when it
    actually beats the stored one.
    """
    os.makedirs(out_dir, exist_ok=True)
    for reg_type in reg_types:
        # Load base settings to fill in fixed fields (input_dim, output_dim, units…)
        json_path = os.path.join(config.paths.configs_path,
                                 f"TSC_FL_settings_{reg_type}.json")
        with open(json_path) as f:
            base_settings = {s["dataset"]: s for s in json.load(f)}

        entries = []
        for (ds, rt), score, rec in tracker.items():
            if rt != reg_type:
                continue
            acc, sp, cfg, trial = (
                rec["avg_acc"], rec["avg_sparsity"], rec["cfg"], rec["trial"]
            )
            base  = base_settings.get(ds, {})
            entry = {
                "dataset":              ds,
                "input_dim":            base.get("input_dim"),
                "output_dim":           base.get("output_dim"),
                "units":                base.get("units", 500),
                "lr":                   round(cfg["lr"],            8),
                "sr":                   round(cfg["sr"],            8),
                "input_scaling":        round(cfg["input_scaling"], 8),
                "input_connectivity":   base.get("input_connectivity", 0.1),
                "rc_connectivity":      base.get("rc_connectivity",   0.1),
                "reg_param":            round(cfg["reg_param"],     8),
                "thres":                thres,
                # The values the search trials actually ran with (main()'s
                # defaults — the search never overrides them), NOT the base
                # settings file's, so this entry reproduces the trial.
                "local_lr":             _RUN_DEFAULTS["local_lr"],
                "global_lr":            _RUN_DEFAULTS["global_lr"],
                "local_epochs":         _RUN_DEFAULTS["local_epochs"],
                "_search_score":        round(score, 4),
                "_search_val_acc":      round(rec.get("val_acc", float("nan")), 4),
                "_search_avg_acc":      round(acc, 4),
                "_search_avg_sparsity": round(sp,  4),
                "_search_trial":        trial,
            }
            if reg_type == "sl1":
                entry["alpha_init"]       = alpha_init
                entry["alpha_multiplier"] = alpha_multiplier
            entries.append(entry)

        if not entries:
            continue

        out_path = os.path.join(out_dir, f"tsc_fl_best_{reg_type}.json")

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
        print(f"Best FL configs ({reg_type}) saved → {out_path}")


# ─── Main search loop ─────────────────────────────────────────────────────────

def search(
    datasets:        list,
    reg_types:       list,
    n_trials:        int,
    n_rounds:        int,
    n_clients:       int,
    seed:            int,
    use_cache:       bool,
    use_line_search: bool,
    patience:        int,
    csv_path:        str,
    partition:       str   = "iid",
    dirichlet_alpha: float = 0.5,
    fix_reservoir:        bool = False,
    centralized_json_dir: str  = "./result",
    use_gpu:         bool  = config.USE_GPU,
    sparsity_weight: float = 0.0,
    json_dir:        str   = "./result",
    thres:            float = config.sl1_defaults.THRES,
    alpha_init:       float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    val_frac:         float = 0.2,
    use_trial_cache:  bool  = True,
    n_jobs:           int   = 1,
):
    init_csv(csv_path, _CSV_FIELDS)
    tracker    = BestTracker()
    index_maps = {rt: _build_index_map(rt) for rt in reg_types}

    # GPU + multiprocessing oversubscribe a single device; serialise in that case.
    if use_gpu and n_jobs != 1:
        print(f"[WARN] --n_jobs={n_jobs} ignored: GPU runs are forced serial "
              f"(parallel workers contend for one device).")
        n_jobs = 1

    # SL1's α continuation grows per round; if the round budget cannot reach the
    # α ceiling, the sparsification schedule is truncated mid-way and the
    # selected configs may not transfer to longer deployment runs.
    if "sl1" in reg_types and alpha_multiplier > 1:
        need = math.ceil(math.log(config.sl1_defaults.ALPHA_MAX / alpha_init,
                                  alpha_multiplier))
        if n_rounds < need:
            print(f"[WARN] sl1: alpha reaches its ceiling only after ~{need} "
                  f"rounds (alpha_init={alpha_init:g} ×{alpha_multiplier:g}/round "
                  f"→ alpha_max={config.sl1_defaults.ALPHA_MAX:g}), but "
                  f"--n_rounds={n_rounds} truncates the sparsification schedule. "
                  f"Selected sl1 configs may not transfer; consider "
                  f"--n_rounds {need}+ (or a larger --alpha_multiplier).")

    # When fixing the reservoir, preload the centralized best configs per reg_type.
    fixed_reservoir = (
        {rt: _load_fixed_reservoir(rt, centralized_json_dir) for rt in reg_types}
        if fix_reservoir else {}
    )

    # The client partition / val carve-out is seeded by *seed* for the WHOLE
    # search (data_seed), while each trial's model randomness uses seed+trial —
    # so all trials compare on the same data split.
    data_seed = seed

    cache_json_path = os.path.splitext(csv_path)[0] + "_trial_cache.json"
    trial_cache     = _TrialCache(cache_json_path) if use_trial_cache else None
    run_ctx = {
        "n_rounds": n_rounds, "n_clients": n_clients, "partition": partition,
        "dirichlet_alpha": dirichlet_alpha, "use_line_search": use_line_search,
        "patience": patience, "thres": thres, "alpha_init": alpha_init,
        "alpha_multiplier": alpha_multiplier, "val_frac": val_frac,
        "data_seed": data_seed,
    }

    total = len(datasets) * len(reg_types) * n_trials

    # ── Phase 1: sample every trial config up front (sequential) ────────────────
    # Deterministic per-(seed, dataset, reg_type, trial) sampling: a trial's cfg
    # depends only on its own coordinates (not on which other datasets /
    # reg_types / n_trials are in the run), so the trial cache hits across
    # subset runs, extended --n_trials, and added reg_types.
    specs = []  # (dataset, reg_type, setting_idx, trial, cfg, trial_seed, suffix,
                #  is_fixed, key, cached)
    for reg_type in reg_types:
        idx_map = index_maps[reg_type]
        for dataset in datasets:
            if dataset not in idx_map:
                print(f"[SKIP] {dataset} not found in TSC_FL_settings_{reg_type}.json")
                continue
            setting_idx = idx_map[dataset]

            fixed = fixed_reservoir.get(reg_type, {}).get(dataset) if fix_reservoir else None
            if fix_reservoir and fixed is None:
                print(f"[WARN] no centralized config for {dataset}/{reg_type}; "
                      f"searching reservoir params for this dataset")

            for trial in range(n_trials):
                trial_rng = random.Random(f"{seed}/{dataset}/{reg_type}/{trial}")
                cfg = _sample_config(reg_type, trial_rng)
                if fixed is not None:
                    cfg.update(fixed)   # freeze sr/lr/input_scaling, search reg_param only
                trial_seed = seed + trial
                key    = _trial_key(dataset, reg_type, cfg, trial_seed, run_ctx)
                cached = trial_cache.get(key) if trial_cache else None
                specs.append((dataset, reg_type, setting_idx, trial, cfg,
                              trial_seed, f"search_t{trial}", fixed is not None,
                              key, cached))

    if trial_cache:
        n_hit = sum(1 for s in specs if s[9] is not None)
        print(f"[TrialCache] {n_hit}/{len(specs)} trials served from cache")

    # ── Phase 2: run the uncached trials (parallel when n_jobs>1) ───────────────
    task_args = [
        (dataset, reg_type, n_rounds, n_clients, trial_seed, setting_idx,
         partition, dirichlet_alpha, use_cache, suffix, use_line_search,
         patience, use_gpu, thres, alpha_init, alpha_multiplier, val_frac,
         data_seed, cfg)
        for (dataset, reg_type, setting_idx, trial, cfg, trial_seed, suffix,
             _, _key, cached) in specs
        if cached is None
    ]
    pending_res = iter(parallel_map(_fl_trial_worker, task_args, n_jobs,
                                    progress="  FL trials"))

    # ── Phase 3: record results in submit order (deterministic) ─────────────────
    for idx, (dataset, reg_type, setting_idx, trial, cfg, trial_seed, suffix,
              is_fixed, key, cached) in enumerate(specs, start=1):
        if cached is not None:
            res = cached
        else:
            res = next(pending_res)
        if isinstance(res, tuple) and res and res[0] == "__error__":
            print(f"\n[{idx}/{total}] Trial {trial+1}/{n_trials} — "
                  f"{dataset}/{reg_type}  [ERROR] {res[1]}")
            val_acc = avg_acc = avg_sparsity = \
                avg_macro_f1 = avg_balanced_acc = float("nan")
        else:
            val_acc, avg_acc, avg_sparsity, avg_macro_f1, avg_balanced_acc = res
            if trial_cache and cached is None and not (val_acc != val_acc):
                trial_cache.set(key, res)

        # Selection score uses the VALIDATION accuracy (test stays report-only).
        score = _composite_score(val_acc, avg_sparsity, sparsity_weight)
        is_best = tracker.update((dataset, reg_type), score, {
            "val_acc":          val_acc,
            "avg_acc":          avg_acc,
            "avg_sparsity":     avg_sparsity,
            "avg_macro_f1":     avg_macro_f1,
            "avg_balanced_acc": avg_balanced_acc,
            "cfg":              deepcopy(cfg),
            "trial":            trial,
        })

        # Only announce a trial when it sets a new best score — the full per-trial
        # record still goes to the CSV below, so nothing is lost.
        if is_best:
            print(f"\n[{idx}/{total}] Trial {trial+1}/{n_trials} — [NEW BEST] "
                  f"{dataset}/{reg_type}"
                  + ("  [reservoir fixed]" if is_fixed else "")
                  + ("  [CACHED]" if cached is not None else ""))
            print("  params:", {k: (f"{v:.4g}" if isinstance(v, float) else v)
                                + (" (fixed)" if k in _RESERVOIR_KEYS and is_fixed else "")
                                for k, v in cfg.items()})
            print(f"  → val_acc={val_acc:.2f}%  test_acc={avg_acc:.2f}%  "
                  f"avg_sparsity={avg_sparsity:.1f}%  score={score:.2f}")

        append_csv_row(csv_path, _CSV_FIELDS, {
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset":          dataset,
            "reg_type":         reg_type,
            "trial":            trial,
            "score":            f"{score:.4f}",
            "val_acc":          f"{val_acc:.4f}",
            "avg_acc":          f"{avg_acc:.4f}",
            "avg_sparsity":     f"{avg_sparsity:.4f}",
            "avg_macro_f1":     f"{avg_macro_f1:.4f}",
            "avg_balanced_acc": f"{avg_balanced_acc:.4f}",
            # Uniform SL1 schedule columns (constant across the run).
            "thres":            thres,
            "alpha_init":       alpha_init,
            "alpha_multiplier": alpha_multiplier,
            **{k: (f"{v:.6g}" if isinstance(v, float) else v)
               for k, v in cfg.items()},
        })

    # ── Federated CE+IMP baseline at each dataset's SL1 sparsity ────────────────
    # Dense-CE FL run, then federated IMP prune-refit to the sparsity SL1 reached
    # (kept federated, so it is a fair head-to-head).  Recorded as reg_type
    # "ce_imp"; skipped when SL1 was not searched.
    if "sl1" in reg_types:
        none_idx = _build_index_map("none")
        for dataset in datasets:
            best = tracker.get((dataset, "sl1"))
            if best is None or dataset not in none_idx:
                continue
            _s, rec = best
            target_sp = rec["avg_sparsity"]
            reservoir_cfg = {k: rec["cfg"][k] for k in _RESERVOIR_KEYS
                             if k in rec["cfg"]}
            print(f"\n[baseline] {dataset}/ce_imp  (dense CE + federated IMP "
                  f"→ {target_sp:.1f}% sparse)")
            try:
                # Same seed + partition as the best sl1 trial, so the matched-
                # sparsity comparison runs on the identical reservoir and split.
                res = run_experiment(
                    reg_type="none", n_rounds=n_rounds, n_clients=n_clients,
                    seed=seed + rec["trial"], setting_idx=none_idx[dataset],
                    partition=partition,
                    dirichlet_alpha=dirichlet_alpha, val_frac=val_frac,
                    data_seed=data_seed, use_cache=use_cache,
                    exp_suffix="ce_imp", use_line_search=use_line_search,
                    patience=patience, use_gpu=use_gpu, thres=thres,
                    alpha_init=alpha_init, alpha_multiplier=alpha_multiplier,
                    param_overrides=reservoir_cfg, imp_target_sp=target_sp,
                    verbose=False,
                ).get(dataset, {})
            except Exception as exc:                              # noqa: BLE001
                print(f"  [ERROR: {exc}]")
                res = {}
            imp_acc = res.get("imp_acc", float("nan"))
            imp_sp  = res.get("imp_sparsity", float("nan"))
            imp_f1  = res.get("imp_macro_f1", float("nan"))
            imp_bal = res.get("imp_balanced_acc", float("nan"))
            print(f"  → acc={imp_acc:.2f}%  f1={imp_f1:.2f}%  bal={imp_bal:.2f}%  "
                  f"sparsity={imp_sp:.1f}%")
            append_csv_row(csv_path, _CSV_FIELDS, {
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset":          dataset,
                "reg_type":         "ce_imp",
                "trial":            "baseline",
                "avg_acc":          f"{imp_acc:.4f}",
                "avg_sparsity":     f"{imp_sp:.4f}",
                "avg_macro_f1":     f"{imp_f1:.4f}",
                "avg_balanced_acc": f"{imp_bal:.4f}",
            })

    _report(tracker, sparsity_weight)
    _save_best_json(tracker, reg_types, index_maps, json_dir,
                    thres, alpha_init, alpha_multiplier, sparsity_weight)
    print(f"\nAll results saved → {csv_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    _all_datasets = sorted(config.load_tsc_dataset_meta().keys())

    p = argparse.ArgumentParser(description="FedSL1ESN hyperparameter search")
    p.add_argument("--datasets",  nargs="+", default=_all_datasets)
    p.add_argument("--reg_types", nargs="+", default=["none", "l2", "sl1"],
                   choices=["none", "l2", "sl1"])
    p.add_argument("--n_trials",  type=int, default=20)
    p.add_argument("--n_rounds",  type=int, default=10,
                   help="FL rounds per trial — keep small for fast search (default: 10)")
    p.add_argument("--n_clients", type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--no_cache",  action="store_true")
    p.add_argument("--partition", default="iid", choices=["iid", "dirichlet"],
                   help="Client data split: 'iid' (default) or 'dirichlet' (non-iid label skew)")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Dirichlet concentration for --partition dirichlet; "
                        "smaller = stronger label skew (default: 0.5)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of each client's train shard pooled into the "
                        "validation set the search selects on (test stays "
                        "report-only; default: 0.2)")
    p.add_argument("--no_trial_cache", action="store_true",
                   help="Disable the persistent per-trial result cache")
    p.add_argument("--fix_reservoir", action="store_true",
                   help="Freeze sr/lr/input_scaling to the centralized best config "
                        "(tsc_centralized_best_<reg>.json) and search reg_param only")
    p.add_argument("--centralized_json_dir", default="./result",
                   help="Directory holding tsc_centralized_best_<reg>.json "
                        "(used by --fix_reservoir; default: ./result)")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout math; auto-falls back to CPU")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--line_search", dest="line_search", action="store_true", default=True,
                   help="Enable Armijo line search on the server Newton step (default: on)")
    p.add_argument("--no_line_search", dest="line_search", action="store_false",
                   help="Disable the server Newton-step line search (use fixed step α=1)")
    p.add_argument("--patience",  type=int, default=3)
    p.add_argument("--csv_path",  default="./result/tsc_fl_search.csv")
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight of sparsity in composite score (0=pure acc, 1=pure sparsity)")
    p.add_argument("--json_dir",  default="./result",
                   help="Directory for tsc_fl_best_<reg_type>.json files")
    # ── SL1 sparsification schedule (uniform across datasets; NOT searched) ──────
    p.add_argument("--thres", type=float, default=config.sl1_defaults.THRES,
                   help=f"SL1 soft-threshold magnitude (default: {config.sl1_defaults.THRES})")
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help=f"Initial SmoothL1 alpha (default: {config.sl1_defaults.ALPHA_INIT})")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help=f"Per-round alpha growth factor (default: {config.sl1_defaults.ALPHA_MULTIPLIER})")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel worker processes for trials (1=serial [default], "
                        "-1=all cores). Each trial is a full FL experiment, so keep "
                        "this modest to avoid CPU/RAM oversubscription. Forced to 1 "
                        "under --gpu (single-device contention).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    search(
        datasets        = args.datasets,
        reg_types       = args.reg_types,
        n_trials        = args.n_trials,
        n_rounds        = args.n_rounds,
        n_clients       = args.n_clients,
        seed            = args.seed,
        use_cache       = not args.no_cache,
        use_line_search = args.line_search,
        patience        = args.patience,
        csv_path        = args.csv_path,
        partition       = args.partition,
        dirichlet_alpha = args.dirichlet_alpha,
        fix_reservoir        = args.fix_reservoir,
        centralized_json_dir = args.centralized_json_dir,
        use_gpu         = args.use_gpu,
        sparsity_weight = args.sparsity_weight,
        json_dir        = args.json_dir,
        thres            = args.thres,
        alpha_init       = args.alpha_init,
        alpha_multiplier = args.alpha_multiplier,
        val_frac         = args.val_frac,
        use_trial_cache  = not args.no_trial_cache,
        n_jobs           = args.n_jobs,
    )
