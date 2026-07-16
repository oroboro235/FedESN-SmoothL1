# tsc_centralized_noniid_eval.py — centralized ESN + regularization on non-iid data.
#
# Purpose
# -------
# Local-only baseline for the FedSL1ESN study: how well can a *centralized* ESN
# readout learn when its training data is a single non-iid (label-skewed) shard,
# with NO collaboration between clients?  This is the natural lower bound to
# contrast against FedSL1ESN, and it isolates "can the model learn from skewed
# local data" from the federated aggregation.
#
# Pipeline per (dataset, reg_type)
# --------------------------------
#   1. SEARCH  — random hyperparameter search on the *pooled* train set
#                (stratified train/val split).  Stable, so the non-iid effect in
#                step 3 is not confounded by per-shard tuning noise.  With
#                --fix_reservoir, sr/lr/input_scaling are frozen to the
#                centralized best config and only reg_param is searched.
#   2. POOLED  — train the best config on the FULL train set → test_acc (upper
#                bound reference).
#   3. LOCAL   — Dirichlet-partition the train set into n_clients non-iid shards
#                (identical partitioner to tsc_fl_eval.py for fair comparison),
#                train the best config independently on each shard (sharing one
#                reservoir seed, mirroring FL's shared Win/W), evaluate every
#                shard on the GLOBAL test set → mean/std/min/max accuracy.
#
# All heavy lifting reuses tsc_centralized_search.py's helpers; the Dirichlet
# split reuses tsc_fl_eval._dirichlet_train_indices.
#
# Results:
#   ./result/tsc_centralized_noniid.csv     — one row per (dataset, reg_type)
#
# Usage:
#   python tsc_centralized_noniid_eval.py
#   python tsc_centralized_noniid_eval.py --datasets har --reg_types sl1 \
#       --dirichlet_alpha 0.3 --n_clients 5
#   python tsc_centralized_noniid_eval.py --fix_reservoir --n_trials 10

import argparse
import json
import os
import random
import time

import numpy as np

import config
from data_loader import one_hot, standardize
from utils import init_csv, append_csv_row

# Reuse the centralized pipeline helpers (DRY).
from tsc_centralized_search import (
    _load_dataset, _ensure_3d, _stratified_split, _run_trial,
    _sample_config, _composite_score, _load_dataset_meta,
)
# Reuse the EXACT same non-iid partitioner the FL experiments use, so the
# centralized baseline and FedSL1ESN see comparable label skew.
from tsc_fl_eval import _dirichlet_train_indices


_RESERVOIR_KEYS = ("sr", "lr", "input_scaling")


# ─── Fixed reservoir params from centralized search ──────────────────────────

def _load_fixed_reservoir(reg_type: str, json_dir: str) -> dict:
    """Load {dataset: {sr, lr, input_scaling}} from the centralized best JSON."""
    path = os.path.join(json_dir, f"tsc_centralized_best_{reg_type}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--fix_reservoir needs {path}; run tsc_centralized_search.py "
            f"for reg_type='{reg_type}' first (or pass --centralized_json_dir)")
    with open(path) as f:
        return {s["dataset"]: {k: s[k] for k in _RESERVOIR_KEYS}
                for s in json.load(f)}


# ─── Config assembly ─────────────────────────────────────────────────────────

def _make_config(rng, epochs, reg_type, thres, alpha_init, alpha_multiplier,
                 fixed: dict = None,
                 patience: int = config.sl1_defaults.PATIENCE,
                 stag_tol: float = config.sl1_defaults.STAG_TOL) -> dict:
    """Sample a config and attach the (un-searched) SL1 schedule fields.

    The sr/lr/input_scaling draws are always consumed so the reg_param RNG
    stream is unchanged whether or not the reservoir is fixed.
    """
    cfg = _sample_config(rng, epochs)
    cfg["thres"]            = thres
    cfg["alpha_init"]       = alpha_init
    cfg["alpha_multiplier"] = alpha_multiplier
    cfg["patience"]         = patience
    cfg["stag_tol"]         = stag_tol
    if fixed is not None:
        cfg.update(fixed)   # freeze sr/lr/input_scaling
    return cfg


# ─── Search on the pooled train set ──────────────────────────────────────────

def _search_best_config(dataset, reg_type, meta, Xtr_full, ytr_full_int,
                        n_trials, epochs, seed, sparsity_weight,
                        thres, alpha_init, alpha_multiplier,
                        fixed, use_gpu, rng,
                        patience=config.sl1_defaults.PATIENCE,
                        stag_tol=config.sl1_defaults.STAG_TOL) -> dict:
    """Random search on a stratified train/val split of the pooled data.

    Returns the best cfg (by composite val score) with val metrics attached.
    """
    Xtr_sub, ytr_sub_int, Xval, yval_int = _stratified_split(
        Xtr_full, ytr_full_int, val_frac=0.2, seed=seed
    )
    n_classes  = meta["output_dim"]
    ytr_sub_oh = one_hot(ytr_sub_int, n_classes)

    best = {"score": -np.inf}
    for trial in range(n_trials):
        cfg = _make_config(rng, epochs, reg_type,
                           thres, alpha_init, alpha_multiplier, fixed,
                           patience=patience, stag_tol=stag_tol)
        try:
            val_acc, val_sp = _run_trial(
                Xtr_sub, ytr_sub_oh, Xval, yval_int,
                meta, reg_type, cfg, seed=seed, use_gpu=use_gpu,
            )
        except Exception as exc:
            print(f"      [trial {trial}] ERROR {exc}")
            continue
        score = _composite_score(val_acc, val_sp, sparsity_weight)
        tag = "  [reservoir fixed]" if fixed is not None else ""
        print(f"      trial {trial+1}/{n_trials}{tag}  "
              f"reg={cfg['reg_param']:.4g}  val_acc={val_acc:.2f}%  "
              f"val_sp={val_sp:.1f}%  score={score:.2f}")
        if score > best["score"]:
            best = {"score": score, "val_acc": val_acc, "val_sparsity": val_sp,
                    "cfg": cfg}
    if "cfg" not in best:
        raise RuntimeError(f"all search trials failed for {dataset}/{reg_type}")
    return best


# ─── Local-only non-iid evaluation ───────────────────────────────────────────

def _eval_local_only(meta, reg_type, cfg, Xtr_full, ytr_full_int, Xte, yte_int,
                     n_clients, alpha, seed, use_gpu) -> dict:
    """Train the given config on each non-iid shard; evaluate on global test.

    Also trains a pooled (full-train) reference for the upper bound.  All trials
    share one reservoir seed so only the data differs (mirrors FL's shared Win/W).
    """
    n_classes = meta["output_dim"]

    # Pooled upper-bound reference.
    pooled_acc, pooled_sp = _run_trial(
        Xtr_full, one_hot(ytr_full_int, n_classes), Xte, yte_int,
        meta, reg_type, cfg, seed=seed, use_gpu=use_gpu,
    )

    # Non-iid shards (same partitioner as the FL pipeline).
    part_rng = np.random.default_rng(seed)
    shards   = _dirichlet_train_indices(
        ytr_full_int, n_clients, alpha, min_per_client=10, rng=part_rng
    )

    per_client = []
    for cid, idx in enumerate(shards):
        Xc = Xtr_full[idx]
        yc = ytr_full_int[idx]
        hist = np.bincount(yc, minlength=n_classes)
        acc, sp = _run_trial(
            Xc, one_hot(yc, n_classes), Xte, yte_int,
            meta, reg_type, cfg, seed=seed, use_gpu=use_gpu,
        )
        per_client.append({"client": cid, "n": len(idx), "acc": acc,
                           "sparsity": sp, "hist": hist.tolist()})
        print(f"      client {cid} (n={len(idx):4d}, hist={hist.tolist()})  "
              f"acc={acc:.2f}%  sparsity={sp:.1f}%")

    accs = np.array([c["acc"] for c in per_client])
    sps  = np.array([c["sparsity"] for c in per_client])
    return {
        "pooled_acc":      pooled_acc,
        "pooled_sparsity": pooled_sp,
        "local_mean_acc":  float(accs.mean()),
        "local_std_acc":   float(accs.std()),
        "local_min_acc":   float(accs.min()),
        "local_max_acc":   float(accs.max()),
        "local_mean_sparsity": float(sps.mean()),
        "per_client":      per_client,
    }


# ─── CSV output ──────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "dirichlet_alpha", "n_clients",
    "sr", "lr", "input_scaling", "reg_param", "thres",
    "alpha_init", "alpha_multiplier", "epochs",
    "search_val_acc", "search_val_sparsity",
    "pooled_acc", "pooled_sparsity",
    "local_mean_acc", "local_std_acc", "local_min_acc", "local_max_acc",
    "local_mean_sparsity",
]


# ─── Main driver ─────────────────────────────────────────────────────────────

def run(datasets, reg_types, n_clients, dirichlet_alpha, n_trials, epochs,
        seed, use_cache, use_gpu, sparsity_weight, csv_path,
        thres, alpha_init, alpha_multiplier,
        fix_reservoir, centralized_json_dir,
        patience=config.sl1_defaults.PATIENCE,
        stag_tol=config.sl1_defaults.STAG_TOL):
    rng = random.Random(seed)
    init_csv(csv_path, _CSV_FIELDS)
    dataset_meta = _load_dataset_meta()

    fixed_reservoir = (
        {rt: _load_fixed_reservoir(rt, centralized_json_dir) for rt in reg_types}
        if fix_reservoir else {}
    )

    summary = []
    for dataset in datasets:
        if dataset not in dataset_meta:
            print(f"[SKIP] '{dataset}' not in {config.paths.tsc_dataset_meta}")
            continue
        meta = dataset_meta[dataset]
        print(f"\n{'='*68}")
        print(f"Dataset: {dataset}  (classes={meta['output_dim']}, "
              f"units={meta['units']})  |  Dirichlet(alpha={dirichlet_alpha}), "
              f"{n_clients} clients")
        print(f"{'='*68}")

        try:
            Xtr_raw, ytr_int, Xte_raw, yte_int = _load_dataset(dataset, use_cache)
        except Exception as exc:
            print(f"  [ERROR] loading {dataset}: {exc}")
            continue

        Xtr_raw = _ensure_3d(Xtr_raw)
        Xte_raw = _ensure_3d(Xte_raw)
        # Global standardization (stats from train only) — matches tsc_fl_eval.py,
        # which standardizes before partitioning.
        Xtr, Xte = standardize(Xtr_raw, Xte_raw)

        for reg_type in reg_types:
            fixed = fixed_reservoir.get(reg_type, {}).get(dataset) if fix_reservoir else None
            if fix_reservoir and fixed is None:
                print(f"  [WARN] no centralized config for {dataset}/{reg_type}; "
                      f"searching reservoir params too")

            print(f"\n  reg_type = {reg_type}")
            print(f"    [search] {n_trials} trials on pooled train set"
                  + ("  (reservoir fixed)" if fixed is not None else ""))
            try:
                best = _search_best_config(
                    dataset, reg_type, meta, Xtr, ytr_int,
                    n_trials, epochs, seed, sparsity_weight,
                    thres, alpha_init, alpha_multiplier, fixed, use_gpu, rng,
                    patience=patience, stag_tol=stag_tol,
                )
            except Exception as exc:
                print(f"    [ERROR] search failed: {exc}")
                continue
            cfg = best["cfg"]
            print(f"    [best] val_acc={best['val_acc']:.2f}%  "
                  f"reg_param={cfg['reg_param']:.4g}  "
                  f"sr={cfg['sr']:.3g} lr={cfg['lr']:.3g} is={cfg['input_scaling']:.3g}")

            print(f"    [eval] local-only on {n_clients} non-iid shards")
            res = _eval_local_only(
                meta, reg_type, cfg, Xtr, ytr_int, Xte, yte_int,
                n_clients, dirichlet_alpha, seed, use_gpu,
            )
            print(f"    → pooled(upper bound) acc={res['pooled_acc']:.2f}%   "
                  f"local-only acc={res['local_mean_acc']:.2f}% "
                  f"± {res['local_std_acc']:.2f} "
                  f"(min {res['local_min_acc']:.1f} / max {res['local_max_acc']:.1f})")

            append_csv_row(csv_path, _CSV_FIELDS, {
                "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset":          dataset,
                "reg_type":         reg_type,
                "dirichlet_alpha":  dirichlet_alpha,
                "n_clients":        n_clients,
                "sr":               f"{cfg['sr']:.6g}",
                "lr":               f"{cfg['lr']:.6g}",
                "input_scaling":    f"{cfg['input_scaling']:.6g}",
                "reg_param":        f"{cfg['reg_param']:.6g}",
                "thres":            cfg["thres"],
                "alpha_init":       cfg["alpha_init"],
                "alpha_multiplier": cfg["alpha_multiplier"],
                "epochs":           cfg["epochs"],
                "search_val_acc":      f"{best['val_acc']:.4f}",
                "search_val_sparsity": f"{best['val_sparsity']:.4f}",
                "pooled_acc":          f"{res['pooled_acc']:.4f}",
                "pooled_sparsity":     f"{res['pooled_sparsity']:.4f}",
                "local_mean_acc":      f"{res['local_mean_acc']:.4f}",
                "local_std_acc":       f"{res['local_std_acc']:.4f}",
                "local_min_acc":       f"{res['local_min_acc']:.4f}",
                "local_max_acc":       f"{res['local_max_acc']:.4f}",
                "local_mean_sparsity": f"{res['local_mean_sparsity']:.4f}",
            })
            summary.append((dataset, reg_type, res))

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"CENTRALIZED ESN ON NON-IID DATA — SUMMARY "
          f"(Dirichlet alpha={dirichlet_alpha}, {n_clients} clients)")
    print(f"{'='*68}")
    print(f"  {'dataset':28s} {'reg':5s} {'pooled':>8s} {'local(mean±std)':>20s}")
    for dataset, reg_type, res in summary:
        print(f"  {dataset:28s} {reg_type:5s} {res['pooled_acc']:7.2f}% "
              f"{res['local_mean_acc']:8.2f}% ± {res['local_std_acc']:.2f}")
    print(f"\nResults saved → {csv_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    _all = sorted(config.load_tsc_dataset_meta().keys())

    p = argparse.ArgumentParser(
        description="Centralized ESN + regularization on non-iid (local-only) data")
    p.add_argument("--datasets",  nargs="+", default=_all)
    p.add_argument("--reg_types", nargs="+", default=["none", "l2", "sl1"],
                   choices=["none", "l2", "sl1"])
    p.add_argument("--n_clients", type=int, default=5,
                   help="Number of non-iid shards (local-only learners)")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Dirichlet concentration; smaller = stronger label skew")
    p.add_argument("--n_trials", type=int, default=15,
                   help="Random search trials on the pooled train set (default: 15)")
    p.add_argument("--epochs",   type=int, default=500)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU)
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    p.add_argument("--sparsity_weight", type=float, default=0.0,
                   help="Weight of sparsity in the search score (0=pure acc)")
    p.add_argument("--csv_path", default="./result/tsc_centralized_noniid.csv")
    p.add_argument("--fix_reservoir", action="store_true",
                   help="Freeze sr/lr/input_scaling to tsc_centralized_best_<reg>.json "
                        "and search reg_param only")
    p.add_argument("--centralized_json_dir", default="./result",
                   help="Directory holding tsc_centralized_best_<reg>.json")
    p.add_argument("--thres", type=float, default=config.sl1_defaults.THRES)
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT)
    p.add_argument("--alpha_multiplier", type=float,
                   default=config.sl1_defaults.ALPHA_MULTIPLIER)
    p.add_argument("--patience", type=int, default=config.sl1_defaults.PATIENCE,
                   help="Newton stagnation early-stop window; 0 disables "
                        f"(default: {config.sl1_defaults.PATIENCE})")
    p.add_argument("--stag_tol", type=float, default=config.sl1_defaults.STAG_TOL,
                   help=f"Stagnation tolerance (default: {config.sl1_defaults.STAG_TOL})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        datasets             = args.datasets,
        reg_types            = args.reg_types,
        n_clients            = args.n_clients,
        dirichlet_alpha      = args.dirichlet_alpha,
        n_trials             = args.n_trials,
        epochs               = args.epochs,
        seed                 = args.seed,
        use_cache            = not args.no_cache,
        use_gpu              = args.use_gpu,
        sparsity_weight      = args.sparsity_weight,
        csv_path             = args.csv_path,
        thres                = args.thres,
        alpha_init           = args.alpha_init,
        alpha_multiplier     = args.alpha_multiplier,
        patience             = args.patience,
        stag_tol             = args.stag_tol,
        fix_reservoir        = args.fix_reservoir,
        centralized_json_dir = args.centralized_json_dir,
    )
