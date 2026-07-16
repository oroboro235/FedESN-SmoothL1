# tsc_noniid_compare.py — pooled vs local-only vs FedSL1ESN, at one Dirichlet alpha.
#
# Builds the head-to-head table that quantifies how much of the non-iid penalty
# federated collaboration (FedSL1ESN) recovers over training alone (local-only):
#
#     recovery% = (fl_acc − local_acc) / (pooled_acc − local_acc) × 100
#
#   pooled (upper bound) ─────────────────────────────────────●  100%
#                                                  fl_acc ─────●
#                                            local_acc ──●  0%   (no collaboration)
#
# Data sources
# ------------
#   pooled / local-only : read from result/tsc_centralized_noniid.csv
#                         (produced by tsc_centralized_noniid_eval.py — run it
#                          first at the SAME --dirichlet_alpha / --n_clients).
#   FedSL1ESN           : run live via tsc_fl_eval.main() with the same non-iid
#                         partition (there is no stored FL results CSV to read).
#
# Results:
#   ./result/tsc_noniid_compare.csv  — one row per (dataset, reg_type)
#
# Usage:
#   python tsc_noniid_compare.py --datasets har --dirichlet_alpha 0.3 --n_clients 5
#   python tsc_noniid_compare.py --reg_types l2 sl1 --n_rounds 30

import argparse
import csv
import json
import os
import time

import config
from utils import init_csv, append_csv_row
from tsc_fl_eval import main as run_fl, _read_settings


# ─── Read centralized (pooled + local-only) results ──────────────────────────

def _load_centralized(csv_path: str, dataset: str, reg_type: str,
                      alpha: float, n_clients: int):
    """Return the most recent matching centralized-noniid row, or None.

    Matches on (dataset, reg_type, dirichlet_alpha, n_clients) so the comparison
    is apples-to-apples with the FL run.
    """
    if not os.path.exists(csv_path):
        return None
    match = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["dataset"] == dataset and row["reg_type"] == reg_type
                    and abs(float(row["dirichlet_alpha"]) - alpha) < 1e-9
                    and int(row["n_clients"]) == n_clients):
                match = row   # keep last (most recent) match
    return match


# ─── FL run ──────────────────────────────────────────────────────────────────

def _run_fl(dataset, reg_type, alpha, n_clients, n_rounds, seed,
            use_cache, use_gpu):
    """Run FedSL1ESN on the non-iid partition; return (acc, sparsity) or None."""
    idx_map = {s["dataset"]: i for i, s in enumerate(_read_settings(reg_type))}
    if dataset not in idx_map:
        print(f"  [WARN] {dataset} not in TSC_FL_settings_{reg_type}.json — skipping FL")
        return None
    results = run_fl(
        reg_type        = reg_type,
        n_rounds        = n_rounds,
        n_clients       = n_clients,
        seed            = seed,
        setting_idx     = idx_map[dataset],
        partition       = "dirichlet",
        dirichlet_alpha = alpha,
        use_cache       = use_cache,
        use_gpu         = use_gpu,
        exp_suffix      = f"compare_a{alpha}",
        verbose         = False,   # silence per-round trace; compare drives many runs
    )
    r = results.get(dataset)
    return (r["acc"], r["sparsity"]) if r else None


# ─── Recovery metric ─────────────────────────────────────────────────────────

def _recovery_pct(pooled, local, fl):
    """Fraction of the non-iid gap (pooled − local) that FL recovers, in %."""
    gap = pooled - local
    if abs(gap) < 1e-9:
        return None
    return (fl - local) / gap * 100.0


# ─── CSV output ──────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "dataset", "reg_type", "dirichlet_alpha", "n_clients", "n_rounds",
    "pooled_acc", "local_mean_acc", "local_std_acc", "fl_acc", "recovery_pct",
    "pooled_sparsity", "local_mean_sparsity", "fl_sparsity",
]


# ─── Main ────────────────────────────────────────────────────────────────────

def run(datasets, reg_types, alpha, n_clients, n_rounds, seed,
        use_cache, use_gpu, centralized_csv, out_csv):
    init_csv(out_csv, _CSV_FIELDS)
    rows = []

    for dataset in datasets:
        for reg_type in reg_types:
            cen = _load_centralized(centralized_csv, dataset, reg_type, alpha, n_clients)
            if cen is None:
                print(f"[SKIP] no centralized row for {dataset}/{reg_type} "
                      f"(alpha={alpha}, n_clients={n_clients}); "
                      f"run tsc_centralized_noniid_eval.py first")
                continue

            pooled       = float(cen["pooled_acc"])
            local_mean   = float(cen["local_mean_acc"])
            local_std    = float(cen["local_std_acc"])
            pooled_sp    = float(cen["pooled_sparsity"])
            local_sp     = float(cen["local_mean_sparsity"])

            print(f"\n{'─'*60}\nRunning FL: {dataset}/{reg_type} "
                  f"(alpha={alpha}, {n_clients} clients × {n_rounds} rounds)\n{'─'*60}")
            fl = _run_fl(dataset, reg_type, alpha, n_clients, n_rounds, seed,
                         use_cache, use_gpu)
            if fl is None:
                continue
            fl_acc, fl_sp = fl
            rec = _recovery_pct(pooled, local_mean, fl_acc)

            rows.append({
                "dataset": dataset, "reg_type": reg_type,
                "pooled": pooled, "local_mean": local_mean, "local_std": local_std,
                "fl": fl_acc, "rec": rec,
                "pooled_sp": pooled_sp, "local_sp": local_sp, "fl_sp": fl_sp,
            })
            append_csv_row(out_csv, _CSV_FIELDS, {
                "timestamp":           time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset":             dataset,
                "reg_type":            reg_type,
                "dirichlet_alpha":     alpha,
                "n_clients":           n_clients,
                "n_rounds":            n_rounds,
                "pooled_acc":          f"{pooled:.4f}",
                "local_mean_acc":      f"{local_mean:.4f}",
                "local_std_acc":       f"{local_std:.4f}",
                "fl_acc":              f"{fl_acc:.4f}",
                "recovery_pct":        "" if rec is None else f"{rec:.2f}",
                "pooled_sparsity":     f"{pooled_sp:.4f}",
                "local_mean_sparsity": f"{local_sp:.4f}",
                "fl_sparsity":         f"{fl_sp:.4f}",
            })

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"NON-IID COMPARISON — Dirichlet(alpha={alpha}), {n_clients} clients, "
          f"FL {n_rounds} rounds")
    print(f"  recovery% = how much of the (pooled − local-only) gap FedSL1ESN closes")
    print(f"{'='*82}")
    hdr = (f"  {'dataset':22s} {'reg':4s} "
           f"{'pooled':>8s} {'local-only':>14s} {'FedSL1ESN':>10s} {'recovery':>9s}"
           f"  ││ {'sparsity: pooled':>16s} {'local':>7s} {'FL':>7s}")
    print(hdr)
    print(f"  {'-'*len(hdr)}")
    for r in rows:
        rec = "  n/a" if r["rec"] is None else f"{r['rec']:6.1f}%"
        print(f"  {r['dataset']:22s} {r['reg_type']:4s} "
              f"{r['pooled']:7.2f}% {r['local_mean']:7.2f}±{r['local_std']:4.1f} "
              f"{r['fl']:9.2f}% {rec:>9s}"
              f"  ││ {r['pooled_sp']:15.1f}% {r['local_sp']:6.1f}% {r['fl_sp']:6.1f}%")
    print(f"\nSaved → {out_csv}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare pooled / local-only / FedSL1ESN on non-iid data")
    p.add_argument("--datasets",  nargs="+", required=True)
    p.add_argument("--reg_types", nargs="+", default=["none", "l2", "sl1"],
                   choices=["none", "l2", "sl1"])
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Must match the centralized-noniid run (default: 0.5)")
    p.add_argument("--n_clients", type=int, default=5,
                   help="Must match the centralized-noniid run (default: 5)")
    p.add_argument("--n_rounds",  type=int, default=20,
                   help="FL communication rounds (default: 20)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--no_cache",  action="store_true")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU)
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    p.add_argument("--centralized_csv", default="./result/tsc_centralized_noniid.csv")
    p.add_argument("--out_csv", default="./result/tsc_noniid_compare.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        datasets        = args.datasets,
        reg_types       = args.reg_types,
        alpha           = args.dirichlet_alpha,
        n_clients       = args.n_clients,
        n_rounds        = args.n_rounds,
        seed            = args.seed,
        use_cache       = not args.no_cache,
        use_gpu         = args.use_gpu,
        centralized_csv = args.centralized_csv,
        out_csv         = args.out_csv,
    )
