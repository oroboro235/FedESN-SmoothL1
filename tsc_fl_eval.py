# tsc_fl_eval.py — unified entry point for all FedSL1ESN (FL classification) experiments.
#
# Federated training + evaluation: trains the readout via Newton-step aggregation
# across clients and reports per-round accuracy / sparsity.  Differences between
# experiments are CLI arguments.  Pair with tsc_fl_search.py for hparam search.
#
# Quick-start examples:
#   python tsc_fl_eval.py                              # no-reg, 5 rounds, all datasets
#   python tsc_fl_eval.py --reg_types sl1              # SL1, 5 rounds
#   python tsc_fl_eval.py --reg_types all --datasets jpv har   # 3 reg × 2 datasets
#   python tsc_fl_eval.py --reg_types sl1 --n_rounds 20 --no_cache  # long run
#   python tsc_fl_eval.py --setting_idx 2              # only the 3rd dataset entry
#
# All per-dataset hyperparameters come from configs/TSC_FL_settings_<reg_type>.json.
# The server initialises a shared reservoir (Win, W); only Wout is federated.

import argparse
import json
import os
from typing import Optional

import numpy as np

import config
import funcs
import pruning
from client import Client_TSC
from server import Server_FedAvg
from data_loader import read_data, one_hot, standardize


# ─── CLI arguments ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedSL1ESN experiment runner")
    p.add_argument("--reg_types", nargs="+", default=["none"],
                   choices=["none", "l2", "sl1", "all"],
                   help="Regularisation type(s) to evaluate — each runs against "
                        "its own TSC_FL_settings_<reg_type>.json; 'all' expands "
                        "to none l2 sl1 (default: none)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset names to evaluate (default: every entry in the "
                        "settings JSON); names missing from a reg_type's file "
                        "are skipped with a warning")
    p.add_argument("--n_rounds", type=int, default=5,
                   help="Number of federated learning rounds (default: 5)")
    p.add_argument("--n_clients", type=int, default=5,
                   help="Number of FL clients (default: 5)")
    p.add_argument("--seed", type=int, default=1234,
                   help="Global random seed (default: 1234)")
    p.add_argument("--global_lr", type=float, default=1.0,
                   help="Server learning rate for Wout update (default: 1.0)")
    p.add_argument("--local_lr", type=float, default=1.0,
                   help="Client learning rate for Wout update (default: 1.0)")
    p.add_argument("--local_epochs", type=int, default=100,
                   help="Local training epochs per round (default: 100)")
    # ── SL1 sparsification schedule (uniform across all datasets; overrides JSON) ─
    p.add_argument("--thres", type=float, default=config.sl1_defaults.THRES,
                   help="SL1 soft-threshold magnitude; |w| < thres set to 0 "
                        f"(uniform across datasets, overrides JSON; default: {config.sl1_defaults.THRES})")
    p.add_argument("--alpha_init", type=float, default=config.sl1_defaults.ALPHA_INIT,
                   help="Initial SmoothL1 smoothing parameter alpha "
                        f"(default: {config.sl1_defaults.ALPHA_INIT})")
    p.add_argument("--alpha_multiplier", type=float, default=config.sl1_defaults.ALPHA_MULTIPLIER,
                   help="Per-round geometric growth factor for alpha "
                        f"(default: {config.sl1_defaults.ALPHA_MULTIPLIER})")
    p.add_argument("--setting_idx", type=int, default=None,
                   help="Run only this 0-based index from the JSON settings list "
                        "(wins over --datasets; omit to run all entries)")
    p.add_argument("--no_cache", action="store_true",
                   help="Reload data from disk even if a cache .npz already exists")
    # ── Client data partition ──────────────────────────────────────────────────
    p.add_argument("--partition", default="iid", choices=["iid", "dirichlet"],
                   help="How to split train data across clients: 'iid' (bootstrap, "
                        "default) or 'dirichlet' (label-skew non-iid)")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Dirichlet concentration for --partition dirichlet; "
                        "smaller = stronger label skew (default: 0.5)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of each client's local train shard pooled into "
                        "a server-side validation set; best round + early stop "
                        "select on val, test is report-only (0 = old "
                        "select-on-test behaviour; default: 0.2)")
    p.add_argument("--data_seed", type=int, default=None,
                   help="Seed for the client partition / val carve-out (default: "
                        "--seed). Fix it while varying --seed to keep the same "
                        "partition across runs")
    p.add_argument("--gpu", dest="use_gpu", action="store_true", default=config.USE_GPU,
                   help="Use GPU (CuPy) for the readout gradient/Hessian/loss math; "
                        "automatically falls back to CPU if CuPy/CUDA is unavailable")
    p.add_argument("--no_gpu", dest="use_gpu", action="store_false",
                   help="Force CPU even if the project default enables GPU")
    p.add_argument("--exp_suffix", default="",
                   help="Optional string appended to the experiment name")
    # ── Line search ───────────────────────────────────────────────────────────
    p.add_argument("--line_search", dest="line_search", action="store_true", default=True,
                   help="Enable Armijo backtracking line search on the server Newton step (default: on)")
    p.add_argument("--no_line_search", dest="line_search", action="store_false",
                   help="Disable the server Newton-step line search (use fixed step α=1)")
    p.add_argument("--ls_c", type=float, default=1e-4,
                   help="Armijo sufficient-decrease constant (default: 1e-4)")
    p.add_argument("--ls_rho", type=float, default=0.5,
                   help="Step shrinkage factor per backtrack iteration (default: 0.5)")
    p.add_argument("--ls_max_iter", type=int, default=20,
                   help="Maximum number of backtrack iterations (default: 20)")
    # ── Early stopping ────────────────────────────────────────────────────────
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stop after this many rounds without improvement "
                        "in BOTH accuracy and sparsity (0 = disabled, default: 0)")
    p.add_argument("--min_delta", type=float, default=0.1,
                   help="Minimum rise (in %%) of accuracy OR sparsity that counts "
                        "as a significant improvement for early stopping "
                        "(default: 0.1)")
    p.add_argument("--save_model", action="store_true",
                   help="Persist the best global model as an .npz "
                        "(off by default — keeps searches from writing one per trial)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-round training trace (only the final "
                        "summary per experiment is printed)")
    return p.parse_args()


# ─── Settings & data helpers ──────────────────────────────────────────────────

def _read_settings(reg_type: str) -> list:
    """Load the list of per-dataset hyperparameter dicts from configs/.

    Called once at the start of main(); the JSON files live in configs/
    (see config.paths.configs_path).
    """
    json_path = os.path.join(config.paths.configs_path, f"TSC_FL_settings_{reg_type}.json")
    with open(json_path, "r") as f:
        return json.load(f)


def _load_data(dataset: str, use_cache: bool) -> tuple:
    """Return (Xtr, ytr, Xte, yte), using a .npz cache when possible.

    The cache avoids re-running potentially slow preprocessing on every run.
    Set use_cache=False (--no_cache flag) to force a fresh load.
    """
    cache_file = config.paths.cache_path + dataset + ".npz"
    if use_cache and os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]

    Xtr, ytr, Xte, yte = read_data(dataset)
    np.savez(cache_file, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def _dirichlet_train_indices(y_int: np.ndarray, n_clients: int,
                             alpha: float, min_per_client: int,
                             rng: np.random.Generator) -> list:
    """Assign training indices to clients via Dirichlet label skew.

    For each class, a Dirichlet(alpha, …, alpha) draw gives the per-client
    proportions of that class's samples; samples are split *without* replacement
    so the partition is disjoint.  Small alpha → strong label skew (each client
    concentrates on a few classes); large alpha → close to IID.

    Clients left with fewer than ``min_per_client`` samples are topped up with a
    random draw (with replacement) from the global pool, so degenerate per-client
    Hessians from too-few samples are avoided — matching the IID path's guarantee.

    Returns: list of n_clients index arrays into the training set.
    """
    n_train  = len(y_int)
    classes  = np.unique(y_int)
    cl_idx: list = [[] for _ in range(n_clients)]

    for c in classes:
        c_indices = np.where(y_int == c)[0]
        rng.shuffle(c_indices)
        proportions = rng.dirichlet([alpha] * n_clients)
        # Cut points along the shuffled class indices according to proportions.
        cuts  = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(c_indices, cuts)):
            cl_idx[cid].extend(chunk.tolist())

    out = []
    for cid in range(n_clients):
        idx = np.array(cl_idx[cid], dtype=int)
        if len(idx) < min_per_client:
            # Top up under-filled clients from the global pool (with replacement).
            extra = rng.choice(n_train, size=min_per_client - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        rng.shuffle(idx)
        out.append(idx)
    return out


def _partition(Xtr, ytr, Xte, yte, n_clients: int,
               partition: str = "iid", dirichlet_alpha: float = 0.5,
               min_per_client: int = 10,
               seed: int = None) -> tuple:
    """Split data into n_clients subsets.

    Train:
      - partition="iid": bootstrap (with replacement) so each client gets
        max(total // n_clients, min_per_client) samples.  Handles
        n_clients > len(Xtr) and prevents degenerate per-client Hessians.
      - partition="dirichlet": disjoint label-skew split via a Dirichlet(alpha)
        draw per class (see _dirichlet_train_indices); under-filled clients are
        topped up to min_per_client.
    Test:  disjoint strided slices (a global, near-balanced test set shared by
           all clients); falls back to the full test set when there are fewer
           test samples than clients.  This is kept identical for both partition
           modes so non-iid runs measure learning/generalisation on a common
           yardstick rather than per-client personalisation.
    """
    rng = np.random.default_rng(seed)
    n_train = len(Xtr)
    n_test  = len(Xte)

    Xtr_s, ytr_s = [], []
    if partition == "dirichlet":
        # Labels arrive one-hot from the caller; recover integer class indices.
        y_int = ytr.argmax(axis=1) if ytr.ndim == 2 else ytr
        for idx in _dirichlet_train_indices(
            y_int, n_clients, dirichlet_alpha, min_per_client, rng
        ):
            Xtr_s.append(Xtr[idx])
            ytr_s.append(ytr[idx])
    elif partition == "iid":
        samples_per_client = max(n_train // n_clients, min_per_client)
        for _ in range(n_clients):
            idx = rng.choice(n_train, size=samples_per_client, replace=True)
            Xtr_s.append(Xtr[idx])
            ytr_s.append(ytr[idx])
    else:
        raise ValueError(f"Unknown partition '{partition}' "
                         f"(expected 'iid' or 'dirichlet')")

    Xte_s, yte_s = [], []
    if n_test >= n_clients:
        for i in range(n_clients):
            idx = np.arange(i, n_test, n_clients)
            Xte_s.append(Xte[idx])
            yte_s.append(yte[idx])
    else:
        for _ in range(n_clients):
            Xte_s.append(Xte)
            yte_s.append(yte)

    return Xtr_s, ytr_s, Xte_s, yte_s


# ─── Per-client worker functions (called via parallelbar.progress_map) ────────
#
# These functions must be module-level (not lambdas) so that multiprocessing
# can pickle them.  They read from _clients, a module-level list populated in
# main() before progress_map is called.
# NOTE: This relies on fork-based multiprocessing (default on Linux).

_clients: list = []   # populated in main(); do not modify elsewhere


def _train_client(client_id: int) -> tuple:
    """Compute gradient and Hessian for one client; return (grad, hessian, client_id)."""
    grad, hessian = _clients[client_id].train()
    _clients[client_id].model.nodes[0].reset()   # reset only the reservoir state
    return grad, hessian, client_id


def _evaluate_client(client_id: int) -> dict:
    """Evaluate one client and return a metrics dict."""
    result = _clients[client_id].evaluate()
    _clients[client_id].model.reset()
    return result


def _last_states(X_seqs, reservoir) -> np.ndarray:
    """Last reservoir state per sequence (used for the pooled val split)."""
    states = []
    for x in X_seqs:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        reservoir.reset()
        states.append(reservoir.run(x)[-1].copy())
    reservoir.reset()
    return np.vstack(states)


def _carve_val_shards(Xtr_s: list, ytr_s: list, val_frac: float,
                      data_seed: int) -> tuple:
    """Hold out val_frac of every client's LOCAL train shard, pooled server-side.

    The pooled validation set is the round-selection / early-stop metric — the
    test shards stay untouched and are only *reported*, so hyperparameter search
    and best-round selection no longer read the test set.

    Returns (Xtr_s, ytr_s, Xval, yval_int) — train shards shrunk in place.
    """
    rng = np.random.RandomState(data_seed)
    Xval, yval = [], []
    for i in range(len(Xtr_s)):
        n     = len(Xtr_s[i])
        n_val = max(1, int(round(n * val_frac)))
        idx   = rng.permutation(n)
        Xval.extend(Xtr_s[i][idx[:n_val]])
        yval.extend(ytr_s[i][idx[:n_val]])
        Xtr_s[i] = Xtr_s[i][idx[n_val:]]
        ytr_s[i] = ytr_s[i][idx[n_val:]]
    yval = np.asarray(yval)
    yval_int = yval.argmax(axis=1) if yval.ndim == 2 else yval.astype(int)
    return Xtr_s, ytr_s, Xval, yval_int


def _federated_imp(server, n_clients: int, w_dense: np.ndarray,
                   target_sp: float, n_classes: int,
                   n_rounds: int = 5, n_refit: int = 5) -> np.ndarray:
    """Federated Iterative Magnitude Pruning — the prune-aware CE baseline for FL.

    The federated analogue of pruning.ce_imp_prune: starting from the dense
    (reg_type='none') global model, ramp a hard support mask to *target_sp* over
    *n_rounds*; after each prune, do *n_refit* masked federated Newton steps —
    clients compute their (grad, Hessian) at the broadcast masked weights, the
    server sums them and solves the Newton step on the SURVIVING support per
    class, and the pruned coordinates are re-zeroed each step.  Unlike a
    centralized IMP reference this keeps the data federated, so it is a fair
    head-to-head with federated SL1 at matched sparsity.

    Uses the module-level _clients / _train_client (same fork-shared state as the
    main loop).  Returns the pruned global Wout (units, n_classes).
    """
    from scipy.sparse import csr_matrix
    from newton_solver import damped_solve

    current = np.asarray(w_dense, dtype=float).copy()
    for sp in pruning._prune_schedule(current, target_sp, n_rounds):
        current, _ = pruning.magnitude_prune(current, sp)
        mask = (current != 0.0)
        for _ in range(n_refit):
            # Broadcast the masked weights, then collect client grad/Hessian there.
            server.global_Wout = csr_matrix(current)
            for cid in range(n_clients):
                _clients[cid].receive_parameters(server.global_Wout)
            collected = [_train_client(cid)[:2] for cid in range(n_clients)]
            units = current.shape[0]
            agg_grad = sum((g for g, _ in collected), np.zeros((units, n_classes)))
            agg_hess = [sum((h[k] for _, h in collected),
                            np.zeros((units, units))) for k in range(n_classes)]
            direction = np.zeros_like(current)
            for k in range(n_classes):
                idx = np.where(mask[:, k])[0]
                if idx.size == 0:
                    continue
                delta, _ = damped_solve(agg_hess[k][np.ix_(idx, idx)], agg_grad[idx, k])
                direction[idx, k] = -delta
            current = (current + direction) * mask
    server.global_Wout = csr_matrix(current)
    for cid in range(n_clients):
        _clients[cid].receive_parameters(server.global_Wout)
    return current


# ─── Logging helpers ──────────────────────────────────────────────────────────
# Terminal-only: file logging (result/log/) was removed — terminal output is
# captured externally by a shell wrapper.

# Per-round logging verbosity. main() sets this from its *verbose* argument, so a
# hyperparameter search (many main() calls) can silence the per-round spam while a
# standalone run keeps the full trace.  (Module-level: each forked search worker
# sets its own copy.)
_VERBOSE = True


def _log(msg: str):
    """Write *msg* to stdout when verbose (see _VERBOSE / main(verbose=...))."""
    if _VERBOSE:
        print(msg)


def _log_results(results: list):
    """Print per-client accuracy and sparsity table."""
    _log("  ID\tAcc\t\tSparsity")
    for r in results:
        _log(f"  {r['client_id']}\t{r['acc']:.2f}%\t{r['sparsity']:.2f}%")


def _log_weight_distribution(Wout):
    """Print the proportion of global Wout entries falling in each magnitude bin.

    Bins are defined by absolute value thresholds on a log10 scale:
      <1e-2 | [1e-2, 1e-1) | [1e-1, 1) | [1, 10) | [10, 100) | >=100
    """
    from scipy.sparse import issparse
    w_abs = np.abs(Wout.toarray() if issparse(Wout) else np.asarray(Wout)).ravel()
    total = len(w_abs)

    edges  = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    labels = ["<1e-2", "[1e-2, 1e-1)", "[1e-1, 1)", "[1, 10)", "[10, 100)", ">=100"]
    masks  = [
        w_abs < 1e-2,
        (w_abs >= 1e-2)  & (w_abs < 1e-1),
        (w_abs >= 1e-1)  & (w_abs < 1.0),
        (w_abs >= 1.0)   & (w_abs < 10.0),
        (w_abs >= 10.0)  & (w_abs < 100.0),
        w_abs >= 100.0,
    ]

    _log("  Global Wout magnitude distribution:")
    for label, mask in zip(labels, masks):
        _log(f"    |w| {label:<16s}: {mask.sum() / total * 100:6.2f}%")


# ─── Main experiment function ─────────────────────────────────────────────────

def main(
    reg_type: str = "none",
    n_rounds: int = 5,
    n_clients: int = 5,
    seed: int = 1234,
    global_lr: float = 1.0,
    local_lr: float = 1.0,
    local_epochs: int = 100,
    thres: float = config.sl1_defaults.THRES,
    alpha_init: float = config.sl1_defaults.ALPHA_INIT,
    alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
    setting_idx: int = None,
    datasets: Optional[list] = None,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
    val_frac: float = 0.2,
    data_seed: Optional[int] = None,
    use_cache: bool = True,
    use_gpu: bool = config.USE_GPU,
    exp_suffix: str = "",
    use_line_search: bool = True,
    ls_c: float = 1e-4,
    ls_rho: float = 0.5,
    ls_max_iter: int = 20,
    patience: int = 0,
    min_delta: float = 0.1,
    param_overrides: Optional[dict] = None,
    imp_target_sp: Optional[float] = None,
    save_model: bool = False,
    verbose: bool = True,
) -> dict:
    """Run one or more federated learning experiments.

    Can be called directly from tsc_fl_eval_sl1.py (or any other wrapper)
    to share the full implementation without code duplication.

    Args:
        reg_type:     Regularisation applied to readout weights.
        n_rounds:     Number of FL communication rounds.
        n_clients:    Number of simulated FL clients.
        seed:         Random seed for reproducibility.
        global_lr:    Server-side mixing coefficient for Wout update.
        local_lr:     Client-side mixing coefficient when receiving global Wout.
        local_epochs: Gradient steps per client per round.
        setting_idx:  If given, run only settings[setting_idx] (wins over
                      *datasets*; kept for tsc_fl_search / tsc_noniid_compare).
        datasets:     Optional list of dataset names to run (entries missing
                      from the settings JSON are skipped with a warning);
                      None = every entry in the settings file.
        val_frac:     Fraction of every client's LOCAL train shard pooled into a
                      server-side validation set; best round + early stopping
                      select on val, test is only reported (0 disables and
                      restores the old select-on-test behaviour).
        data_seed:    Seed for the client partition and the val carve-out;
                      None = *seed*. Pass a fixed value while varying *seed*
                      (e.g. per search trial) so all trials compare on the SAME
                      partition and config quality is not confounded with
                      partition luck.
        use_cache:    Load pre-cached .npz data instead of raw files.
        use_gpu:      Offload the readout grad/Hessian/loss math to the GPU via
                      CuPy (auto-falls back to CPU when CuPy/CUDA is missing).
        exp_suffix:   String appended to the experiment name (e.g. "long_run").
    """
    global _clients, _VERBOSE
    _VERBOSE = verbose   # silence the per-round trace when a search drives main()

    np.random.seed(seed)

    # Resolve the GPU request once so the log line reflects what will actually run.
    use_gpu = funcs.resolve_use_gpu(use_gpu)

    all_settings = _read_settings(reg_type)
    if setting_idx is not None:
        settings = [all_settings[setting_idx]]
    elif datasets:
        by_name = {s["dataset"]: s for s in all_settings}
        missing = [d for d in datasets if d not in by_name]
        if missing:
            print(f"[SKIP] not in TSC_FL_settings_{reg_type}.json: "
                  f"{', '.join(missing)}")
        settings = [by_name[d] for d in datasets if d in by_name]
    else:
        settings = all_settings

    results_by_dataset: dict = {}

    for setting in settings:
        # Merge run-level config into the per-dataset setting dict so that
        # Server and Client constructors see a single flat hyperparameter dict.
        setting.update({
            "seed":         seed,
            "reg_type":     reg_type,
            "global_lr":    global_lr,
            "local_lr":     local_lr,
            "local_epochs": local_epochs,
            # SL1 sparsification schedule — uniform across datasets, overrides JSON.
            "thres":            thres,
            "alpha_init":       alpha_init,
            "alpha_multiplier": alpha_multiplier,
        })
        # Allow callers (e.g. tsc_fl_search.py) to override any setting key.
        if param_overrides:
            setting.update(param_overrides)

        suffix     = f"_{exp_suffix}" if exp_suffix else ""
        exp_name   = f"{setting['dataset']}_{reg_type}{suffix}"

        _log(f"\n{'='*60}")
        _log(f"Experiment : {exp_name}")
        _log(f"Dataset    : {setting['dataset']}  |  reg={reg_type}")
        _log(f"Rounds     : {n_rounds}  |  Clients: {n_clients}")
        _log(f"Compute    : {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")
        _log(f"{'='*60}")

        # ── 1. Initialise server — creates shared reservoir weights Win, W ───
        server = Server_FedAvg(hyperparams=setting)
        server.initialize_global_model()
        _log("Global reservoir initialised.")

        # ── 2. Load and prepare data ─────────────────────────────────────────
        Xtr, ytr, Xte, yte = _load_data(setting["dataset"], use_cache)

        # Feature-wise z-score normalization (stats computed from train set only)
        Xtr, Xte = standardize(Xtr, Xte)

        # One-hot encode integer labels if necessary
        if ytr.ndim == 1:
            ytr = one_hot(ytr, setting["output_dim"])
            yte = one_hot(yte, setting["output_dim"])

        # Partition into client subsets (IID bootstrap or Dirichlet label skew;
        # strided global test set in both cases)
        _data_seed = seed if data_seed is None else data_seed
        Xtr_s, ytr_s, Xte_s, yte_s = _partition(
            Xtr, ytr, Xte, yte, n_clients,
            partition=partition, dirichlet_alpha=dirichlet_alpha,
            min_per_client=10, seed=_data_seed,
        )
        if partition == "dirichlet":
            _log(f"Partition  : Dirichlet(alpha={dirichlet_alpha}) label skew"
                 f"  (data_seed={_data_seed})")
        else:
            _log(f"Partition  : IID (bootstrap)  (data_seed={_data_seed})")

        # Pooled validation split (selection metric; test is report-only).
        val_states, yval_int = None, None
        if val_frac > 0:
            Xtr_s, ytr_s, Xval, yval_int = _carve_val_shards(
                Xtr_s, ytr_s, val_frac, _data_seed)
            val_states = _last_states(Xval, server.res)
            _log(f"Validation : {len(yval_int)} samples pooled from client "
                 f"train shards (val_frac={val_frac}) — selection metric")

        # ── 3. Initialise clients ────────────────────────────────────────────
        # All clients share the same frozen reservoir (Win, W) from the server.
        # Only the readout Wout is trained locally and aggregated.
        _clients = []
        for i in range(n_clients):
            c = Client_TSC(
                i,
                data=[Xtr_s[i], ytr_s[i], Xte_s[i], yte_s[i]],
                seed=seed,
                use_gpu=use_gpu,
            )
            c.receive_hyperparams(setting)
            c.initialize_model(setting["output_dim"], server.res.Win, server.res.W)
            _clients.append(c)

        _log(f"Initialised {n_clients} clients — "
             f"~{len(Xtr_s[0])} train / {len(Xte_s[0])} test samples each.")

        # Log per-client label histograms so the (non-)iid skew is verifiable.
        n_cls = setting["output_dim"]
        _log("  Per-client train label distribution:")
        for cid in range(n_clients):
            yc = ytr_s[cid].argmax(axis=1) if ytr_s[cid].ndim == 2 else ytr_s[cid]
            hist = np.bincount(yc.astype(int), minlength=n_cls)
            _log(f"    client {cid} (n={len(yc)}): {hist.tolist()}")

        # ── 4. Federated learning rounds ─────────────────────────────────────
        _log("Starting FL rounds...")
        global_Wout = None
        best_global_Wout   = None
        best_sel_acc          = -1.0  # selection metric: val acc (test acc if val_frac=0)
        best_avg_acc          = -1.0  # TEST metrics recorded at the best-val round
        best_avg_sparsity     = 0.0
        best_avg_macro_f1     = 0.0
        best_avg_balanced_acc = 0.0
        rounds_no_improve  = 0
        # Early-stop reference peaks: highest selection-acc / sparsity seen so far.
        es_best_acc        = -np.inf
        es_best_sparsity   = -np.inf
        # Per-round test acc/sparsity history (report-only; for round-evolution plots).
        round_history      = []

        alpha_mult = setting.get("alpha_multiplier", config.sl1_defaults.ALPHA_MULTIPLIER)
        alpha_max  = setting.get("alpha_max", config.sl1_defaults.ALPHA_MAX)

        for rnd in range(n_rounds):
            _log(f"\n--- Round {rnd + 1}/{n_rounds} "
                 f"(SL1 alpha = {_clients[0].alpha:.4g}) ---")

            collected_params = [None] * n_clients
            for cid in range(n_clients):
                grad, hessian, _ = _train_client(cid)
                collected_params[cid] = (grad, hessian)

            # Evaluate before aggregation (Wout unchanged at this point)
            # _log("Before aggregation:")
            # _log_results([_evaluate_client(cid) for cid in range(n_clients)])

            # Server solves Newton step (with optional line search) and broadcasts
            global_Wout, ls_step = server.aggregate_parameters(
                collected_params,
                clients=_clients if use_line_search else None,
                use_line_search=use_line_search,
                ls_c=ls_c,
                ls_rho=ls_rho,
                ls_max_iter=ls_max_iter,
            )
            if use_line_search:
                _log(f"  Line search step = {ls_step:.6f}")
            for client in _clients:
                client.receive_parameters(global_Wout)

            # Evaluate after aggregation
            _log("After aggregation:")
            round_results = [_evaluate_client(cid) for cid in range(n_clients)]
            _log_results(round_results)
            _log_weight_distribution(global_Wout)

            # Best-model tracking — selected on VAL (test is report-only); falls
            # back to test acc when val_frac=0.
            avg_acc          = np.mean([r["acc"]          for r in round_results])
            avg_sparsity     = np.mean([r["sparsity"]     for r in round_results])
            avg_macro_f1     = np.mean([r["macro_f1"]     for r in round_results])
            avg_balanced_acc = np.mean([r["balanced_acc"] for r in round_results])
            round_history.append({
                "round":    rnd + 1,
                "test_acc": float(avg_acc),
                "sparsity": float(avg_sparsity),
            })
            if val_states is not None:
                W = global_Wout.toarray()
                sel_acc = float(
                    (np.argmax(val_states @ W, axis=1) == yval_int).mean() * 100)
                _log(f"  Val acc = {sel_acc:.2f}%  (selection metric)")
            else:
                sel_acc = avg_acc
            if sel_acc > best_sel_acc:
                best_sel_acc          = sel_acc
                best_avg_acc          = avg_acc
                best_avg_sparsity     = avg_sparsity
                best_avg_macro_f1     = avg_macro_f1
                best_avg_balanced_acc = avg_balanced_acc
                best_global_Wout  = server.global_Wout
                _log(f"  [best] sel acc = {sel_acc:.2f}%  test acc = {avg_acc:.2f}%  "
                     f"sparsity = {avg_sparsity:.2f}%  (round {rnd + 1})")

            # Early-stop bookkeeping: a round is a "significant improvement" if
            # EITHER the selection accuracy (val) OR sparsity rises by more than
            # min_delta over its previous peak. The no-improvement counter only
            # advances when BOTH have plateaued.
            acc_improved      = sel_acc      > es_best_acc      + min_delta
            sparsity_improved = avg_sparsity > es_best_sparsity + min_delta
            if acc_improved or sparsity_improved:
                rounds_no_improve = 0
                if acc_improved:
                    es_best_acc = sel_acc
                if sparsity_improved:
                    es_best_sparsity = avg_sparsity
                _log(f"  [improve] acc {'↑' if acc_improved else '–'}  "
                     f"sparsity {'↑' if sparsity_improved else '–'}  "
                     f"(sel_acc={sel_acc:.2f}%  sparsity={avg_sparsity:.2f}%)")
            else:
                rounds_no_improve += 1
                _log(f"  [no significant change {rounds_no_improve}]  "
                     f"sel_acc={sel_acc:.2f}%  sparsity={avg_sparsity:.2f}%")

            # SL1 alpha continuation: update after evaluation so each round's
            # gradient/Hessian uses the alpha that was active during that round,
            # and the next round starts with the new (larger) alpha.
            for client in _clients:
                client.update_alpha(alpha_mult, alpha_max)
            _log(f"  SL1 alpha → {_clients[0].alpha:.4g} (×{alpha_mult})")

            # Early stopping
            if patience > 0 and rounds_no_improve >= patience:
                _log(f"  Early stop: neither accuracy nor sparsity improved "
                     f"for {patience} rounds.")
                global_Wout = best_global_Wout
                break
    



        # ── 5. Optionally persist the best global model (opt-in via save_model) ─
        # Off by default so hyperparameter searches (one run_experiment per trial)
        # don't litter result/model with a full Win+Wres+Wout .npz per trial —
        # only an explicit final run (--save_model) writes the best model.
        assert best_global_Wout is not None, "No model produced — zero rounds run."
        _log(f"\nBest round (by {'val' if val_states is not None else 'test'} acc "
             f"{best_sel_acc:.2f}%): test acc = {best_avg_acc:.2f}%")
        if save_model:
            model_path = os.path.join(
                config.paths.model_path, f"global_{exp_name}.npz"
            )
            np.savez(
                model_path,
                Win=server.res.Win.toarray(),
                Wres=server.res.W.toarray(),
                Wout=best_global_Wout.toarray(),
            )
            _log(f"Global model saved → {model_path}")

        result = {
            "acc":          best_avg_acc,   # TEST acc at the best-val round
            "sparsity":     best_avg_sparsity,
            "macro_f1":     best_avg_macro_f1,
            "balanced_acc": best_avg_balanced_acc,
            # Selection metric (val acc; equals test acc when val_frac=0) — what
            # tsc_fl_search scores on, so hyperparameters never see the test set.
            "val_acc":      best_sel_acc,
            # Per-round test acc/sparsity (report-only; for round-evolution plots).
            "round_history": round_history,
        }

        # ── Optional federated CE+IMP baseline at a matched sparsity ────────────
        # Prune-then-refit the dense global model to imp_target_sp (kept federated
        # via masked aggregation) and evaluate — the honest CE competitor to
        # federated SL1 at equal sparsity.  Only meaningful for a dense run
        # (reg_type='none'); the caller passes the SL1 sparsity to match.
        if imp_target_sp is not None:
            _log(f"\nFederated CE+IMP baseline → {imp_target_sp:.1f}% sparse")
            w_imp = _federated_imp(server, n_clients, best_global_Wout.toarray(),
                                   imp_target_sp, n_cls)
            imp_res = [_evaluate_client(cid) for cid in range(n_clients)]
            result.update({
                "imp_acc":          float(np.mean([r["acc"]          for r in imp_res])),
                "imp_sparsity":     float(np.mean([r["sparsity"]     for r in imp_res])),
                "imp_macro_f1":     float(np.mean([r["macro_f1"]     for r in imp_res])),
                "imp_balanced_acc": float(np.mean([r["balanced_acc"] for r in imp_res])),
            })
            _log(f"  IMP: acc={result['imp_acc']:.2f}%  f1={result['imp_macro_f1']:.2f}%"
                 f"  bal={result['imp_balanced_acc']:.2f}%  sp={result['imp_sparsity']:.2f}%")

        results_by_dataset[setting["dataset"]] = result

    return results_by_dataset


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    reg_types = (["none", "l2", "sl1"] if "all" in args.reg_types
                 else list(dict.fromkeys(args.reg_types)))
    for rt in reg_types:
        main(
            reg_type        = rt,
            n_rounds        = args.n_rounds,
            n_clients       = args.n_clients,
            seed            = args.seed,
            global_lr       = args.global_lr,
            local_lr        = args.local_lr,
            local_epochs    = args.local_epochs,
            thres           = args.thres,
            alpha_init      = args.alpha_init,
            alpha_multiplier= args.alpha_multiplier,
            setting_idx     = args.setting_idx,
            datasets        = args.datasets,
            partition       = args.partition,
            dirichlet_alpha = args.dirichlet_alpha,
            val_frac        = args.val_frac,
            data_seed       = args.data_seed,
            use_cache       = not args.no_cache,
            use_gpu         = args.use_gpu,
            exp_suffix      = args.exp_suffix,
            use_line_search = args.line_search,
            ls_c            = args.ls_c,
            ls_rho          = args.ls_rho,
            ls_max_iter     = args.ls_max_iter,
            patience        = args.patience,
            min_delta       = args.min_delta,
            save_model      = args.save_model,
            verbose         = not args.quiet,
        )
