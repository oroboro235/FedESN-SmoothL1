# config.py — project-wide path and dataset configuration.
#
# Paths are intentionally kept relative (./...) so the project can be
# run from any machine without editing this file.

# ─── Dataset name registries ─────────────────────────────────────────────────
# Used by data_loader.py and get_data_path() to route to the right subdirectory.
TSR_DATASET_NAMES = ["mg_t17", "lorenz"]          # time-series regression
TSC_DATASET_NAMES = ["har", "char", "ucr", "uea"] # time-series classification


# ─── Directory paths ──────────────────────────────────────────────────────────
class paths:
    """Project-level directory paths (relative to the repository root).

    Usage (backward-compatible with the old `config.path.xxx` style):
        import config
        config.path.cache_path   # → "./cache/"
    """
    cache_path   = "./cache/"       # cached dataset .npz files
    pics_path    = "./result/pic/"  # saved figures
    data_root    = "./raw/"         # original raw dataset files
    model_path   = "./result/model/" # saved model weights
    configs_path = "./configs/"     # per-dataset FL hyperparameter JSON files
    tsc_dataset_meta = "./configs/datasets_tsc.json"  # TSC dataset metadata registry

# Backward-compatible alias: original code accessed config.path.xxx
path = paths


def load_settings(task: str, reg_type: str) -> dict:
    """Return {dataset: hyperparameter-entry} from a configs/ settings file.

    task ∈ {"tsc", "tsr", "tsc_fl"} → configs/TSC_settings_<reg_type>.json /
    TSR_settings_<reg_type>.json / TSC_FL_settings_<reg_type>.json.

    These files hold the tuned per-dataset hyperparameters (promoted from the
    search scripts' result/*_best_*.json output) and are consumed by the eval
    and comparison scripts (their use_best_params modes, on by default).
    Returns {} when the file does not exist, so callers can fall back to
    built-in defaults.
    """
    import json
    import os

    prefix = {"tsc": "TSC", "tsr": "TSR", "tsc_fl": "TSC_FL"}[task]
    p = os.path.join(paths.configs_path, f"{prefix}_settings_{reg_type}.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {e["dataset"]: e for e in json.load(f)}


def load_tsc_dataset_meta() -> dict:
    """Return {dataset_name: {input_dim, output_dim, units}} for all TSC datasets.

    Single source of TSC dataset *metadata*: configs/datasets_tsc.json, a plain
    registry of [{dataset, input_dim, output_dim, units}, ...].  Adding a new
    classification dataset means adding one entry there — the TSC_FL_settings_
    <reg_type>.json files remain purely per-dataset *hyperparameter* stores and
    are no longer scanned for metadata.

    Falls back to the legacy source (union over the TSC_FL_settings files) with a
    warning if the registry file is missing, so old checkouts keep working.
    """
    import json
    import os

    meta = {}
    if os.path.exists(paths.tsc_dataset_meta):
        with open(paths.tsc_dataset_meta) as f:
            for s in json.load(f):
                meta[s["dataset"]] = {
                    "input_dim":  s["input_dim"],
                    "output_dim": s["output_dim"],
                    "units":      s.get("units", 500),
                }
        return meta

    print(f"[WARN] {paths.tsc_dataset_meta} not found — falling back to "
          f"scanning TSC_FL_settings_*.json for dataset metadata")
    for rt in ("none", "l2", "sl1"):
        p = os.path.join(paths.configs_path, f"TSC_FL_settings_{rt}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for s in json.load(f):
                meta[s["dataset"]] = {
                    "input_dim":  s["input_dim"],
                    "output_dim": s["output_dim"],
                    "units":      s.get("units", 500),
                }
    return meta


# ─── Compute backend default ──────────────────────────────────────────────────
# Single switch for the project-wide default of GPU acceleration.  Every entry
# script exposes a `--gpu` CLI flag whose default is this value, so flipping it
# here changes the default for all scripts at once.  The actual GPU usage is
# still resolved at runtime by funcs.resolve_use_gpu() (falls back to CPU when
# CuPy/CUDA is unavailable), so setting this True on a CPU-only machine is safe.
USE_GPU = False


# ─── SL1 sparsification schedule defaults ─────────────────────────────────────
class sl1_defaults:
    """Uniform defaults for the SL1 sparsification schedule.

    These three knobs control SmoothL1 sparsification of the readout weights and
    are intentionally kept *identical across all datasets*: every entry script
    exposes them as CLI arguments (--thres / --alpha_init / --alpha_multiplier)
    that default to the values here, so there is a single place to change them.

    Note: reg_param (λ) is NOT here — it stays per-dataset (loaded from the
    configs/TSC_FL_settings_<reg_type>.json files / hyperparameter search).

        THRES            — soft/hard threshold magnitude; |w| < thres set to 0.
        ALPHA_INIT       — initial SmoothL1 smoothing parameter α.
        ALPHA_MULTIPLIER — single geometric growth factor for α (continuation),
                           applied uniformly across FL rounds, the centralized
                           Newton path, and the Adam loop.
        ALPHA_MAX        — upper bound for the α continuation schedule.
        PATIENCE         — stagnation early-stop window for the centralized Newton
                           path: stop once the non-zero support is unchanged for
                           this many iterations after α reaches ALPHA_MAX (0
                           disables, restoring the run-to-max_iter behaviour). The
                           summed-gradient opt_tol is unreachable with per-iteration
                           thresholding, so without this the solver always runs the
                           full max_iter.
        STAG_TOL         — max change in non-zero fraction across the PATIENCE
                           window to count as "frozen" (0.002 = 0.2 percentage pts).
    """
    THRES            = 1e-5
    ALPHA_INIT       = 1.0
    ALPHA_MULTIPLIER = 2.0
    ALPHA_MAX        = 5e6
    PATIENCE         = 8
    STAG_TOL         = 2e-3


class newton_smooth_defaults:
    """Early-stop for the smooth (l2 / none) centralized Newton path.

    The CE (+L2) objective is smooth and convex, so the Gauss-Newton fit is
    effectively done within tens of iterations. But on strongly-collinear ESN
    reservoir states ‖g‖∞ plateaus around ~1e-4 and never reaches the 1e-6
    optimality tolerance (and the absolute progress tolerance 1e-9 is likewise
    unreachable), so solve_newton_smooth used to grind the full max_iter (=epochs,
    e.g. 500 → ~30 s/trial) for a fit that had converged by iteration ~25. A
    *relative*-objective plateau is the reachable convergence signal here.

        REL_TOL  — relative objective-improvement floor: an iteration counts as
                   stalled when (f_old − f_new) / max(|f_old|, 1) < REL_TOL.
        PATIENCE — stop after this many consecutive stalled iterations (0 disables,
                   restoring the run-to-max_iter behaviour).
    """
    REL_TOL  = 1e-5
    PATIENCE = 5


# ─── Data path helper ─────────────────────────────────────────────────────────

def get_data_path(dataset_name: str) -> str:
    """Return the raw-data directory for *dataset_name*.

    Called by data_loader.py for every dataset reader function.
    Raises ValueError for unrecognised names so typos fail loudly.
    """
    if dataset_name in TSR_DATASET_NAMES:
        return paths.data_root + "tsr/" + dataset_name + "/"
    if dataset_name in TSC_DATASET_NAMES:
        if dataset_name == "ucr":
            return paths.data_root + "tsc/UCR_univariate/"
        if dataset_name == "uea":
            return paths.data_root + "tsc/UEA_multivariate/"
        return paths.data_root + "tsc/" + dataset_name + "/"
    raise ValueError(f"Unknown dataset name: '{dataset_name}'")


if __name__ == "__main__":
    print(f"cache:   {paths.cache_path}")
    print(f"data:    {paths.data_root}")
    print(f"models:  {paths.model_path}")
    print(f"configs: {paths.configs_path}")
