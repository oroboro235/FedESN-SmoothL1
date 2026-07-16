# utils.py — general utilities used across the project.
#
# Contents:
#   logsumexp          — thin wrapper around scipy (used by funcs.py primitives)
#   get_fft_curves_*   — FFT-based signal decomposition helpers (exploratory)
#   get_diff_sampling_rate_series — multi-rate Mackey-Glass generator
#   plot_readout       — visualise readout weight sparsity (standalone use)

import csv
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
from scipy.special import logsumexp as sci_lse


# ─── Search utilities (shared by all three hparam search scripts) ─────────────

def log_uniform(lo: float, hi: float, rng) -> float:
    """Log-uniform random sample in [lo, hi]; *rng* is a random.Random instance."""
    return 10 ** rng.uniform(np.log10(lo), np.log10(hi))


# ─── Parallel trial execution (shared by all three hparam search scripts) ─────
#
# All three searches have the same shape: an outer (dataset × reg_type) loop and
# an inner batch of *independent* trials/steps.  parallel_map() runs that inner
# batch across processes while preserving submit order, so each script can keep
# sampling its configs sequentially (RNG stream / seed derivation unchanged) and
# only fan out the expensive evaluation.  Workers read large read-only data via
# fork-inherited module globals (copy-on-write) rather than per-task pickling.

def resolve_n_jobs(n_jobs: int) -> int:
    """Normalise a --n_jobs value: 0/None → 1 (serial), negative → all cores."""
    if not n_jobs:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def _worker_init():
    """Pool-worker initialiser: cap BLAS env threads so N workers don't each
    spawn N threads (best-effort; covers spawn/lazy-init BLAS libraries)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _call_star(packed):
    """Unpack ``(func, args)`` and call it, limiting BLAS threads at runtime.

    threadpool_limits caps the per-process thread pool regardless of start
    method, preventing N×N oversubscription when many workers run numpy/BLAS.
    """
    func, args = packed
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return func(*args)
    except ImportError:
        return func(*args)


def _render_progress(done: int, total: int, t_start: float, label: str):
    """Overwrite a single-line progress bar on stderr (carriage-return refresh).

    Called once per completed task; shows a filled bar, done/total, percent,
    throughput and a rough ETA. Uses stderr so it doesn't intermix with the
    scripts' stdout CSV/summary prints, and \\r keeps it to one redrawn line.
    """
    frac    = done / total if total else 1.0
    width   = 30
    filled  = int(width * frac)
    bar     = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - t_start
    rate    = done / elapsed if elapsed > 0 else 0.0
    eta     = (total - done) / rate if rate > 0 else 0.0
    end     = "\n" if done >= total else ""
    print(f"\r{label} [{bar}] {done}/{total} ({frac * 100:4.1f}%)  "
          f"{rate:5.1f} it/s  ETA {eta:5.0f}s", end=end, flush=True,
          file=sys.stderr)


def parallel_map(func, args_list, n_jobs: int, progress: str = None):
    """Apply *func* to each positional-args tuple in *args_list*, in order.

    ``n_jobs <= 1`` (or a single task) runs serially in-process — byte-for-byte
    the same path as the original loop, and the only safe mode under GPU use.
    Otherwise dispatches to a ProcessPoolExecutor with up to *n_jobs* workers
    (``n_jobs < 0`` = all cores).  Results come back in submit order regardless
    of completion order, so downstream tracking/CSV output stays deterministic.

    If *progress* is a non-empty string, a live progress bar labelled with it is
    drawn on stderr, incremented once per *actually completed* task (real
    completion order, not submit order), so the otherwise-silent parallel batch
    shows steady advancement + ETA. Results are still returned in submit order.

    *func* and every item in *args_list* must be picklable (top-level callables,
    plain data).  Large read-only inputs should be shared via fork-inherited
    module globals instead of being placed in *args_list*.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n     = resolve_n_jobs(n_jobs)
    total = len(args_list)

    if n == 1 or total <= 1:
        # Route through _call_star so the serial path caps BLAS threads exactly
        # like a pool worker does.  Without this, a serial run lets OpenBLAS use
        # every core, and np.linalg.solve() on the readout's 500x500 per-class
        # Hessian degrades catastrophically: ~189 ms at 24 threads vs ~2.5 ms at
        # 1 (LU sync overhead swamps a matrix this small; matmul is unaffected).
        # The Newton solvers do one solve per class per iteration, so the serial
        # path was ~75x slower than a single parallel worker.
        results = []
        t0 = time.time()
        for a in args_list:
            results.append(_call_star((func, a)))
            if progress:
                _render_progress(len(results), total, t0, progress)
        return results

    # Fill results by submit index (deterministic order) while counting real
    # completions via as_completed for an accurate live bar.
    results = [None] * total
    t0      = time.time()
    with ProcessPoolExecutor(max_workers=n, initializer=_worker_init) as ex:
        fut_to_idx = {
            ex.submit(_call_star, (func, a)): i
            for i, a in enumerate(args_list)
        }
        done = 0
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()
            done += 1
            if progress:
                _render_progress(done, total, t0, progress)
    return results


def init_csv(path: str, fieldnames: list):
    """Create a CSV file at *path* with a header row if it does not yet exist."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: str, fieldnames: list, row: dict):
    """Append one dict row to an existing CSV (header must already exist)."""
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)


class BestTracker:
    """Track the best record per key, by a caller-supplied score (higher is better).

    Storage-agnostic: each update supplies a pre-computed *score* and an arbitrary
    *record* dict.  Shared by all three hyperparameter-search scripts
    (tsr_centralized_search, tsc_centralized_search, tsc_fl_search); each script
    keeps its own score formula and report / JSON-export formatting, but the
    "keep the highest-scoring config per (dataset, reg_type)" bookkeeping lives
    here in one place.

    Typical usage:
        tracker = BestTracker()
        score   = my_score_formula(...)
        tracker.update((dataset, reg_type), score, {"acc": ..., "cfg": cfg})
        for key, score, record in tracker.items():
            ...
    """

    def __init__(self):
        self._best: dict = {}   # key -> (score, record)

    def update(self, key, score: float, record: dict) -> bool:
        """Store *record* under *key* if *score* beats the current best.

        Returns True if the record was stored (a new best), else False.
        """
        if key not in self._best or score > self._best[key][0]:
            self._best[key] = (score, dict(record))
            return True
        return False

    def get(self, key):
        """Return (score, record) for *key*, or None if absent."""
        return self._best.get(key)

    def items(self):
        """Yield (key, score, record) tuples, sorted by key."""
        for key, (score, record) in sorted(self._best.items()):
            yield key, score, record


# ─── logsumexp ───────────────────────────────────────────────────────────────

def logsumexp(b, axis: int = 1):
    """Numerically stable logsumexp via scipy.

    Wraps scipy.special.logsumexp so the rest of the codebase has a single
    import point; the *axis* default matches the manual implementation that
    was previously inlined in funcs.py.
    """
    return sci_lse(b, axis=axis)


# ─── FFT helpers ─────────────────────────────────────────────────────────────

def get_fft_curves_split_n(x, n: int = 3, max_freq=None, is_plot: bool = False):
    """Decompose signal *x* into *n* equal frequency bands via FFT.

    Returns a list of *n* single-channel signals (shape: (len(x), 1)).
    Used for exploratory frequency-domain analysis of time-series data.
    """
    x = x.squeeze()
    fft_result = rfft(x)
    t = np.arange(len(x))
    freqs = rfftfreq(len(x), 1 / len(x))

    if max_freq is None or max_freq > np.max(freqs):
        max_freq = np.max(freqs)

    step_size = max_freq / n
    results = []

    for i in range(n):
        start_freq = i * step_size
        end_freq = start_freq + step_size
        freq_indices = np.where((freqs >= start_freq) & (freqs < end_freq))[0]
        if len(freq_indices) == 0:
            continue

        band_fft = np.zeros_like(fft_result)
        band_fft[freq_indices] = fft_result[freq_indices]
        signal = irfft(band_fft, n=len(t))
        results.append(signal.reshape(-1, 1))

        if is_plot:
            plt.figure(figsize=(12, 8))
            plt.plot(t, signal, label=f"{start_freq:.2f} – {end_freq:.2f} Hz")
            plt.ylabel("Magnitude")
            plt.xlabel("Time")
            plt.title(f"FFT band {i + 1}/{n}")
            plt.legend()
            plt.grid(True)
            plt.show()

    if is_plot:
        plt.figure(figsize=(12, 8))
        plt.plot(t, x, label="Original")
        plt.plot(t, np.sum(results, axis=0), label="Reconstructed")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("FFT reconstruction")
        plt.show()

    return results


def get_fft_curves_top_n(x, top_n: int = 3):
    """Extract the *top_n* dominant frequency components from signal *x*.

    Returns a list of single-frequency time-domain signals.
    """
    x = x.squeeze()
    fft_result = rfft(x)
    t = np.arange(len(x))
    freqs = rfftfreq(len(x), 1 / len(x))

    mag = np.abs(fft_result)
    non_dc = freqs > 0
    top_indices = np.argsort(mag[non_dc])[-top_n:][::-1]
    top_freqs = freqs[non_dc][top_indices]
    top_mag = mag[non_dc][top_indices]

    print(f"Top {top_n} frequencies: {top_freqs} Hz, magnitudes: {top_mag}")

    individual_signals = []
    for freq_idx in top_indices:
        actual_idx = np.where(freqs == freqs[non_dc][freq_idx])[0][0]
        single_fft = np.zeros_like(fft_result)
        single_fft[actual_idx] = fft_result[actual_idx]
        if actual_idx < len(fft_result) - 1:
            single_fft[actual_idx + 1] = fft_result[actual_idx + 1]
        individual_signals.append(irfft(single_fft, n=len(t)))

    final_signal = np.sum(individual_signals, axis=0)
    plt.plot(t, x, label="Original")
    plt.plot(t, final_signal, label="Reconstructed (top-n)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title(f"FFT top-{top_n} reconstruction")
    plt.show()

    return individual_signals


def get_diff_sampling_rate_series(
    n: int = 5,
    sample_period: list = None,
    len_forecast: int = 1,
    len_window: int = 100,
    step: int = 1,
    n_samples: int = 100,
):
    """Generate *n* Mackey-Glass sub-series at different sampling rates.

    Returns (diff_sampling_rate_series, raw_signal).
    """
    if sample_period is None:
        sample_period = [1] * n

    from reservoirpy.datasets import mackey_glass

    len_ts = (len_window + step * n_samples) * max(sample_period)
    raw = mackey_glass(len_ts)
    raw = 2 * (raw - raw.min()) / (raw.max() - raw.min()) - 1

    series = [
        raw[:: sample_period[i]][: len_window + step * n_samples]
        for i in range(n)
    ]
    return np.array(series), raw


# ─── Readout visualisation ───────────────────────────────────────────────────

def plot_readout(Wout: np.ndarray):
    """Bar-chart of readout weight magnitudes with sparsity annotation.

    Args:
        Wout: Readout weight matrix or flattened array.
    """
    sparsity = (Wout == 0).mean() * 100
    print(f"Wout sparsity: {sparsity:.2f}%")

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])
    ax.set_ylabel("Coefs. of $W_{out}$")
    ax.set_xlabel("Reservoir neuron index")
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig("readout_coefs.png")
    plt.show()
