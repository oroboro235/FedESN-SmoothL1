# synth_data.py — controllable synthetic time-series classification generator.
#
# Mechanism A of the non-iid study: produce a *pooled* dataset of known
# difficulty so the existing Dirichlet label-skew partition (tsc_fl_eval._partition)
# and the centralized/FL pipelines can run on it unchanged.  Because difficulty is
# a single knob (observation noise sigma) and the ground-truth class dynamics are
# known, the non-iid effect can be measured cleanly — without a real dataset's
# idiosyncrasies confounding the "alpha → accuracy" curve.
#
# Each class is a phase-locked sinusoid at a class-specific frequency: every
# sequence of class c is sin(omega_c * t + phi) sampled from t=0, so the
# discriminative information lives in the trajectory *endpoint* — which is what
# this project's readout reads (the ESN's last hidden state, _extract_states).
# Per-sample variability comes from observation noise and small amplitude jitter
# (NOT a random realisation), so the last state stays class-discriminative.
#
# NOTE: an earlier stochastic AR(2) version was discarded — its random phase made
# the last hidden state a meaningless snapshot (ESN stuck at chance) even though
# the full power spectrum was separable.  Phase-locking is what makes the task
# learnable by a last-state readout.
#
# Difficulty is controlled by:
#   - band         : frequency range the class freqs span; narrower packs classes
#                    closer together (harder)
#   - sigma        : additive observation noise (higher = harder)
#   - phase_jitter : per-sample phase randomisation; larger erodes the endpoint
#                    determinism the readout relies on (harder)
#   - n_classes    : number of frequencies to tell apart (more = harder)
#
# Presets are calibrated (verified with an ESN last-state ridge readout) to span
# a real separability ladder — roughly 100% / 83% / 47% accuracy
# (5-class chance = 20%) — so the difficulty axis is meaningful.
#
# Three preset difficulties are registered (synth_easy / synth_med / synth_hard);
# data_loader.read_data() dispatches these names, and configs/datasets_tsc.json
# carries a matching metadata entry (input_dim=1, output_dim=5) for each
# (FL hyperparameters stay per-reg_type in configs/TSC_FL_settings_*.json).

import numpy as np


# ─── Preset difficulties (referenced by data_loader.read_data) ────────────────
# All share n_classes=5, n_features=1 so a single metadata entry shape
# (input_dim=1, output_dim=5) applies to every preset.
SYNTH_CONFIGS = {
    "synth_easy": dict(band=(0.10, 0.45), sigma=0.20, phase_jitter=0.0),
    "synth_med":  dict(band=(0.10, 0.45), sigma=0.60, phase_jitter=0.3),
    "synth_hard": dict(band=(0.10, 0.45), sigma=1.00, phase_jitter=0.8),
}

# Shared shape/size defaults (kept fixed so presets only vary difficulty).
_DEFAULTS = dict(
    n_classes=5, seq_len=100, n_features=1,
    n_train_per_class=150, n_test_per_class=70, seed=0,
)


def _sine(T: int, omega: float, sigma: float, phase_jitter: float,
          rng) -> np.ndarray:
    """One phase-locked sinusoid of frequency omega, sampled from t=0.

    sin(omega * t + phi),  phi = phase_jitter * N(0,1).  With phase_jitter=0 the
    sequence is fully deterministic given its class, so the endpoint (and hence
    the ESN's last hidden state) is class-discriminative; observation noise sigma
    and a small amplitude jitter supply per-sample variability.
    """
    t   = np.arange(T)
    phi = phase_jitter * rng.standard_normal()
    amp = 1.0 + 0.1 * rng.standard_normal()
    return amp * np.sin(omega * t + phi) + sigma * rng.standard_normal(T)


def make_synth(sigma: float, phase_jitter: float = 0.0,
               band: tuple = (0.10, 0.45),
               n_classes: int = _DEFAULTS["n_classes"],
               seq_len: int = _DEFAULTS["seq_len"],
               n_features: int = _DEFAULTS["n_features"],
               n_train_per_class: int = _DEFAULTS["n_train_per_class"],
               n_test_per_class: int = _DEFAULTS["n_test_per_class"],
               seed: int = _DEFAULTS["seed"]) -> tuple:
    """Generate a balanced K-class phase-locked-sinusoid classification dataset.

    Returns (Xtr, ytr, Xte, yte) with X shape (n, seq_len, n_features) and
    integer labels — the same contract as the other read_data_* readers.
    """
    rng = np.random.default_rng(seed)
    # Class frequencies evenly spread across the requested band (in units of π).
    freqs = np.linspace(band[0] * np.pi, band[1] * np.pi, n_classes)

    def _gen(n_per_class: int) -> tuple:
        n = n_per_class * n_classes
        X = np.zeros((n, seq_len, n_features))
        y = np.repeat(np.arange(n_classes), n_per_class)
        for i in range(n):
            for f in range(n_features):
                X[i, :, f] = _sine(seq_len, freqs[y[i]], sigma, phase_jitter, rng)
        idx = rng.permutation(n)           # shuffle so labels aren't ordered
        return X[idx], y[idx]

    Xtr, ytr = _gen(n_train_per_class)
    Xte, yte = _gen(n_test_per_class)
    return Xtr, ytr, Xte, yte


def make_synth_named(name: str) -> tuple:
    """Generate one of the registered presets by name (used by read_data)."""
    if name not in SYNTH_CONFIGS:
        raise ValueError(f"Unknown synth preset '{name}'. "
                         f"Known: {list(SYNTH_CONFIGS)}")
    return make_synth(**SYNTH_CONFIGS[name])


if __name__ == "__main__":
    for nm in SYNTH_CONFIGS:
        Xtr, ytr, Xte, yte = make_synth_named(nm)
        print(f"{nm:12s}  train {Xtr.shape}  test {Xte.shape}  "
              f"classes {len(set(ytr))}  "
              f"(balanced: {np.bincount(ytr).tolist()})")
