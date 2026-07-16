# data_loader.py — dataset readers for time-series regression and classification.
#
# Public entry point:
#   read_data(name) → (X_train, y_train, X_test, y_test)
#
# Supported dataset names:
#   Regression : "mg", "lorenz"
#   Classification: "har", "char", "jpv",
#                   and any name in uni_names (UCR univariate datasets)

import os
import numpy as np
import scipy.io

from config import get_data_path


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _pad_sequences(sequences, max_len: int, n_features: int) -> np.ndarray:
    """Left-pad (pre-pad) a list of variable-length sequences to a fixed length.

    Args:
        sequences:  List of arrays, each shape (seq_len, n_features).
        max_len:    Target sequence length; sequences longer than this are truncated.
        n_features: Feature dimension.

    Returns:
        Array of shape (n_samples, max_len, n_features) filled with zeros outside
        the valid region.

    Called by: read_data_char (inside preprocess_written_char),
               read_data_jpv.
    """
    out = np.zeros((len(sequences), max_len, n_features))
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len <= max_len:
            out[i, max_len - seq_len:, :] = seq   # left-pad with zeros
        else:
            out[i] = seq[:max_len, :]              # truncate from the right
    return out


# ─── Time-series regression ───────────────────────────────────────────────────

def read_data_mg():
    """Mackey-Glass t=17 dataset (one-step-ahead prediction).

    Called by: read_data("mg").
    """
    data = np.load(get_data_path("mg_t17") + "mackey_glass_t17.npy")
    data = data.reshape(-1, 1)

    len_train, len_test = 5000, 1000
    u = data[:len_train + len_test]        # input: u(t)
    v = data[1: len_train + len_test + 1]  # target: u(t+1)

    return u[:len_train], v[:len_train], u[len_train:], v[len_train:]


def read_data_lorenz():
    """Lorenz attractor dataset (one-step-ahead prediction, 3-D).

    Called by: read_data("lorenz").
    """
    data = np.load(get_data_path("lorenz") + "lorenz_full.npy")  # (10000, 3)

    len_train, len_test = 5000, 1000
    u = data[:len_train + len_test]
    v = data[1: len_train + len_test + 1]

    return u[:len_train], v[:len_train], u[len_train:], v[len_train:]


# ─── Generated autonomous chaotic systems (regression benchmarks) ─────────────
# Additional one-step-ahead forecasting benchmarks beyond mg / lorenz, generated
# deterministically (no data files).  All are AUTONOMOUS maps/flows — the next
# state is a function of the current state only — so the free-running
# auto-regressive forecast in tsr_centralized_eval (which feeds predictions back
# as inputs) is well defined.  (NARMA and the Santa Fe laser are deliberately
# omitted: NARMA is input-driven, so it cannot free-run without an exogenous
# stream, and the laser needs an external recording.)

def _split_onestep(traj: np.ndarray, len_train: int = 5000, len_test: int = 1000):
    """One-step-ahead (u(t) → u(t+1)) train/test split, matching mg / lorenz."""
    traj = traj.reshape(len(traj), -1)
    u = traj[:len_train + len_test]
    v = traj[1: len_train + len_test + 1]
    return u[:len_train], v[:len_train], u[len_train:], v[len_train:]


def read_data_henon(transient: int = 1000):
    """Hénon map (2-D discrete chaos): x' = 1 − a·x² + y,  y' = b·x  (a=1.4, b=0.3)."""
    a, b = 1.4, 0.3
    n = 6001 + transient
    xy = np.empty((n, 2))
    xy[0] = (0.1, 0.1)
    for t in range(1, n):
        x, y = xy[t - 1]
        xy[t] = (1.0 - a * x * x + y, b * x)
    return _split_onestep(xy[transient:])


def read_data_logistic(transient: int = 1000):
    """Logistic map (1-D discrete chaos): x' = r·x·(1 − x)  (r = 3.9999)."""
    r = 3.9999
    n = 6001 + transient
    x = np.empty(n)
    x[0] = 0.371
    for t in range(1, n):
        x[t] = r * x[t - 1] * (1.0 - x[t - 1])
    return _split_onestep(x[transient:])


def read_data_rossler(dt: float = 0.1, transient: int = 2000):
    """Rössler attractor (3-D continuous chaos), RK4-integrated then sampled at *dt*.

    ẋ = −y − z,  ẏ = x + a·y,  ż = b + z·(x − c)   (a=0.2, b=0.2, c=5.7).
    """
    a, b, c = 0.2, 0.2, 5.7

    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    n = 6001 + transient
    traj = np.empty((n, 3))
    traj[0] = (0.0, 1.0, 1.05)
    for t in range(1, n):
        s = traj[t - 1]
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        traj[t] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return _split_onestep(traj[transient:])


# ─── Time-series classification ───────────────────────────────────────────────

def read_data_har():
    """UCI HAR dataset: 9-channel inertial signals → 6 activity classes.

    Called by: read_data("har").
    """
    sensor_signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def _load_subset(data_dir: str, subset: str) -> tuple:
        """Load sensor signals and integer labels for one split (train / test)."""
        features = []
        for sig in sensor_signals:
            fp = os.path.join(data_dir, subset, "Inertial Signals", f"{sig}_{subset}.txt")
            features.append(np.loadtxt(fp)[:, :, np.newaxis])
        X = np.concatenate(features, axis=-1)  # (samples, 128, 9)

        label_path = os.path.join(data_dir, subset, f"y_{subset}.txt")
        y = np.loadtxt(label_path).astype(int) - 1  # convert 1-6 → 0-5
        return X, y

    data_dir = get_data_path("har")
    X_train, y_train = _load_subset(data_dir, "train")
    X_test,  y_test  = _load_subset(data_dir, "test")

    # Shuffle within each split to remove label ordering bias
    tr_idx = np.random.permutation(len(X_train))
    te_idx = np.random.permutation(len(X_test))
    return X_train[tr_idx], y_train[tr_idx], X_test[te_idx], y_test[te_idx]


def read_data_char():
    """Written-character dataset: pen trajectories → 20 character classes.

    Called by: read_data("char").
    """

    def _preprocess(data_dir: str, max_len: int = 205, num_classes: int = 20,
                    shuffle: bool = True):
        data = scipy.io.loadmat(data_dir)
        mixout = data["mixout"][0]
        consts = data["consts"][0, 0]
        charlabels = consts["charlabels"][0] - 1  # 1-based → 0-based

        # Pad all trajectories to max_len using the shared helper
        seqs = _pad_sequences([m.T for m in mixout], max_len, n_features=3)

        # Restrict to the first num_classes characters (use first half for train,
        # second half for test to maintain balanced class distribution)
        if 0 < num_classes < 20:
            class_starts = [0] + [
                i for i in range(1, len(charlabels))
                if charlabels[i] != charlabels[i - 1]
            ]
            p1 = class_starts[:num_classes]
            p2 = class_starts[20: 20 + num_classes]
            seqs = np.vstack([
                seqs[p1[0]: p1[-1]],
                seqs[p2[0]: p2[-1]],
            ])
            charlabels = np.hstack([
                charlabels[p1[0]: p1[-1]],
                charlabels[p2[0]: p2[-1]],
            ])
        elif num_classes != 20:
            raise ValueError("num_classes must be in [1, 20]")

        if shuffle:
            idx = np.random.permutation(len(seqs))
            seqs, charlabels = seqs[idx], charlabels[idx]

        return seqs, charlabels

    def _train_test_split(X, y, test_size: float = 0.2):
        split = int(len(X) * (1 - test_size))
        return X[:split], y[:split], X[split:], y[split:]

    mat_path = get_data_path("char") + "mixoutALL_shifted.mat"
    X, y = _preprocess(mat_path, num_classes=20)
    return _train_test_split(X, y, test_size=0.2)


def read_data_jpv():
    """Japanese Vowels dataset: variable-length utterances → 9 speaker classes.

    Sequences are left-padded to the length of the longest sequence.
    Called by: read_data("jpv").
    """
    from reservoirpy.datasets import japanese_vowels

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = japanese_vowels()

    # Determine the maximum sequence length across both splits
    max_len = max(
        max(len(s) for s in X_train_raw),
        max(len(s) for s in X_test_raw),
    )

    # Pad both splits using the shared helper (n_features=12 for this dataset)
    X_train = _pad_sequences(X_train_raw, max_len, n_features=12)
    X_test  = _pad_sequences(X_test_raw,  max_len, n_features=12)

    # Y is one-hot → convert to integer class indices
    Y_train = np.argmax(np.array(Y_train_raw).squeeze(), axis=1)
    Y_test  = np.argmax(np.array(Y_test_raw).squeeze(),  axis=1)

    return X_train, Y_train, X_test, Y_test


# ─── UCR univariate datasets ─────────────────────────────────────────────────

# Names recognised by read_data_uni(); each maps to a subdirectory under UCR_univariate/
uni_names = [
    "ECG5000",
    "DistalPhalanxOutlineCorrect",
    "Yoga",
    "Strawberry",
]


def read_data_uni(dataset_name: str):
    """Load a UCR univariate time-series classification dataset via aeon.

    Args:
        dataset_name: Must be one of uni_names.

    Called by: read_data(name) when name is in uni_names.
    """
    from aeon.datasets import load_from_ts_file

    root = get_data_path("ucr")
    X_train, y_train = load_from_ts_file(
        root + dataset_name + "/" + dataset_name + "_TRAIN.ts", return_type="numpy3d"
    )
    X_test, y_test = load_from_ts_file(
        root + dataset_name + "/" + dataset_name + "_TEST.ts", return_type="numpy3d"
    )

    # aeon returns (samples, channels, timesteps) → swap to (samples, timesteps, channels)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test  = np.swapaxes(X_test,  1, 2)

    # Labels are strings; convert to 0-based integers
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)
    if dataset_name not in ["DistalPhalanxOutlineCorrect"]:
        y_train -= 1
        y_test  -= 1

    # Shuffle within each split
    tr_idx = np.random.permutation(len(X_train))
    te_idx = np.random.permutation(len(X_test))
    return X_train[tr_idx], y_train[tr_idx], X_test[te_idx], y_test[te_idx]


# ─── Unified entry point ──────────────────────────────────────────────────────

def read_data(name: str):
    """Load a dataset by name and return (X_train, y_train, X_test, y_test).

    This is the single dataset entry point imported by the eval and search scripts.
    """
    readers = {
        "mg":       read_data_mg,
        "lorenz":   read_data_lorenz,
        "henon":    read_data_henon,
        "logistic": read_data_logistic,
        "rossler":  read_data_rossler,
        "har":      read_data_har,
        "char":     read_data_char,
        "jpv":      read_data_jpv,
    }
    if name in readers:
        return readers[name]()
    if name in uni_names:
        return read_data_uni(name)
    from synth_data import SYNTH_CONFIGS, make_synth_named
    if name in SYNTH_CONFIGS:
        return make_synth_named(name)
    raise ValueError(f"Unknown dataset: '{name}'. "
                     f"Known: {list(readers) + uni_names + list(SYNTH_CONFIGS)}")


# ─── Shared preprocessing utilities (imported by experiment scripts) ──────────

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer label vector to one-hot matrix, shape (n, n_classes)."""
    return np.eye(n_classes)[y.astype(int)]


def standardize(Xtr: np.ndarray, *others: np.ndarray) -> tuple:
    """Feature-wise z-score normalisation; statistics computed from Xtr only.

    Works for both (n, T, F) and (n, T) shaped arrays — operates along the
    last (feature) axis.  Zero-variance features are left unchanged.

    Returns a tuple of all passed arrays normalised with Xtr statistics.
    Example: Xtr_n, Xte_n = standardize(Xtr, Xte)
             Xtr_n, Xval_n, Xte_n = standardize(Xtr, Xval, Xte)
    """
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std  = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0)
    std  = np.where(std == 0, 1.0, std)
    return tuple((X - mean) / std for X in (Xtr, *others))


if __name__ == "__main__":
    for ds in ["jpv", "har", "ECG5000"]:
        Xtr, ytr, Xte, yte = read_data(ds)
        print(f"{ds:30s}  train {Xtr.shape}  test {Xte.shape}  classes {len(set(ytr))}")
