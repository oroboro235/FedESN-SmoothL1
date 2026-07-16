# client.py — FL client for time-series classification with an ESN readout.
#
# Pipeline per client (called from tsc_fl_eval.py each round):
#   1. receive_hyperparams()  — store the server's hyperparameter dict
#   2. initialize_model()     — build Reservoir >> Clr_Node with shared Win / W
#   3. train()                — compute gradient & Hessian at current Wout, return to server
#   4. receive_parameters()   — replace local Wout with the server's Newton-updated global
#   5. evaluate()             — run inference and report accuracy + sparsity

from functools import partial
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from reservoirpy.nodes import Reservoir
from readout_node import Clr_Node
import funcs
import metrics


class Client_TSC:
    """Federated learning client for time-series classification.

    The reservoir weights (Win, W) are frozen and shared across all clients.
    Each round the client computes the gradient and block-diagonal Gauss-Newton
    Hessian of its local cross-entropy loss w.r.t. the current global Wout,
    and sends both to the server for Newton-step aggregation.

    Args:
        client_id: Unique integer identifier.
        data:      (X_train, y_train, X_test, y_test) for this client's shard.
        seed:      Random seed forwarded to the Reservoir constructor.
        use_gpu:   Offload the readout gradient/Hessian/loss math to the GPU via
                   CuPy (resolved against availability — falls back to CPU when
                   CuPy/CUDA is missing).  The reservoir forward pass stays on CPU.
    """

    def __init__(self, client_id: int, data: tuple, seed: int = None,
                 use_gpu: bool = False):
        self.client_id = client_id
        self.X_train, self.y_train = data[0], data[1]
        self.X_test,  self.y_test  = data[2], data[3]
        self.model: Any       = None   # reservoirpy pipeline: Reservoir >> Clr_Node
        self.hyperparams: Any = None   # set by receive_hyperparams()
        self.local_lr    = 1.0    # set by receive_hyperparams()
        self.seed        = seed
        # Resolve once at construction so every grad/Hessian/loss call reuses it.
        self.use_gpu     = funcs.resolve_use_gpu(use_gpu)
        self._cached_X_states: Any = None  # reservoir states saved after train(), reused by line search
        # The reservoir (Win, W) is frozen across all rounds, so the last hidden
        # state per sequence never changes — compute once and reuse every round.
        self._train_states: Any = None  # cached last reservoir states for X_train
        self._test_states:  Any = None  # cached last reservoir states for X_test
        self.alpha: float = 1.0   # current SL1 smoothing param; initialised by receive_hyperparams(), updated by update_alpha()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def receive_hyperparams(self, hyperparams: dict):
        """Store the server's hyperparameter dict before model initialisation.

        Must be called before initialize_model().
        Called from: tsc_fl_eval.py → main() during client setup.
        """
        self.hyperparams = hyperparams
        self.local_lr    = hyperparams["local_lr"]
        self.alpha       = float(hyperparams.get("alpha_init", 1.0))

    def update_alpha(self, multiplier: float = 2.0, alpha_max: float = 1e5):
        """Multiply the SL1 smoothing parameter by *multiplier* after each FL round.

        This implements a geometric continuation schedule:
            alpha_t+1 = min(alpha_t * multiplier, alpha_max)

        Starting from a small alpha_init (≈ 1), alpha grows each round, making
        the SmoothL1 penalty a progressively better approximation of L1 and
        increasing the sparsifying pressure on the readout weights.

        Called from: tsc_fl_eval.py → main() at the END of each FL round,
        AFTER evaluation, so that the current round uses the current alpha and
        the next round starts with the updated value.
        """
        self.alpha = min(self.alpha * multiplier, alpha_max)

    def initialize_model(self, num_classes: int, Win, Wres):
        """Build the ESN pipeline: Reservoir >> Clr_Node.

        The reservoir uses the server-generated Win and Wres so that all
        clients share the same fixed feature extractor; only Wout differs.

        Args:
            num_classes: Number of output classes (readout output dimension).
            Win:         Input weight matrix from the server (sparse).
            Wres:        Recurrent weight matrix from the server (sparse).

        Called from: tsc_fl_eval.py → main() during client setup.
        """
        if Win is None or Wres is None:
            raise ValueError("Win and Wres must be provided by the server.")
        if self.hyperparams is None:
            raise RuntimeError("Call receive_hyperparams() before initialize_model().")

        hp = self.hyperparams

        reservoir = Reservoir(
            units              = hp["units"],
            lr                 = hp["lr"],
            sr                 = hp["sr"],
            W                  = Wres,
            Win                = Win,
            input_scaling      = hp["input_scaling"],
            input_connectivity = hp["input_connectivity"],
            rc_connectivity    = hp["rc_connectivity"],
            seed               = self.seed,
        )

        readout = Clr_Node(
            reg_param    = hp["reg_param"],
            reg_type     = hp["reg_type"],
            thres        = hp["thres"],
            Wout         = np.zeros((hp["units"], num_classes)),
            bias         = np.zeros(num_classes),
            fit_bias     = False,
            epochs       = hp["local_epochs"],
            input_dim    = hp["units"],
            output_dim   = num_classes,
        )

        self.model = reservoir >> readout

    # ── FL round helpers ───────────────────────────────────────────────────────

    def train(self) -> tuple:
        """Compute gradient and block-diagonal Hessian at the current Wout.

        Runs only the reservoir (no weight update) to extract the last hidden
        state per training sequence, then evaluates the Softmax cross-entropy
        gradient and per-class Gauss-Newton Hessian blocks.

        The reservoir states are cached in self._cached_X_states so that
        compute_local_loss() can reuse them during the server's line search
        without repeating the reservoir forward pass.

        Returns:
            grad:    np.ndarray of shape (units, n_classes) — gradient of local
                     cross-entropy loss w.r.t. Wout.
            hessian: list of n_classes np.ndarray each of shape (units, units) —
                     per-class Gauss-Newton Hessian blocks.

        Called from: tsc_fl_eval.py → _train_client() via parallelbar.
        """
        if self._train_states is None:
            # Reservoir is frozen, so this forward pass is identical every round;
            # compute it on the first round only.
            self._train_states = self._get_last_reservoir_states(self.X_train)
        X_states = self._train_states
        self._cached_X_states = X_states   # cache for server-side line search
        return self._compute_grad_hessian(X_states, self.y_train)

    def _get_last_reservoir_states(self, X) -> np.ndarray:
        """Run the full pipeline; collect only the last reservoir state per sequence.

        We run the complete pipeline (reservoir >> readout) rather than the
        reservoir node alone to ensure the node is properly initialised on the
        first call.  After each model.run(x), nodes[0].state holds the last
        hidden state of that sequence as a (1, units) array.

        The pipeline is reset between sequences so that each sample starts from
        the zero initial condition (reservoirpy does NOT auto-reset between runs).
        """
        states = []
        for x in X:
            self.model.run(x)        # initialises nodes on first call; runs full ESN
            # nodes[0].state is (1, units) after a run; take the single row
            # states.append(np.asarray(self.model.nodes[0].state)[-1].copy())
            states.append(self.model.nodes[0].state["out"].copy())
            self.model.reset()       # safe here because state was just created
        return np.array(states)      # (n_samples, units)

    def _resolve_fns(self) -> tuple:
        """Return (fval_fn, grad_fn, hess_fn) bound to current hyperparams and alpha.

        Each returned callable has signature f(w, X, y) so callers need no
        knowledge of reg_type, reg_param, or alpha.  Called by
        _compute_grad_hessian() and compute_local_loss() to avoid repeating
        the same if/elif/else dispatch in both places.
        """
        hp        = self.hyperparams
        reg_type  = hp["reg_type"]
        reg_param = hp["reg_param"]
        alpha     = self.alpha
        gpu       = self.use_gpu

        if reg_type == "sl1":
            return (
                partial(funcs.ce_sl1_fval, alpha=alpha, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_sl1_grad, alpha=alpha, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_sl1_hess, alpha=alpha, _lambda=reg_param, use_cupy=gpu),
            )
        elif reg_type == "l2":
            return (
                partial(funcs.ce_l2_fval, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_l2_grad, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_l2_hess, _lambda=reg_param, use_cupy=gpu),
            )
        else:
            return (
                partial(funcs.ce_none_fval, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_none_grad, _lambda=reg_param, use_cupy=gpu),
                partial(funcs.ce_none_hess, _lambda=reg_param, use_cupy=gpu),
            )

    def _compute_grad_hessian(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Gradient and per-class Gauss-Newton Hessian of the local loss at current Wout.

        Args:
            X: Reservoir last states, shape (n_samples, units).
            y: One-hot labels, shape (n_samples, n_classes).

        Returns:
            grad:    (units, n_classes)
            hessian: list of n_classes arrays each (units, units)
        """
        Wout = self.model.nodes[1].Wout
        _, grad_fn, hess_fn = self._resolve_fns()
        return grad_fn(Wout, X, y), hess_fn(Wout, X, y)

    def compute_local_loss(self, Wout_candidate: np.ndarray) -> float:
        """Evaluate local loss at a candidate Wout without modifying the model.

        Called by the server's Armijo line search to evaluate f(w + α*d).
        Reservoir states are Wout-independent, so only the loss function is
        re-evaluated at the candidate point using the states cached by train().

        Args:
            Wout_candidate: Candidate readout weights, shape (units, n_classes).

        Returns:
            Scalar local loss value.

        Called from: server.py → Server_FedAvg._line_search().
        """
        if self._cached_X_states is None:
            raise RuntimeError("Call train() before compute_local_loss().")
        fval_fn, _, _ = self._resolve_fns()
        return fval_fn(Wout_candidate, self._cached_X_states, self.y_train)

    def receive_parameters(self, global_Wout: csr_matrix):
        """Replace the local readout Wout with the server's Newton-updated global.

        local_lr=1.0 (default) copies the global Wout directly.
        Smaller values retain a fraction of the previous local weights via EMA:
            Wout_new = (1 - local_lr) * Wout_local + local_lr * Wout_global

        Called from: tsc_fl_eval.py → main() after each aggregation step.
        """
        readout = self.model.nodes[1]
        readout.Wout = (
            (1 - self.local_lr) * readout.Wout
            + self.local_lr * global_Wout.toarray()
        )

    def evaluate(self) -> dict:
        """Run inference on the test shard and return accuracy and sparsity.

        Returns:
            dict with keys: 'client_id', 'acc' (%), 'macro_f1' (%),
            'balanced_acc' (%), 'sparsity' (%).  macro-F1 / balanced accuracy
            weight every class equally, so they stay honest on the imbalanced
            datasets (ECG5000, DistalPhalanx…) where plain accuracy inflates.

        Called from: tsc_fl_eval.py → _evaluate_client() via parallelbar.
        """
        # The reservoir is frozen, so the last hidden state per test sequence is
        # constant across rounds — extract it once and reuse. Only Wout changes,
        # and the readout is linear (out = states @ Wout, bias=0), so predictions
        # are argmax(states @ Wout) without re-running the reservoir.
        if self._test_states is None:
            self._test_states = self._get_last_reservoir_states(self.X_test)
        Wout    = self.model.nodes[1].Wout
        y_pred  = np.argmax(self._test_states @ Wout, axis=1)
        y_true  = np.argmax(np.array(self.y_test).squeeze(), axis=1)

        scores   = metrics.classification_scores(y_true, y_pred, Wout.shape[1])
        sparsity = (Wout == 0).mean() * 100

        return {"client_id": self.client_id, "sparsity": sparsity, **scores}
