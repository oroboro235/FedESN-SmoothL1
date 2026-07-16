# server.py — Newton-step aggregation server for the FedSL1ESN pipeline.
#
# Responsibilities:
#   1. initialize_global_model() — generate shared reservoir weights (Win, W).
#   2. aggregate_parameters()    — aggregate client gradients & Hessians,
#                                  solve a Newton step per output class,
#                                  apply sparsity threshold, update global Wout.
#
# The server never trains locally; its only update is the Newton aggregation step.
# Called from: tsc_fl_eval.py → main()

from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from reservoirpy.nodes import Reservoir
from reservoirpy.mat_gen import uniform

from newton_solver import damped_newton_direction, armijo_line_search


class Server_FedAvg:
    """Newton-step aggregation server for Echo State Network federated learning.

    Each round clients send (gradient, Hessian) evaluated at the current global
    Wout.  The server aggregates them as:
        ΔW_k = (ΣH_k)⁻¹ (Σg_k)   for each output class k
        Wout_k ← Wout_k − ΔW_k

    Owns the global reservoir (frozen after initialisation) and the global
    readout Wout (updated each round via the Newton step).

    Args:
        hyperparams: Per-dataset hyperparameter dict loaded from configs/.
                     Must contain: units, lr, sr, input_dim, input_scaling,
                     input_connectivity, rc_connectivity, seed,
                     output_dim, global_lr, thres, reg_param, reg_type.
    """

    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams
        self.global_lr        = hyperparams["global_lr"]
        self.res: Any         = None   # Reservoir node — set by initialize_global_model()
        self.global_Wout: Any = None   # csr_matrix   — set by initialize_global_model()

    # The per-class damped Newton solve and Armijo line search live in
    # newton_solver.py so the centralized Clr_Node solver and this server share
    # one implementation (the server is the single-step, multi-client case).

    # ── Setup ──────────────────────────────────────────────────────────────────

    def initialize_global_model(self):
        """Generate the shared reservoir weights (Win, W) and zero-initialise Wout.

        The reservoir is built once and its weights are broadcast to every client
        via client.initialize_model(output_dim, server.res.Win, server.res.W).
        Win and W are never updated; only Wout participates in federation.

        Called from: tsc_fl_eval.py → main() before client setup.
        """
        hp = self.hyperparams
        self.res = Reservoir(
            units              = hp["units"],
            lr                 = hp["lr"],
            sr                 = hp["sr"],
            Win                = uniform,
            W                  = uniform,
            input_dim          = hp["input_dim"],
            input_scaling      = hp["input_scaling"],
            input_connectivity = hp["input_connectivity"],
            rc_connectivity    = hp["rc_connectivity"],
            seed               = hp["seed"],
        )
        # reservoirpy generates Win and W lazily on the first forward pass
        self.res.initialize(np.zeros((1, hp["input_dim"])))

        self.global_Wout = csr_matrix(np.zeros((hp["units"], hp["output_dim"])))

    # ── FL round step ─────────────────────────────────────────────────────────

    def _line_search(self, current_w: np.ndarray, direction: np.ndarray,
                     agg_grad: np.ndarray, clients: list,
                     c: float = 1e-4, rho: float = 0.5,
                     max_iter: int = 20) -> float:
        """Armijo backtracking line search over the aggregated global loss.

        Finds the largest α in {1, ρ, ρ², …} satisfying the Armijo condition:
            f(w + α·d) ≤ f(w) + c·α·⟨G, D⟩_F
        where f is the sum of client local losses, G is the aggregated gradient,
        and D is the Newton descent direction.

        Reservoir states are already cached in each client from train(), so
        compute_local_loss() only re-evaluates the readout loss — no extra
        reservoir forward pass is needed.

        Args:
            current_w:  Current global Wout array, shape (units, n_classes).
            direction:  Newton descent direction D = -H⁻¹g, same shape.
            agg_grad:   Aggregated gradient G, same shape.
            clients:    List of Client_TSC instances (must have called train()).
            c:          Armijo sufficient-decrease constant (default 1e-4).
            rho:        Step shrinkage factor per backtrack iteration (default 0.5).
            max_iter:   Maximum number of backtracks before returning last α.

        Returns:
            Scalar step size α ∈ (0, 1].

        Called from: aggregate_parameters() when use_line_search=True.
        """
        # The aggregated global loss is the sum of client local losses; reservoir
        # states are cached in each client, so this only re-evaluates the readout.
        loss_fn = lambda w: sum(cl.compute_local_loss(w) for cl in clients)
        return armijo_line_search(loss_fn, current_w, direction, agg_grad,
                                  c=c, rho=rho, max_iter=max_iter)

    def aggregate_parameters(self, collected_params: list,
                             clients: Optional[list] = None,
                             use_line_search: bool = False,
                             ls_c: float = 1e-4,
                             ls_rho: float = 0.5,
                             ls_max_iter: int = 20) -> tuple:
        """Newton-step aggregation: solve (ΣH_k)⁻¹(Σg_k) per output class.

        Steps:
            1. Sum client gradients and per-class Hessian blocks.
            2. Solve the Newton equation per class; build the descent direction D.
            3. (Optional) Armijo backtracking line search to choose step size α.
            4. Apply update: new_Wout = current + α · D.
            5. Zero out entries with |w| < thres (promote sparsity).

        Args:
            collected_params: list of (grad, hessian) from each client.
                grad:    np.ndarray (units, n_classes)
                hessian: list of n_classes np.ndarray each (units, units)
            clients:       List of Client_TSC instances; required when
                           use_line_search=True so _line_search() can query
                           local loss values.
            use_line_search: Enable Armijo backtracking (default False).
            ls_c:          Armijo sufficient-decrease constant (default 1e-4).
            ls_rho:        Step shrinkage factor per backtrack (default 0.5).
            ls_max_iter:   Max backtrack iterations (default 20).

        Returns:
            (global_Wout, alpha):
                global_Wout — updated sparse csr_matrix, broadcast to clients.
                alpha       — step size used (1.0 when line search is disabled).

        Called from: tsc_fl_eval.py → main() after each local training round.
        """
        hp      = self.hyperparams
        n_cls   = hp["output_dim"]
        current = self.global_Wout.toarray()   # (units, n_classes)
        units   = current.shape[0]

        # Step 1: sum gradients and Hessian blocks across all clients
        agg_grad = sum(
            (g for g, _ in collected_params),
            np.zeros((units, n_cls))
        )
        agg_hessian = [
            sum((h[k] for _, h in collected_params), np.zeros((units, units)))
            for k in range(n_cls)
        ]

        # Convergence guard: if the aggregated gradient is already tiny, the model
        # has effectively converged.  Taking a Newton step from here amplifies
        # numerical noise and causes weight drift — skip the update instead.
        grad_norm = float(np.linalg.norm(agg_grad, 'fro'))
        if grad_norm < 1e-6:
            return self.global_Wout, 0.0

        # Step 2: solve Newton equation per class, build descent direction D
        #   d_k = −(H_k + μI)⁻¹ g_k  — damped_newton_direction handles the
        #   adaptive ridge per class (shared with the centralized solver).
        direction = damped_newton_direction(agg_hessian, agg_grad)

        # Step 3: choose step size via Armijo line search or use α=1
        if use_line_search and clients is not None:
            alpha = self._line_search(current, direction, agg_grad, clients,
                                      c=ls_c, rho=ls_rho, max_iter=ls_max_iter)
        else:
            alpha = 1.0

        # Step 4: apply update
        new_Wout = current + alpha * direction

        # Step 5: hard sparsity threshold
        # new_Wout[np.abs(new_Wout) < hp["thres"]] = 0.0
        # soft sparsity threshold (alternative):
        new_Wout = np.sign(new_Wout) * np.maximum(np.abs(new_Wout) - hp["thres"], 0.0)

        self.global_Wout = csr_matrix(new_Wout)
        return self.global_Wout, alpha
