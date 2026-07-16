# readout_node.py — custom trainable readout nodes for reservoirpy ESN pipelines.
#
# Class hierarchy:
#
#   _BaseReadoutNode (TrainableNode)
#       ├── Reg_Node   — regression readout  (MSE + optional SmoothL1)
#       └── Clr_Node   — classification readout (CE + optional SL1 / L2)
#                        ← used by client.py in the FL pipeline
#
# The base class provides the shared infrastructure (forward pass, alpha
# schedule, dimension validation) so subclasses only need to implement fit().

import numpy as np
from typing import Optional, Union, Callable

import config
from funcs import resolve_use_gpu
from reservoirpy.node import TrainableNode
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_array
from reservoirpy.utils.data_validation import check_node_input

# Loss / gradient functions — see funcs.py for implementations
from funcs import (
    mse_sl1_fval, mse_sl1_grad,
    ce_sl1_fval, ce_sl1_grad,
    ce_l2_fval, ce_l2_grad,
    ce_none_fval, ce_none_grad,
)

# L1General solver — used by Reg_Node in non-FL (offline) mode
from L1General_python.lossFuncs import make_squared_error
from L1General_python.L1General import L1GeneralUnconstrainedApx


# ─── Shared base class ────────────────────────────────────────────────────────

class _BaseReadoutNode(TrainableNode):
    """Shared structure for ESN readout nodes.

    Handles:
      • dimension validation at construction time
      • reservoirpy integration (initialize / _step / _run)
      • SmoothL1 alpha continuation schedule (update_alpha)

    Subclasses implement fit() with their task-specific objective.
    Do NOT instantiate this class directly; use Reg_Node or Clr_Node.
    """

    def __init__(
        self,
        reg_param: float,
        reg_type: str,
        alpha: float,
        thres: float,
        learning_rate: float,
        fit_bias: bool,
        Wout: Optional[Union[Weights, Callable]],
        bias: Optional[Union[Weights, Callable]],
        epochs: int,
        input_dim: Optional[int],
        output_dim: Optional[int],
        name: Optional[str],
        max_alpha: float,   # upper bound for the alpha continuation schedule
        alpha_multiplier: float,  # single geometric growth factor for alpha
        verbose: bool,
        use_gpu: bool = False,   # offload readout math to GPU (resolved vs availability)
    ):
        self.reg_param = reg_param
        self.reg_type = reg_type
        self.alpha = alpha
        self.alpha_multiplier = alpha_multiplier
        self.thres = thres
        self.learning_rate = learning_rate
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.epochs = epochs
        self.name = name
        self.state = {}
        self.max_alpha = max_alpha
        self.verbose = verbose
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Resolve once: True only if a CuPy/CUDA stack is actually present.
        self.use_gpu = resolve_use_gpu(use_gpu)

        # Validate Wout dimensions against explicitly provided input_dim / output_dim
        if is_array(Wout):
            if input_dim is not None and Wout.shape[0] != input_dim:
                raise ValueError(
                    f"'input_dim' ({input_dim}) and Wout.shape[0] ({Wout.shape[0]}) mismatch."
                )
            self.input_dim = Wout.shape[0]
            if output_dim is not None and Wout.shape[1] != output_dim:
                raise ValueError(
                    f"'output_dim' ({output_dim}) and Wout.shape[1] ({Wout.shape[1]}) mismatch."
                )
            self.output_dim = Wout.shape[1]

        if is_array(bias):
            if output_dim is not None and bias.shape[0] != output_dim:
                raise ValueError(
                    f"'output_dim' ({output_dim}) and bias.shape[0] ({bias.shape[0]}) mismatch."
                )
            self.output_dim = bias.shape[0]

    # ── reservoirpy integration ───────────────────────────────────────────────

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        """Register input/output dimensions with reservoirpy and allocate state.

        Called automatically by reservoirpy on the first data batch.
        """
        self._set_input_dim(x)
        self._set_output_dim(y)

        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)

        self.state = {"out": np.zeros((self.output_dim,))}
        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        """Single-timestep forward pass: out = x @ Wout + bias."""
        return {"out": x @ self.Wout + self.bias}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        """Full-sequence forward pass; returns (last_state, all_outputs)."""
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out

    # ── SmoothL1 continuation schedule ───────────────────────────────────────

    def update_alpha(self, epoch: int):
        """Increase the SmoothL1 smoothing parameter α after each epoch.

        Uses the single uniform growth factor self.alpha_multiplier (shared with
        the FL and Newton paths), capped at max_alpha.  A larger α makes the
        SmoothL1 penalty closer to true L1 but harder to optimise — gradual
        increase balances sparsity and convergence.
        """
        self.alpha = min(self.alpha * self.alpha_multiplier, self.max_alpha)


# ─── Regression readout ───────────────────────────────────────────────────────

class Reg_Node(_BaseReadoutNode):
    """Regression readout node: MSE objective + optional SmoothL1 regularisation.

    Two training modes (selected via the *isFL* flag in fit()):
      isFL=False  — offline, uses the L1General Newton solver (default)
      isFL=True   — federated, uses PyTorch SGD with hand-computed gradients

    Called from: tsr_centralized_search.py / tsr_centralized_eval.py (regression tasks).
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reg_type: str = "none",
        alpha_init: float = config.sl1_defaults.ALPHA_INIT,
        alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
        thres: float = config.sl1_defaults.THRES,
        learning_rate: float = 0.01,
        fit_bias: bool = False,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        epochs: int = 100,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
        use_gpu: bool = False,
    ):
        super().__init__(
            reg_param=reg_param, reg_type=reg_type, alpha=alpha_init, thres=thres,
            alpha_multiplier=alpha_multiplier,
            learning_rate=learning_rate, fit_bias=fit_bias, Wout=Wout, bias=bias,
            epochs=epochs, input_dim=input_dim, output_dim=output_dim, name=name,
            max_alpha=1e6,  # regression uses a tighter cap than classification
            verbose=False,
            use_gpu=use_gpu,
        )

    def fit(
        self,
        x: NodeInput,
        y: Optional[NodeInput] = None,
        warmup: int = 0,
        isFL: bool = False,
        useCupy: Optional[bool] = None,
    ) -> np.ndarray:
        """Train the regression readout.

        Args:
            x:     Reservoir states, shape (n_samples, timesteps, units)
                   or (n_samples, units).
            y:     Targets.
            isFL:  If True, use gradient-based update compatible with FL
                   (PyTorch SGD + mse_sl1_grad). If False, use L1General solver.
            useCupy: GPU offload for the offline SL1 (L1General) solver.  When
                   None (default) it follows the node's own ``use_gpu`` setting;
                   pass True/False to override per call.
        """
        if useCupy is None:
            useCupy = self.use_gpu
        check_node_input(x, expected_dim=self.input_dim)
        if not self.initialized:
            self.initialize(x, y)

        # Use only the last reservoir state per sequence
        if x.ndim == 3 and y.ndim == 3:
            x = x[:, -1, :]
            y = y[:, -1, :]

        if self.reg_type == "sl1" and not isFL and not self.fit_bias:
            # ── Offline: Newton continuation with per-step soft-threshold ────
            # Mirrors the classification path (Clr_Node → solve_newton_l1) and
            # the FL server: an L1General-style α continuation that soft-
            # thresholds every step by self.thres, so thres / alpha_init /
            # alpha_multiplier are honoured here (unlike the old L1General
            # solver, whose one-off hard cut used a fixed 1e-4 and ignored them).
            from newton_solver import solve_newton_mse_l1

            self.Wout = solve_newton_mse_l1(
                x, y, np.zeros((self.input_dim, self.output_dim)),
                self.reg_param,
                alpha_init=self.alpha,
                alpha_max=self.max_alpha,
                update1=self.alpha_multiplier,
                update2=self.alpha_multiplier,
                thres=self.thres,
                max_iter=self.epochs,
                verbose=self.verbose,
            )

        elif self.reg_type == "sl1" and not isFL and self.fit_bias:
            # ── Offline with fit_bias: L1General Newton solver ───────────────
            # The per-step soft-threshold path above cannot express an
            # *unregularised* bias (the SmoothL1 penalty uses a single scalar λ),
            # so a fitted bias still routes through L1General, which supports a
            # per-coordinate lambda_vec with the bias entry zeroed.  No caller
            # currently sets fit_bias=True for sl1; kept for correctness.
            X_aug = np.hstack([x, np.ones((x.shape[0], 1))])
            p = X_aug.shape[1]
            lambda_vec = self.reg_param * np.ones(p)
            lambda_vec[-1] = 0.0  # do not regularise the bias term

            _Wout = np.zeros((self.input_dim, self.output_dim))
            _bias = np.zeros(self.output_dim)

            make_obj = make_squared_error(X_aug, y, use_cupy=useCupy)
            options  = {"verbose": 0, "mode": 0, "progTol": 1e-12}
            for i in range(self.output_dim):
                W0 = np.zeros((p, 1))
                W_opt, _ = L1GeneralUnconstrainedApx(
                    make_obj(i), W0, lambda_vec.copy(), options)
                _Wout[:, i] = W_opt[:-1].reshape(-1)
                _bias[i] = W_opt[-1, 0]

            self.Wout = _Wout
            self.bias = _bias

        elif self.reg_type == "sl1" and isFL:
            # ── Federated: gradient-based update via PyTorch SGD ─────────────
            import torch
            from torch.optim import SGD

            _Wout = torch.tensor(self.Wout, requires_grad=True)
            optimizer = SGD([_Wout], lr=self.learning_rate)

            for e in range(self.epochs):
                w_np = _Wout.detach().numpy()
                # Compute gradient from combined MSE + SmoothL1 objective
                w_grad = mse_sl1_grad(w_np, x, y, self.alpha, self.reg_param)
                self.update_alpha(e)

                optimizer.zero_grad()
                _Wout.grad = torch.tensor(w_grad)
                optimizer.step()

                self.Wout = _Wout.detach().numpy()
                self.Wout[np.abs(self.Wout) < self.thres] = 0.0  # hard threshold

        elif self.reg_type == "l2":
            from sklearn.linear_model import Ridge
            self._fit_sklearn(
                x, y, Ridge(alpha=self.reg_param, fit_intercept=self.fit_bias)
            )

        elif self.reg_type == "l1":
            import warnings
            from sklearn.linear_model import Lasso
            from sklearn.exceptions import ConvergenceWarning
            # ESN reservoir states are strongly collinear, so Lasso coordinate
            # descent converges slowly. A raised cap lets well-regularised configs
            # converge cleanly; under-regularised ones (tiny reg_param ≈ OLS on
            # collinear states) can still stall — but those produce poor forecasts
            # and are dropped by the search's selection metric, so the resulting
            # ConvergenceWarning is expected noise here and is suppressed.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                self._fit_sklearn(
                    x, y, Lasso(alpha=self.reg_param, fit_intercept=self.fit_bias,
                                max_iter=10000)
                )

        elif self.reg_type == "none":
            # Plain OLS.  reg_param is ignored.  (Using Lasso(alpha=0) here would
            # trip sklearn's ill-conditioning warning and converge poorly under
            # coordinate descent, so solve the unregularised problem directly.)
            from sklearn.linear_model import LinearRegression
            self._fit_sklearn(
                x, y, LinearRegression(fit_intercept=self.fit_bias)
            )

        return self.Wout

    def _fit_sklearn(self, x: np.ndarray, y: np.ndarray, estimator) -> None:
        """Train Wout (and optionally bias) per output column using a sklearn estimator."""
        _Wout = np.zeros((self.input_dim, self.output_dim))
        _bias = np.zeros(self.output_dim) if self.fit_bias else None
        for i in range(self.output_dim):
            estimator.fit(x, y[:, i])
            _Wout[:, i] = estimator.coef_.copy()
            if self.fit_bias:
                _bias[i] = estimator.intercept_
        self.Wout = _Wout
        if self.fit_bias:
            self.bias = _bias


# ─── Classification readout ───────────────────────────────────────────────────

class Clr_Node(_BaseReadoutNode):
    """Classification readout node: cross-entropy + optional SL1 / L2 regularisation.

    Two solvers (selected via the *solver* argument):
      solver="newton"  — L1General-style Newton-path on CE + SmoothL1 (default
                         for sl1; second-order, far sharper sparsity).  Shares
                         newton_solver.solve_newton_l1 with the FL server.
      solver="adam"    — first-order PyTorch Adam loop with hand-computed
                         gradients (works for sl1 / l2 / none; CuPy-friendly).

    Called from: client.py → Client_TSC.initialize_model() (FL: only used as a
                 Wout container + forward pass — neither solver runs there, the
                 server takes the Newton step).
                 tsc_centralized_search.py → fit_from_states() (centralized).
    """

    def __init__(
        self,
        reg_param: float = 0.0,
        reg_type: str = "sl1",
        alpha_init: float = config.sl1_defaults.ALPHA_INIT,
        alpha_multiplier: float = config.sl1_defaults.ALPHA_MULTIPLIER,
        thres: float = config.sl1_defaults.THRES,
        learning_rate: float = 0.01,
        fit_bias: bool = False,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        epochs: int = 100,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
        verbose: bool = False,
        solver: str = "newton",   # "newton" (sl1 only) or "adam"
        patience: int = config.sl1_defaults.PATIENCE,
        stag_tol: float = config.sl1_defaults.STAG_TOL,
        use_gpu: bool = False,
    ):
        super().__init__(
            reg_param=reg_param, reg_type=reg_type, alpha=alpha_init, thres=thres,
            alpha_multiplier=alpha_multiplier,
            learning_rate=learning_rate, fit_bias=fit_bias, Wout=Wout, bias=bias,
            epochs=epochs, input_dim=input_dim, output_dim=output_dim, name=name,
            max_alpha=config.sl1_defaults.ALPHA_MAX,  # classification α ceiling
            verbose=verbose,
            use_gpu=use_gpu,
        )
        self.solver   = solver
        self.patience = patience   # stagnation early-stop for the Newton path
        self.stag_tol = stag_tol

    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> np.ndarray:
        """Train from reservoir-state sequences via the reservoirpy pipeline.

        Args:
            x:  List/array of state sequences, each (timesteps, units).
            y:  One-hot labels, shape (n_samples, n_classes).
        """
        check_node_input(x, expected_dim=self.input_dim)
        if not self.initialized:
            self.initialize(x, y)
        last_x = np.array([s[-1] for s in x])
        return self._train(last_x, np.array(y).squeeze())

    def fit_from_states(self, X_states: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
        """Train from pre-extracted last reservoir states, bypassing the pipeline.

        Use when you have already run the reservoir and collected the last hidden
        state per sequence (shape: n_samples × units).  This avoids the
        reservoirpy pipeline API which passes label arrays as the readout input.

        Called from: tsc_centralized_search.py
        """
        if not self.initialized:
            self.input_dim  = X_states.shape[1]
            self.output_dim = y_oh.shape[1]
            self.Wout = np.zeros((self.input_dim, self.output_dim))
            self.bias = np.zeros(self.output_dim)
            self.state = {"out": np.zeros((self.output_dim,))}
            self.initialized = True
        return self._train(X_states, y_oh)

    def _train(self, X_states: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
        """Dispatch to the configured solver (Newton path or Adam loop).

        With solver="newton" (default) every reg_type is solved by the same
        Gauss-Newton family the FL server uses — sl1 via the SmoothL1
        continuation loop, l2/none via the plain smooth Newton loop — so the
        centralized and federated paths share one optimiser. solver="adam"
        falls back to the SGD/Adam loop for l2/none (sl1 always uses Newton).
        """
        if self.solver == "newton":
            if self.reg_type == "sl1":
                return self._newton_loop(X_states, y_oh)
            return self._newton_smooth(X_states, y_oh)
        if self.reg_type == "sl1":
            # sl1 has no Adam path; the SmoothL1 objective needs the Newton loop.
            return self._newton_loop(X_states, y_oh)
        return self._adam_loop(X_states, y_oh)

    def _newton_smooth(self, X_states: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
        """Damped Gauss-Newton solver for the smooth objectives (l2 / none).

        Delegates to newton_solver.solve_newton_smooth, which reuses the same
        per-class direction/line-search primitives as the sl1 and FL paths but
        without α continuation or soft-thresholding (neither applies to the
        smooth CE+L2 / CE objectives).
        """
        from newton_solver import solve_newton_smooth

        self.Wout = solve_newton_smooth(
            X_states, y_oh, self.Wout, self.reg_param, self.reg_type,
            max_iter=self.epochs,
            verbose=self.verbose,
            use_gpu=self.use_gpu,
        )
        return self.Wout

    def _newton_loop(self, X_states: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
        """L1General-style Newton-path solver for CE + SmoothL1.

        Delegates to newton_solver.solve_newton_l1 — the same per-class
        Gauss-Newton machinery the FL server uses for its aggregation step.
        Starts the α continuation small (≈1) and grows it toward max_alpha,
        matching L1General's schedule rather than the Adam path's fixed α.
        """
        from newton_solver import solve_newton_l1

        self.Wout = solve_newton_l1(
            X_states, y_oh, self.Wout, self.reg_param,
            alpha_init=self.alpha,
            alpha_max=self.max_alpha,
            update1=self.alpha_multiplier,
            update2=self.alpha_multiplier,
            thres=self.thres,
            max_iter=self.epochs,
            patience=self.patience,
            stag_tol=self.stag_tol,
            verbose=self.verbose,
            use_gpu=self.use_gpu,
        )
        return self.Wout

    def _adam_loop(self, X_states: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
        """Core Adam optimisation loop shared by fit() and fit_from_states()."""
        import torch
        from torch.optim import Adam

        if self.reg_type == "sl1":
            fval_func, grad_func = ce_sl1_fval, ce_sl1_grad
        elif self.reg_type == "l2":
            fval_func, grad_func = ce_l2_fval, ce_l2_grad
        elif self.reg_type == "none":
            fval_func, grad_func = ce_none_fval, ce_none_grad
        else:
            raise ValueError(f"Unsupported reg_type: '{self.reg_type}'")

        _Wout     = torch.tensor(self.Wout, requires_grad=True)
        optimizer = Adam([_Wout], lr=self.learning_rate)

        for e in range(self.epochs):
            w_np = _Wout.detach().numpy()
            optimizer.zero_grad()

            if self.reg_type == "sl1":
                loss   = fval_func(w_np, X_states, y_oh, self.alpha, self.reg_param,
                                   use_cupy=self.use_gpu)
                w_grad = grad_func(w_np, X_states, y_oh, self.alpha, self.reg_param,
                                   use_cupy=self.use_gpu)
                self.update_alpha(e)
            else:
                loss   = fval_func(w_np, X_states, y_oh, self.reg_param,
                                   use_cupy=self.use_gpu)
                w_grad = grad_func(w_np, X_states, y_oh, self.reg_param,
                                   use_cupy=self.use_gpu)

            _Wout.grad = torch.tensor(w_grad)
            optimizer.step()

            # Soft-threshold (L1 proximal step) in place on the optimiser tensor's
            # shared buffer, so the next Adam step proceeds from the thresholded
            # point (proximal-Adam) and matches the Newton/FL soft-threshold op.
            self.Wout = _Wout.detach().numpy()
            np.copyto(
                self.Wout,
                np.sign(self.Wout) * np.maximum(np.abs(self.Wout) - self.thres, 0.0),
            )

            if e % 100 == 0 and self.verbose:
                print(f"  epoch {e:4d}: loss={loss:.4f}, nonzero={np.count_nonzero(self.Wout)}")

        return self.Wout


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from reservoirpy.nodes import Reservoir
    from reservoirpy.datasets import japanese_vowels

    X_train, X_test, Y_train, Y_test = japanese_vowels()

    reservoir = Reservoir(500, sr=0.9, lr=0.1)

    W_init = np.random.normal(size=(500, 9))
    b_init = np.zeros(9)

    readout = Clr_Node(
        reg_param=1e-2,
        reg_type="sl1",
        thres=1e-5,
        learning_rate=1e-2,
        Wout=W_init,
        bias=b_init,
        fit_bias=False,
        epochs=5000,
        input_dim=500,
        output_dim=9,
        verbose=True,
    )

    model = reservoir >> readout
    model.fit(X_train, Y_train)

    full_s = model.run(X_test)
    digits_test = np.array([s[-1] for s in full_s])
    y_pred = np.argmax(digits_test, axis=1)
    Y_test_int = np.argmax(np.array(Y_test).squeeze(), axis=1)
    acc = (y_pred == Y_test_int).mean()
    sparsity = (readout.Wout == 0).mean() * 100
    print(f"Accuracy: {acc:.4f}  |  Wout sparsity: {sparsity:.1f}%")
