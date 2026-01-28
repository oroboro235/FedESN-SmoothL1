from typing import Any, Optional, Sequence, Union, Callable

from functools import partial
from L1General_python.lossFuncs import SquaredError, SquaredError_noCupy, softmaxLoss2, softmaxLoss2_noCupy
from L1General_python.L1General import L1GeneralUnconstrainedApx

import numpy as np

from reservoirpy.node import ParallelNode, TrainableNode
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_array

from reservoirpy.utils.data_validation import (
    check_node_input,
    check_timeseries,
    check_timestep,
)

class SmoothL1_tsr_Newton_Node(TrainableNode):
    '''
    SmoothL1 Regularization
    Objective: Sum of Squared Error Loss
    Optimizer: Newton method with backtracking line search
    '''
    reg_param: float
    fit_bias: bool
    Wout: Weights
    bias: Weights

    def __init__(
        self,
        reg_param: float = 0.0,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.reg_param = reg_param
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.name = name
        self.state = {}

        self.input_dim = input_dim
        self.output_dim = output_dim

        if is_array(Wout):
            if input_dim is not None and Wout.shape[0] != input_dim:
                raise ValueError(
                    f"Both 'input_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {input_dim} != {Wout.shape[0]}."
                )
            self.input_dim = Wout.shape[0]
            if output_dim is not None and Wout.shape[1] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {Wout.shape[1]}."
                )
            self.output_dim = Wout.shape[1]
        if is_array(bias):
            if output_dim is not None and bias.shape[0] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {bias.shape[0]}."
                )
            self.output_dim = bias.shape[0]

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        self._set_input_dim(x)
        self._set_output_dim(y)

        # initialize matrices
        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)
        self.state = {"out": np.zeros((self.output_dim,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out
    
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0):
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y, expected_dim=self.output_dim)

        if not self.initialized:
            self.initialize(x, y)

        if isinstance(x, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            x = np.concatenate(x, axis=0)
        if is_array(x) and x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        if isinstance(y, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            y = np.concatenate(y, axis=0)
        if is_array(y) and y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        # implement the Newton method and Backtracking line search
        if self.fit_bias:
            # add a column of ones before the X
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        if self.fit_bias:
            Wout_temp = np.zeros((self.input_dim+1, self.output_dim))
        else:
            Wout_temp = np.zeros((self.input_dim, self.output_dim))
        
        # for i in range(self.output_dim):
        #     if self.fit_bias:
        #         Wout_init_1d = np.zeros((self.input_dim+1, 1))
        #     else:
        #         Wout_init_1d = np.zeros((self.input_dim, 1))

        #     _lambda_vec = self.reg_param * np.ones(self.input_dim)
        #     if self.fit_bias:
        #         _lambda_vec = np.hstack((0.0, _lambda_vec))

        #     options = {}
        #     options["verbose"] = 1
        #     options["mode"] = 0
        #     options["progTol"] = 1e-12
        #     Wout_opt, _ = L1GeneralUnconstrainedApx(
        #         partial(SquaredError_noCupy, X=x, y=y[:, i]),   # Objective function
        #         Wout_init_1d,                                   # Initial guess of Wout in one dimension
        #         _lambda_vec,                                    # regularization parameter in vector
        #         options,
        #     )

        #     Wout_temp[:, i] = Wout_opt.reshape(-1)

        Wout_temp_flat = Wout_temp.ravel(order='F').reshape(-1, 1)

        _lambda_vec = self.reg_param * np.ones(self.input_dim)
        if self.fit_bias:
            _lambda_vec = np.hstack((0.0, _lambda_vec))

        repeat_lambda_vec = np.repeat(_lambda_vec.reshape(-1, 1), self.output_dim, axis=1)
        _lambda_vec_flat = repeat_lambda_vec.ravel(order='F').reshape(-1, 1)

        options = {}
        options["verbose"] = 1
        options["mode"] = 0
        options["progTol"] = 1e-12

        Wout_opt_flat, _ = L1GeneralUnconstrainedApx(
            partial(SquaredError_noCupy, X=x, y=y),   # Objective function
            Wout_temp_flat,                                   # Initial guess of Wout in one dimension
            _lambda_vec_flat,                                    # regularization parameter in vector
            options,
        )

        if self.fit_bias:
            Wout_temp = Wout_opt_flat.reshape((self.input_dim+1, self.output_dim))
        else:
            Wout_temp = Wout_opt_flat.reshape((self.input_dim, self.output_dim))

        
        if self.fit_bias:
            self.Wout = Wout_temp[1:, :]
            self.bias = Wout_temp[0, :]
        else:
            self.Wout = Wout_temp

class SmoothL1_tsc_Newton_Node(TrainableNode):
    '''
    SmoothL1 Regularization
    Objective: Softmax Loss
    Optimizer: Newton method with backtracking line search
    '''
    reg_param: float
    fit_bias: bool
    Wout: Weights
    bias: Weights

    def __init__(
        self,
        reg_param: float = 0.0,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.reg_param = reg_param
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.name = name
        self.state = {}

        self.input_dim = input_dim
        self.output_dim = output_dim

        if is_array(Wout):
            if input_dim is not None and Wout.shape[0] != input_dim:
                raise ValueError(
                    f"Both 'input_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {input_dim} != {Wout.shape[0]}."
                )
            self.input_dim = Wout.shape[0]
            if output_dim is not None and Wout.shape[1] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {Wout.shape[1]}."
                )
            self.output_dim = Wout.shape[1]
        if is_array(bias):
            if output_dim is not None and bias.shape[0] != output_dim:
                raise ValueError(
                    f"Both 'output_dim' and 'Wout' are set but their dimensions doesn't "
                    f"match: {output_dim} != {bias.shape[0]}."
                )
            self.output_dim = bias.shape[0]

    def _set_output_dim_labels(self, y: Optional[Union[NodeInput, Timestep]]):
        if y is None:
            return
        if isinstance(y, Sequence):
            if len(y) == 0:
                return
            # output_dim = y[0].shape[-1]
            if y[0].shape[-1] != 1:
                raise ValueError(
                    f"The labels should be a 1D array of shape (n_samples, 1) but got {y[0].shape}."
                )
            # count number of unique labels
            output_dim = len(set(np.concatenate(y)))

        else:
            if y.shape[-1] != 1:
                raise ValueError(
                    f"The labels should be a 1D array of shape (n_samples, 1) but got {y.shape}."
                )
            # count number of unique labels for array
            # np.unique(y)
            output_dim = np.unique(y).shape[0]

        if self.output_dim is not None and self.output_dim != output_dim:
            raise ValueError(
                f"Trying to set {self} input_dim to {output_dim} while it has already been set to {self.output_dim}"
            )
        self.output_dim = output_dim

    def initialize(
        self,
        x: Union[NodeInput, Timestep],
        y: Optional[Union[NodeInput, Timestep]] = None,
    ):
        self._set_input_dim(x)
        self._set_output_dim_labels(y)

        # initialize matrices
        if isinstance(self.Wout, Callable):
            self.Wout = self.Wout(self.input_dim, self.output_dim)
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)
        self.state = {"out": np.zeros((self.output_dim,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x @ self.Wout + self.bias}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        out = x @ self.Wout + self.bias
        return {"out": out[-1]}, out
    
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0):
        check_node_input(x, expected_dim=self.input_dim)
        if y is not None:
            check_node_input(y, expected_dim=1)

        if not self.initialized:
            self.initialize(x, y)

        if isinstance(x, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            x = np.concatenate(x, axis=0)
        if is_array(x) and x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        if isinstance(y, Sequence):
            # Concatenate all the batches as one np.ndarray
            # of shape (timeseries*timesteps, features)
            y = np.concatenate(y, axis=0)
        if is_array(y) and y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        # implement the Newton method and Backtracking line search
        if self.fit_bias:
            # add a column of ones before the X
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        if self.fit_bias:
            Wout_temp = np.zeros((self.input_dim+1, self.output_dim-1))
        else:
            Wout_temp = np.zeros((self.input_dim, self.output_dim-1))
        Wout_temp_flat = Wout_temp.ravel(order='F').reshape(-1, 1)


        _lambda_vec = self.reg_param * np.ones(self.input_dim)
        if self.fit_bias:
            _lambda_vec = np.hstack((0.0, _lambda_vec))
        repeat_lambda_vec = np.repeat(_lambda_vec.reshape(-1, 1), self.output_dim-1, axis=1)
        _lambda_vec_flat = repeat_lambda_vec.ravel(order='F').reshape(-1, 1)

        options = {}
        options["verbose"] = 1
        options["mode"] = 0
        options["progTol"] = 1e-12

        Wout_opt_flat, _ = L1GeneralUnconstrainedApx(
            partial(softmaxLoss2_noCupy, X=x, y=y.squeeze(), k=self.output_dim),   # Objective function
            Wout_temp_flat,                                   # Initial guess of Wout in one dimension
            _lambda_vec_flat,                                    # regularization parameter in vector
            options,
        )

        if self.fit_bias:
            Wout_temp = Wout_opt_flat.reshape((self.input_dim+1, self.output_dim-1))
            Wout_temp = np.concatenate((Wout_temp, np.zeros((self.input_dim+1, 1))), axis=1)
        else:
            Wout_temp = Wout_opt_flat.reshape((self.input_dim, self.output_dim-1))
            Wout_temp = np.concatenate((Wout_temp, np.zeros((self.input_dim, 1))), axis=1)

        
        if self.fit_bias:
            self.Wout = Wout_temp[1:, :]
            self.bias = Wout_temp[0, :]
        else:
            self.Wout = Wout_temp






if __name__ == '__main__':
    # from reservoirpy.nodes import Ridge
    x = np.random.normal(size=(100, 3))
    noise = np.random.normal(scale=0.1, size=(100, 1))
    y1 = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
    y2 = x @ np.array([[1.5], [0.3], [2.]]) + noise + 10.
    y = np.hstack((y1, y2))

    smoothl1_regressor = SmoothL1_tsr_Newton_Node(reg_param=1e-4)

    smoothl1_regressor.fit(x, y)
    print(smoothl1_regressor.Wout, smoothl1_regressor.bias)
