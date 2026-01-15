import funcs

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

from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop

from funcs import SmoothL1_reg

# ===========================================================================
#                        Reservoir node
# ===========================================================================


class SmoothL1_tsr_Newton_Node(TrainableNode):
    '''
    SmoothL1 Regularization
    Objective: Sum of Squared Error Loss (regression)
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
    Objective: Softmax Loss (classification)
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

# use gradient method to solve smoothl1



class SmoothL1_tsr_optimizer_Node(TrainableNode):
    '''
    SmoothL1 Regularization
    Objective: Sum of Squared Error Loss (regression)
    Optimizer: Gradient-based optimizer (Adam, SGD, etc.)
    '''
    reg_param: float
    fit_bias: bool
    Wout: Weights
    bias: Weights
    optimizer_name: str
    learning_rate: float
    epochs: int
    batch_size: Optional[int]

    def __init__(
        self,
        reg_param: float = 0.0,
        fit_bias: bool = True,
        optimizer_name: str = "adam",
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.reg_param = reg_param
        self.fit_bias = fit_bias
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
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
        elif not is_array(self.Wout):
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            self.Wout = np.random.uniform(-limit, limit, (self.input_dim, self.output_dim))
        
        if isinstance(self.bias, Callable):
            self.bias = self.bias(self.output_dim)
        elif not is_array(self.bias):
            self.bias = np.zeros((self.output_dim,))
            
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
            # 对于时间序列数据，我们不建议打乱顺序
            # 但这里我们仍然将多个批次连接起来，保持每个批次内部的时间顺序
            x = np.concatenate(x, axis=0)
        if is_array(x) and x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        if isinstance(y, Sequence):
            y = np.concatenate(y, axis=0)
        if is_array(y) and y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])

        # Convert to PyTorch tensors
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_tensor = torch.FloatTensor(x).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # Prepare parameters
        if self.fit_bias:
            # Add bias to weights as an extra column
            W_tensor = torch.FloatTensor(self.Wout).to(device)
            b_tensor = torch.FloatTensor(self.bias).to(device)
            
            # Combine weights and bias into a single parameter
            W_full = torch.cat([b_tensor.unsqueeze(0), W_tensor], dim=0)
            W_full.requires_grad = True
        else:
            W_full = torch.FloatTensor(self.Wout).to(device)
            W_full.requires_grad = True

        # Choose optimizer
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam([W_full], lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD([W_full], lr=self.learning_rate)
        elif self.optimizer_name == "adagrad":
            optimizer = torch.optim.Adagrad([W_full], lr=self.learning_rate)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta([W_full], lr=self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop([W_full], lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Prepare data for mini-batch training
        n_samples = X_tensor.shape[0]
        if self.batch_size is None or self.batch_size >= n_samples:
            batch_size = n_samples
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)

        # Training loop - 对于时间序列，我们不应该打乱数据
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            reg_SL1 = SmoothL1_reg(reg_param=self.reg_param, if_hess=False)

            # 对于时间序列数据，我们按时间顺序处理批次，不打乱
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                # 按时间顺序获取批次
                X_batch = X_tensor[start_idx:end_idx]
                y_batch = y_tensor[start_idx:end_idx]
                
                # Forward pass
                if self.fit_bias:
                    bias_batch = W_full[0:1, :]
                    W_batch = W_full[1:, :]
                    y_pred = torch.mm(X_batch, W_batch) + bias_batch
                else:
                    y_pred = torch.mm(X_batch, W_full)
                
                # Compute loss: MSE + Smooth L1 regularization
                mse_loss = torch.mean((y_pred - y_batch) ** 2)
                
                # Smooth L1 regularization on weights (exclude bias if fit_bias=True)
                if self.fit_bias:
                    weights = W_full[1:, :]
                else:
                    weights = W_full
                
                # Smooth L1 val to loss
                reg_loss = reg_SL1.cal_func_val(weights.detach().numpy())
                reg_SL1.update_alpha()
                # reg_loss = self.reg_param * torch.sum(smooth_l1)
                
                total_loss = mse_loss + reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/n_batches:.6f}")

        # Convert back to numpy
        if self.fit_bias:
            W_np = W_full[1:, :].detach().cpu().numpy()
            b_np = W_full[0, :].detach().cpu().numpy()
            self.Wout = W_np
            self.bias = b_np
        else:
            self.Wout = W_full.detach().cpu().numpy()

import numpy as np
from typing import Tuple, List

def create_simple_test_data(
    n_samples: int = 50,
    n_features: int = 100,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Create simple linearly separable test data.
    """
    np.random.seed(random_seed)

    # generate two linearly separable classes
    X_class1 = np.random.randn(n_samples//2, n_features) + 1.0
    X_class2 = np.random.randn(n_samples//2, n_features) - 1.0
    
    X = np.vstack([X_class1, X_class2])
    y = np.vstack([
        np.zeros((n_samples//2, 1)),
        np.ones((n_samples//2, 1))
    ]).astype(int)

    # shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X.astype(np.float32), y

def create_sequence_test_data(
    n_sequences: int = 3,
    seq_length: int = 10,
    n_features: int = 5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create sequence test data.
    """
    X_sequences = []
    y_sequences = []
    
    for i in range(n_sequences):
        # generate a sequence
        sequence = np.random.randn(seq_length, n_features)
        # generate binary labels
        labels = np.random.randint(0, 2, size=(seq_length, 1))
        
        X_sequences.append(sequence.astype(np.float32))
        y_sequences.append(labels.astype(int))
    
    return X_sequences, y_sequences















if __name__ == '__main__':
    # # from reservoirpy.nodes import Ridge
    # x = np.random.normal(size=(100000, 3))
    # noise = np.random.normal(scale=0.1, size=(100000, 1))
    # y1 = x @ np.array([[10], [-0.2], [7.]]) + noise + 12.
    # y2 = x @ np.array([[1.5], [0.3], [2.]]) + noise + 10.
    # y = np.hstack((y1, y2))

    # import time

    # smoothl1_regressor = SmoothL1_tsr_Newton_Node(reg_param=1e-4)
    # # smoothl1_regressor = SmoothL1_tsr_optimizer_Node(epochs=5000, batch_size=1000,optimizer_name="sgd", reg_param=1e-4)

    # start_time = time.time()
    # smoothl1_regressor.fit(x, y)
    # end_time = time.time()
    # print("Time elapsed: ", end_time - start_time)
    # print(smoothl1_regressor.Wout, smoothl1_regressor.bias)

    # # test standard data
    # X, y = create_simple_test_data()
    # print(f"Standard data shape: X={X.shape}, y={y.shape}")
    
    # # # test sequence data
    # # X_seq, y_seq = create_sequence_test_data()
    # # print(f"Sequence data: {len(X_seq)} sequences, each with shape {X_seq[0].shape}")

    # # split the data to train and test
    # # from sklearn.model_selection import train_test_split
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # test smoothl1_classifier
    # sl1_classifier = SmoothL1_tsc_Newton_Node(reg_param=1e-4)

    # sl1_classifier.fit(X, y)

    # print(sl1_classifier.Wout, sl1_classifier.bias)

    # reservoir test
    from reservoirpy.nodes import Reservoir, Ridge, Input
    from reservoirpy.datasets import mackey_glass, to_forecasting

    X = mackey_glass(5000)
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    x, y = to_forecasting(X, forecast=10)
    X_train1, y_train1 = x[2000:], y[2000:]
    X_test1, y_test1 = x[:2000], y[:2000] 

    # source = Input()
    reservoir = Reservoir(500, sr=0.9, lr=0.1)
    # readout = SmoothL1_tsr_optimizer_Node(epochs=5000, 
    #                                       batch_size=500,
    #                                       optimizer_name="sgd", 
    #                                       reg_param=1e-1)
    readout = SmoothL1_tsr_Newton_Node(reg_param=1e-4)

    # model = source >> reservoir >> readout
    model = reservoir >> readout

    model.fit(X_train1, y_train1)

    # print(model.nodes[1].Wout)

    def plot_readout(readout):
        import matplotlib.pyplot as plt
        Wout = readout.Wout
        bias = readout.bias
        Wout = np.r_[bias[..., np.newaxis], Wout]

        # calculate the sparsity of the readout
        sparsity = (Wout == 0).mean() * 100
        print(f"Sparsity of the readout: {sparsity:.2f}%")

        fig = plt.figure(figsize=(15, 5))

        ax = fig.add_subplot(111)
        ax.grid(axis="y")
        ax.set_ylabel("Coefs. of $W_{out}$")
        ax.set_xlabel("reservoir neurons index")
        ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

        plt.show()
        plt.savefig("readout_coefs.png")
    
    plot_readout(model.nodes[1])



    






