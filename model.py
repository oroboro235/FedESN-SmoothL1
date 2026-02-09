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

from funcs import mse_sl1_fval, mse_sl1_grad, ce_sl1_fval, ce_sl1_grad

# ===========================================================================
#                        Reservoir node
# ===========================================================================


class SmoothL1_Regressor_Node(TrainableNode):
    """
        SmoothL1_Node class
        A class based on TrainableNode for calculate the readout weights using SmoothL1 regularization.
        Task: Regression
        ObjFunc: MSE
    """
    
    # Regularization Params
    reg_param: float
    # bias term flag
    fit_bias: bool
    # Learned output weights
    Wout: Weights
    # Learned bias
    bias: Weights
    # Learning rate
    learning_rate: float
    # number of epochs
    epochs: int
    # batch size, 
    # batch_size: Optional[int]
    # smoothl1's alpha
    alpha: float
    # threshold for sparsity
    thres: float

    def __init__(
        self,
        reg_param: float = 0.0,
        alpha: float = 1.0,
        thres: float = 1e-5,
        learning_rate: float = 0.01,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        epochs: int = 100,
        # batch_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
    ):
        # definition
        self.reg_param = reg_param
        self.alpha = alpha
        self.thres = thres
        self.learning_rate = learning_rate
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.epochs = epochs
        self.name = name
        self.state = {}

        # input_dim/output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Check if the Wout is legal
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
        
        # Check if the bias is legal
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

        # if batch_size is None, set it to the number of samples
        # if self.batch_size is None:
        #     self.batch_size = x.shape[0]
        

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
    
    def update_alpha(self, e: int):
        update1 = 1.5
        update2 = 1.25
        max_alpha = 1e6

        if e == 0:
            _alpha = self.alpha * update1
        elif e > 0:
            _alpha = self.alpha * update2
        else:
            _alpha = self.alpha

        if _alpha > max_alpha:
            _alpha = max_alpha

        self.alpha = _alpha

    # calculation part
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> np.ndarray:
        # 
        check_node_input(x, expected_dim=self.input_dim)
        
        if not self.initialized:
            self.initialize(x, y)

        if x.ndim == 3 and y.ndim == 3:
            x = x[:, -1, :]
            y = y[:, -1, :]

        # train the weights in multiple epochs
        import torch
        import torch.nn
        from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop

        _Wout = torch.tensor(self.Wout, requires_grad=True)
        optimizer = SGD([_Wout], lr=self.learning_rate)

        for e in range(self.epochs):
            # train the full batch to get the result
            loss = mse_sl1_fval(_Wout.detach().numpy(), x, y, self.alpha, self.reg_param)
            w_grad = mse_sl1_grad(_Wout.detach().numpy(), x, y, self.alpha, self.reg_param)
            self.update_alpha(e)
            optimizer.zero_grad()

            # calculate the gradient and update the weights
            _Wout.grad = torch.tensor(w_grad)
            optimizer.step()
        
            self.Wout = _Wout.detach().numpy()
            # shrink
            self.Wout[np.abs(self.Wout) < self.thres] = 0.0

            # if e % 100 == 0:
            #     print(f"Epoch {e}: loss={loss:.4f}, nonzero={np.count_nonzero(_Wout.detach().numpy())}")

        return self.Wout

            
                
class SmoothL1_Classifier_Node(TrainableNode):
    """
    Docstring for SmoothL1_Classifier_Node
    """
    # Regularization Params
    reg_param: float
    # bias term flag
    fit_bias: bool
    # Learned output weights
    Wout: Weights
    # Learned bias
    bias: Weights
    # Learning rate
    learning_rate: float
    # number of epochs
    epochs: int
    # batch size, 
    # batch_size: Optional[int]
    # smoothl1's alpha
    alpha: float
    # threshold for sparsity
    thres: float

    def __init__(
        self,
        reg_param: float = 0.0,
        alpha: float = 1.0,
        thres: float = 1e-5,
        learning_rate: float = 0.01,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        epochs: int = 100,
        # batch_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        # definition
        self.reg_param = reg_param
        self.alpha = alpha
        self.thres = thres
        self.learning_rate = learning_rate
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.epochs = epochs
        self.name = name
        self.state = {}

        self.verbose = verbose

        # input_dim/output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Check if the Wout is legal
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
        
        # Check if the bias is legal
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

        # if batch_size is None, set it to the number of samples
        # if self.batch_size is None:
        #     self.batch_size = x.shape[0]
        
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
    
    def update_alpha(self, e: int):
        update1 = 1.5
        update2 = 1.25
        max_alpha = 5e6

        if e == 0:
            _alpha = self.alpha * update1
        elif e > 0:
            _alpha = self.alpha * update2
        else:
            _alpha = self.alpha

        if _alpha > max_alpha:
            _alpha = max_alpha

        self.alpha = _alpha

    # calculation part
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> np.ndarray:
        
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x, y)

        # only use the last output of x
        last_x = np.array([s[-1] for s in x])
        y = np.array(y).squeeze()

        # train the weights in multiple epochs
        import torch
        import torch.nn
        from torch.optim import Adam, SGD, Adagrad, Adadelta, RMSprop, LBFGS

        _Wout = torch.tensor(self.Wout, requires_grad=True)
        optimizer = SGD([_Wout], lr=self.learning_rate)
        # optimizer = RMSprop([_Wout], lr=self.learning_rate)
        # optimizer = LBFGS([_Wout], lr=self.learning_rate)

        for e in range(self.epochs):
            # train the full batch to get the result
        
            optimizer.zero_grad()

            loss = ce_sl1_fval(_Wout.detach().numpy(), last_x, y, self.alpha, self.reg_param)
            # calculate the gradient and update the weights
            w_grad = ce_sl1_grad(_Wout.detach().numpy(), last_x, y, self.alpha, self.reg_param)
            self.update_alpha(e)

            _Wout.grad = torch.tensor(w_grad)

            optimizer.step()
        
            self.Wout = _Wout.detach().numpy()
            # shrink
            self.Wout[np.abs(self.Wout) < self.thres] = 0.0

            if e % 100 == 0 and self.verbose:
                print(f"Epoch {e}: loss={loss:.4f}, nonzero={np.count_nonzero(_Wout.detach().numpy())}")

        return self.Wout

                
                
class SmoothL1_Classifier_Node_Newton(TrainableNode):
    """
    Docstring for SmoothL1_Classifier_Node
    """
    # Regularization Params
    reg_param: float
    # bias term flag
    fit_bias: bool
    # Learned output weights
    Wout: Weights
    # Learned bias
    bias: Weights
    # Learning rate
    learning_rate: float
    # number of epochs
    epochs: int
    # batch size, 
    # batch_size: Optional[int]
    # smoothl1's alpha
    alpha: float
    # threshold for sparsity
    thres: float

    def __init__(
        self,
        reg_param: float = 0.0,
        alpha: float = 1.0,
        thres: float = 1e-5,
        learning_rate: float = 0.01,
        fit_bias: bool = True,
        Wout: Optional[Union[Weights, Callable]] = None,
        bias: Optional[Union[Weights, Callable]] = None,
        epochs: int = 100,
        # batch_size: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        name: Optional[str] = None,
        verbose: bool = False,
    ):
        # definition
        self.reg_param = reg_param
        self.alpha = alpha
        self.thres = thres
        self.learning_rate = learning_rate
        self.fit_bias = fit_bias
        self.Wout = Wout
        self.bias = bias
        self.epochs = epochs
        self.name = name
        self.state = {}

        self.verbose = verbose

        # input_dim/output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Check if the Wout is legal
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
        
        # Check if the bias is legal
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

        # if batch_size is None, set it to the number of samples
        # if self.batch_size is None:
        #     self.batch_size = x.shape[0]
        
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
    
    def update_alpha(self, e: int):
        update1 = 1.5
        update2 = 1.25
        max_alpha = 1e6

        if e == 0:
            _alpha = self.alpha * update1
        elif e > 0:
            _alpha = self.alpha * update2
        else:
            _alpha = self.alpha

        if _alpha > max_alpha:
            _alpha = max_alpha

        self.alpha = _alpha

    # calculation part
    def fit(self, x: NodeInput, y: Optional[NodeInput] = None, warmup: int = 0) -> np.ndarray:
        
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x, y)

        # only use the last output of x
        last_x = np.array([s[-1] for s in x])
        y = np.array(y).squeeze()
        y = np.argmax(y, axis=1)

        # train the weights in multiple epochs
        from functools import partial
        from L1General_python.lossFuncs import SquaredError, SquaredError_noCupy, softmaxLoss2, softmaxLoss2_noCupy
        from L1General_python.L1General import L1GeneralUnconstrainedApx

        Wout_flat = self.Wout[:, :-1].ravel(order='F').reshape(-1, 1)

        lambda_vec = self.reg_param * np.ones(Wout_flat.shape)
        lambda_vec_flat = lambda_vec.ravel(order='F').reshape(-1, 1)

        options = {}
        options["mode"] = 0
        options["progTol"] = 1e-12
        options["verbose"] = 0

        Wout_opt_flat, _ = L1GeneralUnconstrainedApx(
            partial(softmaxLoss2_noCupy, X=last_x, y=y.squeeze(), k=self.output_dim),   # Objective function
            Wout_flat,                                   # Initial guess of Wout in one dimension
            lambda_vec_flat,                                    # regularization parameter in vector
            options,
        )

        Wout = Wout_opt_flat.reshape((self.Wout.shape[0], self.Wout.shape[1]-1))
        Wout = np.concatenate((Wout, np.zeros((self.Wout.shape[0], 1))), axis=1)
        self.Wout = Wout
        return self.Wout
            





if __name__ == '__main__':
    # reservoir test
    from reservoirpy.nodes import Reservoir, Ridge, Input
    from reservoirpy.datasets import mackey_glass, to_forecasting, japanese_vowels

    # X = mackey_glass(5000)
    # X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    # x, y = to_forecasting(X, forecast=10)
    # X_train1, y_train1 = x[2000:], y[2000:]
    # X_test1, y_test1 = x[:2000], y[:2000] 

    # # source = Input()
    # reservoir = Reservoir(500, sr=0.9, lr=0.1)
    

    # W_init = np.random.normal(size=(500, 1))
    # b_init = np.zeros(1)


    # readout = SmoothL1_Regressor_Node(
    #     reg_param=1e-2,
    #     thres=1e-5,
    #     learning_rate=1e-3,
    #     Wout=W_init,
    #     fit_bias=False,
    #     epochs=5000,
    #     input_dim=500,
    #     output_dim=1,
    #     name="test_readout"
    # )



    # # model = source >> reservoir >> readout
    # model = reservoir >> readout

    # model.fit(X_train1, y_train1)

    # # print(model.nodes[1].Wout)

    # def plot_readout(readout):
    #     import matplotlib.pyplot as plt
    #     Wout = readout.Wout
    #     # bias = readout.bias
    #     # Wout = np.r_[bias[..., np.newaxis], Wout]

    #     # calculate the sparsity of the readout
    #     sparsity = (Wout == 0).mean() * 100
    #     print(f"Sparsity of the readout: {sparsity:.2f}%")

    #     fig = plt.figure(figsize=(15, 5))

    #     ax = fig.add_subplot(111)
    #     ax.grid(axis="y")
    #     ax.set_ylabel("Coefs. of $W_{out}$")
    #     ax.set_xlabel("reservoir neurons index")
    #     ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

    #     plt.show()
    #     plt.savefig("readout_coefs.png")


    # plot_readout(model.nodes[1])

    X_train, X_test, Y_train, Y_test = japanese_vowels()

    reservoir = Reservoir(500, sr=0.9, lr=0.1)

    W_init = np.random.normal(size=(500, 9))

    b_init = np.zeros(9)

    readout = SmoothL1_Classifier_Node(
        reg_param=1e-2,
        thres=1e-5,
        learning_rate=1e-2,
        Wout=W_init,
        bias=b_init,
        fit_bias=False,
        epochs=5000,
        input_dim=500,
        output_dim=9,
        name="test_readout"
    )

    model = reservoir >> readout

    weights = model.fit(X_train, Y_train)

    # print(model.nodes[1].Wout)
    full_s = model.run(X_test)
    digits_test = np.array([s[-1] for s in full_s])
    # y_pred = np.argmax(digits_test, axis=1)
    
    y_pred = np.argmax(digits_test, axis=1)
    
    # Y_test is one-hot, convert it to digit

    Y_test = np.argmax(np.array(Y_test).squeeze(), axis=1)

    
    acc = (y_pred == Y_test).mean()
    print(f"Accuracy: {acc:.2f}")

    def plot_readout(readout):
        import matplotlib.pyplot as plt
        Wout = readout.Wout
        # bias = readout.bias
        # Wout = np.r_[bias[..., np.newaxis], Wout]

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



    






