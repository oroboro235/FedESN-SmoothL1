import numpy as np
from typing import List, Tuple, Dict, Any
from reservoirpy.nodes import Reservoir
from readout_node import Clr_Node


from scipy.sparse import sparray
from scipy.sparse import csr_matrix


class Client_TSC:
    """Client class"""
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], seed: int=None):
        self.client_id = client_id
        self.X_train, self.y_train = data[0], data[1]
        self.X_test, self.y_test = data[2], data[3]
        self.model = None
        # self.global_Wout = None
        self.hyperparams = None # dict
        self.seed = seed
        self.sparsity = 0.0
    

            
    def initialize_model(self, num_classes: int, Win: sparray = None, Wres: sparray = None):
        """Initialize model"""

        if Wres is None or Win is None:
            raise ValueError("Win and Wres must be provided")

        input_dim = self.X_train.shape[1] # data's dimension
        output_dim = num_classes # number of classes

        # initialize reservoir
        res = Reservoir(
            units=self.hyperparams["units"],
            lr=self.hyperparams["lr"],
            sr=self.hyperparams["sr"],
            W=Wres,
            Win=Win,
            input_scaling=self.hyperparams["input_scaling"],
            input_connectivity=self.hyperparams["input_connectivity"],
            rc_connectivity=self.hyperparams["rc_connectivity"],
            seed=self.seed,
        )

        # initialize readout
        # W_init = np.random.normal(size=(self.hyperparams["units"], output_dim))
        W_init = np.zeros((self.hyperparams["units"], output_dim))
        b_init = np.zeros(output_dim)

        readout = Clr_Node(
            reg_param=self.hyperparams["reg_param"],
            reg_type=self.hyperparams["reg_type"],
            thres=self.hyperparams["thres"],
            learning_rate=self.hyperparams["learning_rate"],
            Wout=W_init,
            bias=b_init,
            fit_bias=False,
            epochs=self.hyperparams["local_epochs"],
            input_dim=self.hyperparams["units"],
            output_dim=output_dim,
        )

        # self.global_Wout = np.zeros_like(W_init)

        self.model = res >> readout

    def receive_hyperparams(self, hyperparams: Dict[str, Any]):
        """Receive hyperparameters from server"""
        if hyperparams is None:
            raise ValueError("Hyperparameters not provided")
        self.hyperparams = hyperparams
        self.local_lr = hyperparams["local_lr"]


    def receive_parameters(self, params: sparray):
        """Receive updated parameters from server, only update readout weights"""
        if params is None:
            raise ValueError("Parameters not provided")
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.nodes[1].Wout = (1 - self.local_lr) * self.model.nodes[1].Wout + self.local_lr * params.toarray()

        
    def train(self) -> Dict[str, np.ndarray]:
        """Local training"""
        if self.model is None or self.hyperparams is None:
            raise ValueError("Model or hyperparameters not initialized")
            
        self.model.fit(self.X_train, self.y_train)

        # turn Wout into sparse format
        Wout = self.model.nodes[1].Wout
        Wout_sparse = csr_matrix(Wout)

        return Wout_sparse
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        full_digits = self.model.run(self.X_test)
        digits = np.array([s[-1] for s in full_digits])
        y_pred = np.argmax(digits, axis=1)

        y_true = np.argmax(np.array(self.y_test).squeeze(), axis=1)

        acc = (y_pred == y_true).mean()*100

        # calculate sparsity of readout
        self.sparsity = (self.model.nodes[1].Wout == 0).mean() * 100
        
        return {
            'acc': acc,
            'sparsity': self.sparsity,
            'client_id': self.client_id,
        }
    
