import numpy as np
from typing import List, Tuple, Dict, Any
import copy

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.node import TrainableNode
from model import SmoothL1_Classifier_Node

global rng_seed
rng_seed = 1234
np.random.seed(rng_seed)

class Client:
    """Client class"""
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], seed: int=None):
        self.client_id = client_id
        self.X_train, self.y_train = data[0], data[1]
        self.X_test, self.y_test = data[2], data[3]
        self.model = None
        self.sparsity = 0.0
        self.hyperparams = None # dict
        self.seed = seed
        
    def receive_hyperparams(self, hyperparams: Dict[str, Any]):
        """Receive hyperparameters from server"""
        self.hyperparams = hyperparams
        
    def initialize_model(self, input_dim: int, output_dim: int, Win: np.ndarray = None, Wres: np.ndarray = None, bias: float = None):
        """Initialize model"""
        if Win is None:
            from reservoirpy.mat_gen import bernoulli
            Win = bernoulli
        
        if Wres is None:
            from reservoirpy.mat_gen import normal
            Wres = normal

        if bias is None:
            bias = 0.0

        res = Reservoir(
            units=self.hyperparams["units"],
            lr=self.hyperparams["lr"],
            sr=self.hyperparams["sr"],
            W=Wres,
            Win=Win,
            bias=bias,
            input_scaling=self.hyperparams["input_scaling"],
            input_connectivity=self.hyperparams["input_connectivity"],
            rc_connectivity=self.hyperparams["rc_connectivity"],
            seed=self.seed,
        )

        W_init = np.random.normal(size=(input_dim, output_dim))
        b_init = np.zeros(output_dim)

        readout = SmoothL1_Classifier_Node(
            reg_param=self.hyperparams["reg_param"],
            thres=self.hyperparams["thres"],
            learning_rate=self.hyperparams["learning_rate"],
            Wout=W_init,
            bias=b_init,
            fit_bias=False,
            epochs=self.hyperparams["local_epochs"],
            input_dim=input_dim,
            output_dim=output_dim,
            name="readout",
        )
        self.model = res >> readout

        
    def train(self) -> Dict[str, np.ndarray]:
        """Local training"""
        if self.model is None or self.hyperparams is None:
            raise ValueError("Model or hyperparameters not initialized")
            
        self.model.fit(self.X_train, self.y_train)

        # calculate sparsity of readout
        self.sparsity = (self.model.nodes[1].Wout == 0).mean() * 100
        
        return self.model.nodes[1].Wout
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        full_digits = self.model.run(self.X_test)
        digits = np.array([s[-1] for s in full_digits])
        y_pred = np.argmax(digits, axis=1)

        y_true = np.argmax(np.array(self.y_test).squeeze(), axis=1)

        acc = (y_pred == y_true).mean()*100
        
        return {
            'acc': acc,
            'sparsity': self.sparsity,
            'client_id': self.client_id,
        }
    
    def receive_parameters(self, params: Dict[str, np.ndarray]):
        """Receive updated parameters from server"""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.nodes[1].Wout = params

class Server:
    """Server class"""
    def __init__(self, hyperparams: dict = None, n_rounds: int = 10):
        self.hyperparams = hyperparams
        self.global_model = None
        self.n_rounds = n_rounds
        
    def initialize_global_model(self, input_dim: int, output_dim: int):
        """Initialize global model"""

        res = Reservoir(
            units=self.hyperparams["units"],
            lr=self.hyperparams["lr"],
            sr=self.hyperparams["sr"],
            input_scaling=self.hyperparams["input_scaling"],
            input_connectivity=self.hyperparams["input_connectivity"],
            rc_connectivity=self.hyperparams["rc_connectivity"],
            seed=rng_seed,
        )
        res.initialize(np.zeros((1, input_dim)))
        b_init = np.zeros(output_dim)
        readout = SmoothL1_Classifier_Node(
            reg_param=self.hyperparams["reg_param"],
            bias = b_init,
            thres=self.hyperparams["thres"],
            learning_rate=self.hyperparams["learning_rate"],
            epochs=self.hyperparams["local_epochs"]
        )
        self.global_model = res >> readout
        
    def transmit_hyperparams(self, clients: List[Client]):
        """1. Transmit hyperparameters to all clients"""
        for client in clients:
            client.receive_hyperparams(self.hyperparams)
            
    def aggregate_parameters(self) -> Dict[str, np.ndarray]:
        """3. Aggregate trained weights from all clients and average them"""
        if not self.client_parameters:
            raise ValueError("No client parameters to aggregate")
            
        # Initialize average parameters
        avg_weights = np.zeros_like(self.client_parameters[0])
        # avg_bias = np.zeros_like(self.client_parameters[0])
        
        # Average parameters from all clients
        for params in self.client_parameters:
            avg_weights += params
            # avg_bias += params['bias']
            
        avg_weights /= len(self.client_parameters)
        # avg_bias /= len(self.client_parameters)
        

        return avg_weights
        # return {
        #     'weights': avg_weights,
        #     'bias': avg_bias
        # }
    
    def update_global_model(self, avg_params: Dict[str, np.ndarray]):
        """4. Update readout weights with the averaged weight"""
        self.global_model.nodes[1].Wout = avg_params
        
    def transmit_parameters(self, clients: List[Client]):
        """5. Transmit updated weights to all clients"""
        global_params = self.global_model.nodes[1].Wout
        for client in clients:
            client.receive_parameters(global_params)
            
    def run_federated_learning(self, clients: List[Client], input_dim: int, output_dim: int):
        """Run federated learning process"""
        print("Starting federated learning...")
    
        
        # Initialize global model
        self.initialize_global_model(input_dim, output_dim)
        Wres = self.global_model.nodes[0].W
        Win = self.global_model.nodes[0].Win
        res_bias = self.global_model.nodes[0].bias
        
#        1. Server transmit the hyperparameters to all clients
        # print("1. Server transmitting hyperparameters to all clients...")
        # Initialize models for all clients
        self.transmit_hyperparams(clients)
        for client in clients:
            client.initialize_model(self.hyperparams["units"], output_dim, Win=Win, Wres=Wres, bias=res_bias)
            # client.initialize_model(self.hyperparams["units"], output_dim)

        for round_num in range(self.n_rounds):
            print(f"\n=== Federated Learning Round {round_num + 1} ===")
                        
            # Clear client parameters from previous round
            self.client_parameters = []
            
            # 2. Clients train the readouts with their own datasets
            # print("2. Clients training locally...")
            for client in clients:
                client_params = client.train()
                client.model.reset()
                self.client_parameters.append(client_params)
            
            # 3. Server aggregates the trained weights from all clients, then does average
            # print("3. Server aggregating and averaging weights...")
            avg_params = self.aggregate_parameters()
            
            # 4. Server updates the readout weights with the averaged weight
            # print("4. Server updating global model...")
            self.update_global_model(avg_params)
            
            # 5. Server transmits the updated weights to all clients
            # print("5. Server transmitting updated weights to all clients...")
            self.transmit_parameters(clients)
            
            # 6. Clients evaluate the performance on their own datasets
            # print("6. Clients evaluating performance...")
            round_results = []
            for client in clients:
                result = client.evaluate()
                client.model.reset()
                round_results.append(result)
                print(f"  Client {client.client_id}: acc: {result['acc']:.2f}%, sparsity: {result['sparsity']:.2f}%")
            
            # Print round results
            avg_acc = np.mean([r['acc'] for r in round_results])
            avg_spar = np.mean([r['sparsity'] for r in round_results])
            
            print(f"  Average accuracy: {avg_acc:.2f}%, Average sparsity: {avg_spar:.2f}%")
            print(f"Global Sparsity: {(self.global_model.nodes[1].Wout == 0).mean()*100:.2f}%")

        
        print("\nFederated learning completed!")
        return self.global_model

def initialize_clients(n_clients: int, no_cross: bool = True, n_sample_per_client: int = 10):
    # load the data
    from reservoirpy.datasets import japanese_vowels
    X_train, X_test, Y_train, Y_test = japanese_vowels()
    # split the data into n_clients parts
    # 
    X_train_split = []
    Y_train_split = []
    X_test_split = []
    Y_test_split = []

    # shuffle data
    # dataset is list
    dataset = list(zip(X_train, Y_train, X_test, Y_test))
    np.random.shuffle(dataset)
    X_train, Y_train, X_test, Y_test = zip(*dataset)

    if no_cross:
        # no overlay between clients, fully split
        n_sample_per_client_train = len(X_train) // n_clients
        n_sample_per_client_test = len(X_test) // n_clients
        for i in range(n_clients):
            X_train_split.append(X_train[i*n_sample_per_client_train:(i+1)*n_sample_per_client_train])
            Y_train_split.append(Y_train[i*n_sample_per_client_train:(i+1)*n_sample_per_client_train])
            X_test_split.append(X_test[i*n_sample_per_client_test:(i+1)*n_sample_per_client_test])
            Y_test_split.append(Y_test[i*n_sample_per_client_test:(i+1)*n_sample_per_client_test])
    else:
        # overlay between clients, randomly split
        for i in range(n_clients):
            idx = np.random.choice(len(X_train), n_sample_per_client, replace=False)
            X_train_split.append(X_train[idx])
            Y_train_split.append(Y_train[idx])
            idx = np.random.choice(len(X_train), n_sample_per_client//2, replace=False)
            X_test_split.append(X_test[idx])
            Y_test_split.append(Y_test[idx])

    # create clients
    clients = []
    random_seeds = np.random.randint(0, 10000, n_clients)
    for i in range(n_clients):
        client = Client(i, (X_train_split[i], Y_train_split[i], X_test_split[i], Y_test_split[i]), seed=random_seeds[i])
        clients.append(client)
    return clients, X_test, Y_test

def main():
    """Main function: Demonstrate federated learning process"""
    
    # Hyperparameters
    n_clients = 5
    # samples_per_client = 100
    input_dim = 12
    output_dim = 9
    
    # Create clients
    print(f"Creating {n_clients} clients...")
    clients, Xte, Yte = initialize_clients(n_clients)
    
    
    # Create server and run federated learning
    hyperparams = {
        "units": 100,
        "lr": 0.1,
        "sr": 1.0,
        "input_scaling": 0.1,
        "input_connectivity": 0.1,
        "rc_connectivity": 0.1,
        "reg_param": 1e-2,
        "thres": 1e-5,
        "learning_rate": 1e-2,
        "local_epochs": 5000,
    }
    n_rounds = 20
    server = Server(hyperparams=hyperparams, n_rounds=n_rounds)
    
    # Run federated learning
    global_model = server.run_federated_learning(clients, input_dim, output_dim)
    global_model.nodes[0].reset()
    # Print final results
    print("\n=== Final Results ===")
    # Evaluate global model on test set 
    y_pred = global_model.run(Xte)
    y_pred = np.array([s[-1] for s in y_pred])
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(np.array(Yte).squeeze(), axis=1)
    acc = (y_pred == y_true).mean()*100
    print(f"Global model accuracy: {acc:.2f}%")
    # sparsity
    print(f"Sparsity: {(global_model.nodes[1].Wout == 0).mean()*100:.2f}%")
    print(f"Learned weights shape: {global_model.nodes[1].Wout.shape}")

if __name__ == "__main__":
    main()