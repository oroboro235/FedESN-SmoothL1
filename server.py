import numpy as np
from typing import List, Tuple, Dict, Any

from reservoirpy.nodes import Reservoir

from scipy.sparse import csr_matrix



class Server_FedAvg:
    """Server class"""
    def __init__(self, hyperparams: dict = None):
        self.hyperparams = hyperparams
        self.global_lr = hyperparams["global_lr"]
        self.res = None
        self.global_Wout = None

        
    def initialize_global_model(self):
        """Initialize global model"""

        from reservoirpy.mat_gen import uniform

        res = Reservoir(
            units=self.hyperparams["units"],
            lr=self.hyperparams["lr"],
            sr=self.hyperparams["sr"],
            Win=uniform,
            W=uniform,
            input_dim=self.hyperparams["input_dim"],
            input_scaling=self.hyperparams["input_scaling"],
            input_connectivity=self.hyperparams["input_connectivity"],
            rc_connectivity=self.hyperparams["rc_connectivity"],
            seed=self.hyperparams["seed"],
        )
        res.initialize(np.zeros((1, self.hyperparams["input_dim"])))

        self.res = res

        self.global_Wout = csr_matrix(np.zeros((self.hyperparams["units"], self.hyperparams["output_dim"])))

        
    # def transmit_hyperparams(self, clients: List[Client]):
    #     """Transmit hyperparameters to all clients"""
    #     for client in clients:
    #         client.receive_hyperparams(self.hyperparams)
            
    def aggregate_parameters(self, collected_params: list) -> Dict[str, np.ndarray]:
        """Aggregate trained weights from all clients and average them"""
        
            
        # Initialize average parameters
        avg_weights = np.zeros((self.hyperparams["units"], self.hyperparams["output_dim"]))
        
        # Average parameters from all clients
        for params in collected_params:
            # fit params into a zero matrix
            _params = np.zeros_like(avg_weights)
            _params[:params.shape[0], :params.shape[1]] = params.toarray()
            avg_weights += _params
            
        avg_weights /= len(collected_params)
        
        # shrink the avg_weights
        avg_weights[np.abs(avg_weights) < self.hyperparams["thres"]] = 0.0

        # update the global Wout

        _Wout = (1 - self.global_lr) * self.global_Wout.toarray() + self.global_lr * avg_weights

        # turn into sparse matrix
        self.global_Wout = csr_matrix(_Wout)

        return self.global_Wout

        
    # def transmit_parameters(self, clients: List[Client], add_random_noise: bool = False):
    #     """5. Transmit updated weights to all clients"""
    #     global_params = self.global_model.nodes[1].Wout
    #     if add_random_noise:
    #         global_params += np.random.normal(scale=0.1, size=global_params.shape)
        
    #     for client in clients:
    #         client.receive_parameters(global_params)
            
#     def run_federated_learning(self, clients: List[Client], input_dim: int, output_dim: int, if_plot: bool = True):
#         """Run federated learning process"""
#         print("Starting federated learning...")
    
        
#         # Initialize global model
#         self.initialize_global_model(input_dim, output_dim)
#         Wres = self.global_model.nodes[0].W
#         Win = self.global_model.nodes[0].Win
#         res_bias = self.global_model.nodes[0].bias
        
# #        1. Server transmit the hyperparameters to all clients
#         # print("1. Server transmitting hyperparameters to all clients...")
#         # Initialize models for all clients
#         self.transmit_hyperparams(clients)
#         for client in clients:
#             client.initialize_model(self.hyperparams["units"], output_dim, Win=Win, Wres=Wres, bias=res_bias)
#             # client.initialize_model(self.hyperparams["units"], output_dim)

#         for round_num in range(self.n_rounds):
#             print(f"\n=== Federated Learning Round {round_num + 1} ===")
                        
#             # Clear client parameters from previous round
#             self.client_parameters = []
            
#             # 2. Clients train the readouts with their own datasets
#             # print("2. Clients training locally...")
#             for client in clients:
#                 client_params = client.train()
#                 client.model.reset()
#                 self.client_parameters.append(client_params)

#             print("Before aggregation:")
#             round_results = []
#             for client in clients:
#                 result = client.evaluate()
#                 client.model.reset()
#                 round_results.append(result)
#                 print(f"  Client {client.client_id}: acc: {result['acc']:.2f}%, sparsity: {result['sparsity']:.2f}%")
            

#             # collect the weight of client and combine them into subplots
#             if if_plot:
#                 from matplotlib import pyplot as plt
#                 plt.figure(figsize=(5, 15))
#                 for i, client in enumerate(clients):
#                     plt.subplot(5, 1, i+1)
#                     plt.bar(np.arange(client.model.nodes[1].Wout.size), client.model.nodes[1].Wout.ravel()[::-1])
#                     plt.title(f"Client {client.client_id}")
#                 plt.tight_layout()
#                 plt.show()

#             # 3. Server aggregates the trained weights from all clients, then does average
#             # print("3. Server aggregating and averaging weights...")
#             avg_params = self.aggregate_parameters()
            
#             # 4. Server updates the readout weights with the averaged weight
#             # print("4. Server updating global model...")
#             self.update_global_model(avg_params)
            
#             # 5. Server transmits the updated weights to all clients
#             # print("5. Server transmitting updated weights to all clients...")
#             self.transmit_parameters(clients)

            
#             # 6. Clients evaluate the performance on their own datasets
#             # print("6. Clients evaluating performance...")
#             print("After aggregation:")
#             round_results = []
#             for client in clients:
#                 result = client.evaluate()
#                 client.model.reset()
#                 round_results.append(result)
#                 print(f"  Client {client.client_id}: acc: {result['acc']:.2f}%, sparsity: {result['sparsity']:.2f}%")
            
#             if if_plot:
#                 from matplotlib import pyplot as plt
#                 plt.figure(figsize=(5, 15))
#                 for i, client in enumerate(clients):
#                     plt.subplot(5, 1, i+1)
#                     plt.bar(np.arange(client.model.nodes[1].Wout.size), client.model.nodes[1].Wout.ravel()[::-1])
#                     plt.title(f"Client {client.client_id}")
#                 plt.tight_layout()
#                 plt.show()

#             # Print round results
#             avg_acc = np.mean([r['acc'] for r in round_results])
#             avg_spar = np.mean([r['sparsity'] for r in round_results])

#             if if_plot:
#                 plot_readout(self.global_model.nodes[1])
            
#             print(f"  Average accuracy: {avg_acc:.2f}%, Average sparsity: {avg_spar:.2f}%")
#             print(f"Global Sparsity: {(self.global_model.nodes[1].Wout == 0).mean()*100:.2f}%")

        
#         print("\nFederated learning completed!")
#         return self.global_model