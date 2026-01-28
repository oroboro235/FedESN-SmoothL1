# FedESN-SmoothL1

FL framework


Startegy 1

Server --- Client A --- Client B --- Client C

1. Server transmit the hyperparameters to all clients. (Optimized)
2. Clients train the readouts with their own datasets.
3. Server aggregates the gradient from all clients, then do average on these gradients.
4. Server updates the readout weights with the averaged gradient.
5. Server transmits the updated weights to all clients.
6. Clients evaluate the performance on their own datasets.


Strategy 2

Server --- Client A --- Client B --- Client C
1. The client transmits the states of the reservoir to the server.
2. Server train the readout from Client's dataset.
3. Server transmits the updated weights to all clients.