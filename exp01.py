# this exp is for 
# testing the SGD on ESN states

import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass, to_forecasting

np.random.seed(1234)

import reservoirpy as rpy


UNITS = 500               # - number of neurons
LEAK_RATE = 0.3           # - leaking rate
SPECTRAL_RADIUS = 0.9    # - spectral radius of W
INPUT_SCALING = 1.0       # - input scaling
RC_CONNECTIVITY = 0.1     # - density of reservoir internal matrix
INPUT_CONNECTIVITY = 0.2  # and of reservoir input matrix
SEED = 1234               # for reproductibility

if __name__ == "__main__":

    len_forecast = 1
    len_window = 500 + len_forecast
    step = 5
    n_example = 1000
    len_ts = len_window + step * n_example

    train_test_ratio = 0.8

    # create batches
    raw = mackey_glass(len_ts)
    raw = 2 * (raw - raw.min()) / (raw.max() - raw.min()) - 1

    batches = []
    for i in range(n_example):
        start = i * step
        end = start + len_window
        batches.append(raw[start:end])
    
    batches = np.array(batches)

    print(batches.shape)

    train_batches = batches[:int(train_test_ratio * n_example)]
    test_batches = batches[int(train_test_ratio * n_example):]

    train_batches_X = train_batches[:, :-len_forecast, :]
    test_batches_X = test_batches[:, :-len_forecast, :]
    train_batches_Y = train_batches[:, len_forecast:, :]
    test_batches_Y = test_batches[:, len_forecast:, :]

    # initialized the reservoir
    res = Reservoir(
        units=UNITS,
        sr=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        lr=LEAK_RATE,
        rc_connectivity=RC_CONNECTIVITY,
        input_connectivity=INPUT_CONNECTIVITY,
        seed=SEED,
    )

    # get states for input X_train and Y_train
    s_X_tr = []
    s_X_te = []
    for i in range(len(train_batches_X)):
        s_X_tr.append(res.run(train_batches_X[i]))
        res.reset()
    
    for i in range(len(test_batches_X)):
        s_X_te.append(res.run(test_batches_X[i]))
        res.reset()
    s_X_tr = np.array(s_X_tr)
    s_X_te = np.array(s_X_te)

    print(s_X_tr.shape, s_X_te.shape)
    print(train_batches_Y.shape, test_batches_Y.shape)


    # ================================================================================================
    def mse_fval(w, X, y):
        n_sample = X.shape[0]
        return (1.0 / n_sample) * np.sum((np.dot(X, w) - y)**2)
    
    def mse_grad(w, X, y):
        n_sample = X.shape[0]
        return (2.0 / n_sample) * np.dot(X.T, np.dot(X, w) - y)

    from scipy.special import logsumexp

    def sl1_fval(w, alpha, _lambda):
        n_feature = w.shape[0]
        
        lse = logsumexp(np.hstack([np.zeros((n_feature, 1)), alpha*w]), axis=1)
        neg_lse = logsumexp(np.hstack([np.zeros((n_feature, 1)), -alpha*w]), axis=1)

        # without bias 
        lambda_vec = (_lambda * np.ones(n_feature)).squeeze()
        fval = np.sum((lambda_vec * (1.0 / alpha)) * (lse + neg_lse))
        return fval
    
    def sl1_grad(w, alpha, _lambda):
        n_feature = w.shape[0]

        lse = logsumexp(np.hstack([np.zeros((n_feature, 1)), alpha*w]), axis=1)

        lambda_vec = (_lambda * np.ones(n_feature)).squeeze()

        grad = (lambda_vec * (1.0 - 2.0 * np.exp(-lse))).reshape(-1, 1)

        return grad
    
    def update_alpha(alpha, cnt=0):
        update1 = 1.5
        update2 = 1.25
        max_alpha = 5e5

        if cnt == 0:
            new_alpha = alpha * update1
        elif cnt > 0:
            new_alpha = alpha * update2
        else:
            new_alpha = alpha

        if new_alpha > max_alpha:
            new_alpha = max_alpha

        return new_alpha
    
    def l2_fval(w, _lambda):
        return 0.5 * _lambda * np.sum(w**2)
    
    def l2_grad(w, _lambda):
        return _lambda * w
    

    # ================================================================================================

    def sgd(X, y, learning_rate=0.1, epochs=1000, batch_size=100):
        n_sample = X.shape[0]

        w = np.random.rand(UNITS, 1)
        # w = np.zeros((UNITS, 1))

        # for loss function
        alpha = 1.0
        thres = 1e-5
        _lambda = 1e-2

        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            grad = 0.0
            for j in range(batch_size):
                grad += mse_grad(w, X_batch[j], y_batch[j])
            grad /= batch_size
            grad += sl1_grad(w, alpha, _lambda)
            # grad += l2_grad(w, _lambda)
            w -= learning_rate * grad

            # shrink to zero
            w[np.abs(w) < thres] = 0.0

            loss = mse_fval(w, X, y) + sl1_fval(w, alpha, _lambda)
            # loss = mse_fval(w, X, y) + l2_fval(w, _lambda)
            loss_history.append(loss)

            alpha = update_alpha(alpha, epoch)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss}, nonzeros={np.count_nonzero(w)}")

        return w, loss_history

    w, loss_history = sgd(s_X_tr, train_batches_Y, learning_rate=1e-2, epochs=5000, batch_size=200)

    

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig("loss_history.png")

            
    # ================================================================================================

    
    

    

