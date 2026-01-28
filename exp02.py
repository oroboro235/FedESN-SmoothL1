# this exp is for 
# testing the SGD on classification task

import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass, to_forecasting, japanese_vowels

from sklearn.metrics import log_loss

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

    # load data with batches.

    X_train, X_test, Y_train, Y_test = japanese_vowels()

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

    for x in X_train:
        state = res.run(x)
        res.reset()
        s_X_tr.append(state[-1, np.newaxis])
    
    for x in X_test:
        state = res.run(x)
        res.reset()
        s_X_te.append(state[-1, np.newaxis])


    s_X_tr = np.array(s_X_tr)
    s_X_te = np.array(s_X_te)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    print(s_X_tr.shape, s_X_te.shape)
    print(Y_train.shape, Y_test.shape)


    # ================================================================================================
    def mse_fval(w, X, y):
        n_sample = X.shape[0]
        return (1.0 / n_sample) * np.sum((np.dot(X, w) - y)**2)
    
    def mse_grad(w, X, y):
        n_sample = X.shape[0]
        return (2.0 / n_sample) * np.dot(X.T, np.dot(X, w) - y)
    

    # softmax
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # 数值稳定
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def cross_entropy_loss(y, X, w, epsilon=1e-12):
        """
        y_true_onehot: (batch_size, num_classes) one-hot
        logits: (batch_size, num_classes) 未经过softmax
        """
        # 计算softmax概率
        probs = softmax(X @ w)
        
        # 裁剪避免log(0)
        probs = np.clip(probs, epsilon, 1.0)
        
        # 交叉熵损失
        loss = -np.sum(y * np.log(probs)) / y.shape[0]
        return loss

    def cross_entropy_gradient(y, X, w):
        """
        计算梯度: p - y
        """
        probs = softmax(X @ w)
        grad = probs - y
        return X.T @ grad



    # from scipy.special import logsumexp

    def logsumexp(b):
        """
        Computes logsumexp across columns
        """
        # B = np.max(b, axis=1, keepdims=True)
        B = np.max(b, axis=1)
        repmat_B = np.tile(B, (b.shape[1], 1)).T
        lse = np.log(np.sum(np.exp(b - repmat_B), axis=1)) + B
        return lse

    def sl1_fval(w, alpha, _lambda):
        n_feature = w.shape[0]
        
        lse = logsumexp(np.hstack([np.zeros((n_feature, 9)), alpha*w]))
        neg_lse = logsumexp(np.hstack([np.zeros((n_feature, 9)), -alpha*w]))

        # without bias 
        lambda_vec = (_lambda * np.ones(n_feature)).squeeze()
        fval = np.sum((lambda_vec * (1.0 / alpha)) * (lse + neg_lse))
        return fval
    
    def sl1_grad(w, alpha, _lambda):
        (n_feature, n_class) = w.shape

        w = w.reshape(-1, 1)

        # lse = logsumexp(np.hstack([np.zeros((n_feature, 9)), alpha*w]), axis=1)
        lse = logsumexp(np.hstack([np.zeros((n_feature*n_class, 1)), alpha*w]))

        lambda_vec = (_lambda * np.ones(n_feature*n_class)).squeeze()

        grad = (lambda_vec * (1.0 - 2.0 * np.exp(-lse))).reshape(-1, 1)

        grad = grad.reshape(n_feature, n_class)

        return grad
    
    def update_alpha(alpha, cnt=0):
        update1 = 1.5
        update2 = 1.25
        max_alpha = 1e7

        if cnt == 0:
            new_alpha = alpha * update1
        elif cnt > 0:
            new_alpha = alpha * update2
        else:
            new_alpha = alpha

        if new_alpha > max_alpha:
            new_alpha = max_alpha

        return new_alpha
    
    

    # ================================================================================================

    def sgd(X, y, learning_rate=1e-1, epochs=1000, batch_size=10):
        n_sample = X.shape[0]

        w = np.random.rand(UNITS, 9)
        # w = np.zeros((UNITS, 1))

        # for loss function
        alpha = 1.0
        thres = 1e-5
        _lambda = 1e-2

        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_sample)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_sample, batch_size):
                X_batch = X_shuffled[i:i+batch_size, 0, :]
                y_batch = y_shuffled[i:i+batch_size, 0, :]
                # X_batch shape = [1, 500]
                # y_batch shape = [1, 9]

                grad = cross_entropy_gradient(y_batch, X_batch, w) 
                grad /= batch_size
                grad += sl1_grad(w, alpha, _lambda)
                w -= learning_rate * grad
                # shrink to zero
                w[np.abs(w) < thres] = 0.0

            loss = cross_entropy_loss(y, X, w) + sl1_fval(w, alpha, _lambda)

            loss_history.append(loss)

            alpha = update_alpha(alpha, epoch)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss}, nonzeros={np.count_nonzero(w)}")

        return w, loss_history
    w, loss_history = sgd(s_X_tr, Y_train, learning_rate=1e-4, epochs=50000, batch_size=10)

    # calculate accuracy
    y_pred = np.argmax(softmax(s_X_te @ w), axis=1)
    acc = np.mean(y_pred == np.argmax(Y_test, axis=1))
    print(f"Accuracy: {acc}")


    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig("loss_history.png")

            
    # ================================================================================================

    
    

    

