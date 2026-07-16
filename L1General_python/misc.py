import numpy as np

def mylogsumexp(b):
    """
    Computes logsumexp across columns
    """
    # B = np.max(b, axis=1, keepdims=True)
    B = np.max(b, axis=1)
    repmat_B = np.tile(B, (b.shape[1], 1)).T
    lse = np.log(np.sum(np.exp(b - repmat_B), axis=1)) + B
    return lse