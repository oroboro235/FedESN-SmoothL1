import numpy as np
from scipy.sparse import diags

def softmaxLoss2(w, X, y, k, return_H=True):
    import cupy as cp
    n, p = X.shape
    X = cp.asarray(X)
    y = cp.asarray(y)
    w = w.reshape((p, k-1), order='F')
    w = cp.hstack((w, cp.zeros((p, 1))))  # last class is assumed to be 0

    # calculate logits and Z
    logits = cp.matmul(X, w)
    logits = logits-cp.max(logits, axis=1, keepdims=True)  # prevent overflow
    Z = cp.sum(cp.exp(logits), axis=1)
    correct_logits = logits[cp.arange(n), y.squeeze()]
    nll = -cp.sum(correct_logits - cp.log(Z))

    # calculate gradient
    g = cp.zeros((p, k-1))
    for c in range(k-1):
        indicator = (y.squeeze() == c).astype(float)
        prob_c = cp.exp(logits[:, c]) / Z
        grad_term = indicator - prob_c
        g[:, c] = cp.matmul(-X.T, grad_term)
    g = g.ravel(order='F')

    # calculate Hessian
    H = None
    if return_H:
        SM = cp.exp(logits[:, :k-1]) / Z[:, None]  # tile Propagate exp(logits) to SM
        H = cp.zeros((p*(k-1), p*(k-1)))
        for c1 in range(k-1):
            for c2 in range(k-1):
                delta = 1.0 if c1 == c2 else 0.0
                D = SM[:, c1] * (delta - SM[:, c2])
                H_block = cp.matmul(X.T, (X * D[:, cp.newaxis]))  # dense matrix multiplication
                H[p*c1:p*(c1+1), p*c2:p*(c2+1)] = H_block
    else:
        H = None

    # return nll, g.reshape(-1, 1), H
    return nll.get(), g.get().reshape(-1, 1), H.get()

def softmaxLoss2_noCupy(w, X, y, k, return_H=True):
    n, p = X.shape
    w = w.reshape((p, k-1), order='F')
    w = np.hstack((w, np.zeros((p, 1))))  # last class is assumed to be 0

    # calculate logits and Z
    logits = np.dot(X, w)
    logits = logits-np.max(logits, axis=1, keepdims=True)  # prevent overflow
    Z = np.sum(np.exp(logits), axis=1)
    
    correct_logits = logits[np.arange(n), y.squeeze()]
    nll = -np.sum(correct_logits - np.log(Z))

    # calculate gradient
    g = np.zeros((p, k-1))
    for c in range(k-1):
        indicator = (y.squeeze() == c).astype(float)
        prob_c = np.exp(logits[:, c]) / Z
        grad_term = indicator - prob_c
        g[:, c] = np.dot(-X.T, grad_term)
    g = g.ravel(order='F')

    # calculate Hessian
    H = None
    if return_H:
        SM = np.exp(logits[:, :k-1]) / Z[:, None]  # tile Propagate exp(logits) to SM
        H = np.zeros((p*(k-1), p*(k-1)))
        for c1 in range(k-1):
            for c2 in range(k-1):
                delta = 1.0 if c1 == c2 else 0.0
                D = SM[:, c1] * (delta - SM[:, c2])
                H_block = np.dot(X.T, (X * D[:, np.newaxis]))  # dense matrix multiplication
                H[p*c1:p*(c1+1), p*c2:p*(c2+1)] = H_block
    else:
        H = None

    # return nll, g.reshape(-1, 1), H
    return nll, g.reshape(-1, 1), H


def make_squared_error(X, Y, use_cupy=False):
    """Precomputed least-squares objective factory for the L1General solver.

    The Gram matrix ``XX = XᵀX`` and the Hessian ``H = 2·XX`` are constant w.r.t.
    the weights *and* shared across every output column, while ``Xy`` / ``yy`` are
    constant per column.  ``SquaredError`` recomputes all of these on *every*
    L1General function/gradient evaluation (O(n·p²) each, hundreds of times per
    fit); here they are computed once so each evaluation is a cheap O(p²) mat-vec.

    Args:
        X:        Design matrix, shape (n, p).
        Y:        Targets, shape (n, k) — all output columns at once.
        use_cupy: Compute the Gram products on GPU (results returned as NumPy,
                  since the L1General solver runs on the CPU).

    Returns:
        ``objective(i) -> obj``, where ``obj(w, return_H=True) -> (f, g, H)`` is
        the squared-error objective for output column *i*:
            f(w) = wᵀXXw − 2·wᵀXy_i + yy_i,  g(w) = 2·(XXw − Xy_i),  H = 2·XX.
    """
    if use_cupy:
        import cupy as xp
        to_np = xp.asnumpy
    else:
        xp = np
        def to_np(a): return a

    Xg = xp.asarray(X)
    Yg = xp.asarray(Y)
    XX = Xg.T @ Xg                       # (p, p) — shared across all columns
    XY = Xg.T @ Yg                       # (p, k)
    YY = xp.sum(Yg * Yg, axis=0)         # (k,)
    H_np = to_np(2.0 * XX)               # constant Hessian, as NumPy for the solver

    def objective(i):
        Xy = XY[:, i:i + 1]
        yy = float(to_np(YY[i]))
        def obj(w, return_H=True, **_):
            wg  = xp.asarray(w).reshape(-1, 1)
            XXw = XX @ wg
            f   = float(to_np((wg.T @ XXw - 2.0 * wg.T @ Xy).squeeze())) + yy
            g   = to_np(2.0 * (XXw - Xy)).reshape(-1, 1)
            # Return a *fresh* copy of the constant Hessian: the L1General apx
            # functions (sigmoidL1 / L2_reg) augment H in place (H += diag(...)),
            # so handing out the shared H_np would accumulate the regulariser's
            # diagonal across evaluations and columns — inflating the Hessian,
            # shrinking the Newton step, and roughly doubling the solver's
            # iteration count (the final g-based solution is unaffected, only
            # convergence speed). The copy is O(p²), cheap next to the O(n·p²)
            # Gram recompute this factory exists to avoid.
            return f, g, (H_np.copy() if return_H else None)
        return obj

    return objective


def SquaredError(w, X, y, return_H=True):
    import cupy as cp
    n, p = X.shape
    X = cp.asarray(X)
    y = cp.asarray(y).squeeze()
    w = cp.asarray(w).squeeze()
    XX = cp.matmul(X.T, X)

    if n < p:
        Xw = cp.matmul(X, w)
        res = Xw - y
        f = cp.sum(res**2)
        g = 2*cp.matmul(X.T, res)
    else:
        # XXw = XX @ w
        XXw = cp.matmul(XX, w)
        # Xy = X.T @ y
        Xy = cp.matmul(X.T, y)
        f = cp.matmul(w.T, XXw) - 2*cp.matmul(w.T, Xy) + cp.matmul(y.T, y)
        f = cp.sum(f)
        g = 2*XXw - 2*Xy

    if return_H:
        H = 2*XX
    else:
        H = None
        
    return f.get(), g.get().reshape(-1, 1), H.get()

def SquaredError_noCupy(w, X, y, return_H=True):
    # Sum of Squared Error
    # case 1 : y has only one column
    if y.shape[1] == 1:
        n, p = X.shape
        # w = w.squeeze()
        XX = np.matmul(X.T, X)

        if n < p:
            Xw = np.matmul(X, w)
            res = Xw - y.reshape(-1, 1)
            f = np.sum(res**2)
            g = 2*np.matmul(X.T, res)
        else:
            # XXw = XX @ w
            XXw = np.matmul(XX, w)
            # Xy = X.T @ y
            Xy = np.matmul(X.T, y)
            f = np.matmul(w.T, XXw) - 2*np.matmul(w.T, Xy) + np.matmul(y.T, y)
            f = np.sum(f)
            g = 2*XXw - 2*Xy.reshape(-1, 1)


        if return_H:
            H = 2*XX
        else:
            H = None
            
        return f, g.reshape(-1, 1), H
    
    elif y.shape[1] > 1:
        # case 2 : y has multiple columns
        n, p = X.shape
        k = y.shape[1]
        W = w.reshape(p, k)
        XX = np.matmul(X.T, X)
    
        # f = ||XW - Y||^2_Fro
        XW = np.matmul(X, W)
        E = XW - y
        # Frobenius norm
        f = np.linalg.norm(E, 'fro')**2

        
        # g = 2*X^T(XW - Y) = 2*X^T*E
        g = 2*np.matmul(X.T, E)

        # H has dim (p*k, p*k), only diagonal blocks are non-zero(X^T*X)
        if return_H:
            H = 2*np.kron(np.eye(k), XX)
        else:
            H = None

        return f, g.reshape(-1, 1) , H

    else:
        raise ValueError("y should have at least one column")






if __name__ == '__main__':
    import pandas as pd
    m, n, k = 1000, 500, 3
    X = np.random.randn(m, n)
    y = np.random.randn(m, k)
    w = np.random.randn(n, k).reshape(-1, 1)
    # w = pd.read_csv("w_mat.csv", header=None).values
    # X = pd.read_csv("X_mat.csv", header=None).values
    # y = pd.read_csv("y_mat.csv", header=None).values.reshape(-1, 1)

    f, g, H = SquaredError(w, X, y)
    # f, g, H = SquaredError_multiTask(w, X, y)
    # f, g, H = SquaredError_multivariateLR(w, X, y)
    print(f.shape, g.shape, H.shape)