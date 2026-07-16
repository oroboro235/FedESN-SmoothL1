import numpy as np
# partial
from functools import partial

def mylogsumexp(b):
    """
    Computes logsumexp across columns
    """
    # B = np.max(b, axis=1, keepdims=True)
    B = np.max(b, axis=1)
    repmat_B = np.tile(B, (b.shape[1], 1)).T
    lse = np.log(np.sum(np.exp(b - repmat_B), axis=1)) + B
    return lse

def func_logisticRegression(w, X, y, k):
    # softmax loss
    m, n = X.shape
    w = w.reshape((n, k-1), order='F')
    w = np.hstack((w, np.zeros((n, 1))))
    logits = np.dot(X, w)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    Z = np.sum(np.exp(logits), axis=1)
    correct_logits = logits[np.arange(m), y.squeeze()]
    return -np.sum(correct_logits - np.log(Z))

def grad_logisticRegression(w, X, y, k):
    m, n = X.shape
    w = w.reshape((n, k-1), order='F')
    w = np.hstack((w, np.zeros((n, 1))))
    logits = np.dot(X, w)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    Z = np.sum(np.exp(logits), axis=1)
    g = np.zeros((n, k-1))
    for c in range(k-1):
        indicator = (y.squeeze() == c).astype(float)
        prob_c = np.exp(logits[:, c]) / Z
        grad_term = indicator - prob_c
        g[:, c] = np.dot(-X.T, grad_term)
    return g.ravel(order='F')

def hess_logisticRegression(w, X, y, k):
    m, n = X.shape
    w = w.reshape((n, k-1), order='F')
    w = np.hstack((w, np.zeros((n, 1))))
    logits = np.dot(X, w)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    Z = np.sum(np.exp(logits), axis=1)
    SM = np.exp(logits[:, :k-1]) / Z[:, None]  # tile Propagate exp(logits) to SM
    H = np.zeros((n*(k-1), n*(k-1)))
    for c1 in range(k-1):
        for c2 in range(k-1):
            delta = 1.0 if c1 == c2 else 0.0
            D = SM[:, c1] * (delta - SM[:, c2])
            H_block = np.dot(X.T, (X * D[:, np.newaxis]))  # dense matrix multiplication
            H[n*c1:n*(c1+1), n*c2:n*(c2+1)] = H_block
    return H

def func_linearRegression(w, X, y):
    m, n = X.shape
    y = y.squeeze()
    w = w.squeeze()
    XX = np.matmul(X.T, X)

    if m < n:
        Xw = np.matmul(X, w)
        res = Xw - y
        f = np.sum(res**2)

    else:
        XXw = np.matmul(XX, w)
        Xy = np.matmul(X.T, y)
        f = np.matmul(w.T, XXw) - 2*np.matmul(w.T, Xy) + np.matmul(y.T, y)
        f = np.sum(f)
    
    return f

def grad_linearRegression(w, X, y):
    m, n = X.shape
    y = y.squeeze()
    w = w.squeeze()
    XX = np.matmul(X.T, X)

    if m < n:
        Xw = np.matmul(X, w)
        res = Xw - y
        g = 2*np.matmul(X.T, res)
    else:
        XXw = np.matmul(XX, w)
        Xy = np.matmul(X.T, y)
        g = 2*XXw - 2*Xy
    return g.ravel(order='F')

def hess_linearRegression(w, X, y):
    XX = np.matmul(X.T, X)
    return 2*XX



# regularization term
# no regularization
def func_noneReg(w, objFunc, lambda_, *args):
    return objFunc(w, *args)
def grad_noneReg(w, gradFunc, lambda_, *args):
    return gradFunc(w, *args)
def hess_noneReg(w, hessFunc, lambda_, *args):
    return hessFunc(w, *args)

# L2 regularization
def func_L2Reg(w, objFunc, lambda_, *args):
    f = objFunc(w, *args)
    f_reg = f + lambda_ * np.sum(w**2)
    return f_reg
def grad_L2Reg(w, gradFunc, lambda_, *args):
    g = gradFunc(w, *args)
    g_reg = g + 2*lambda_*w
    return g_reg
def hess_L2Reg(w, hessFunc, lambda_, *args):
    H = hessFunc(w, *args)
    H_reg = H + 2*lambda_*np.eye(len(w))
    return H_reg


# smoothL1 regularization
def func_smoothL1Reg(w, objFunc, lambda_, *args):
    f = objFunc(w, *args)
    len_w = w.shape[0]
    alpha = 5e4

    lse = mylogsumexp(np.hstack((np.zeros((len_w,1)), alpha*w.reshape(-1,1))))
    neg_lse = mylogsumexp(np.hstack((np.zeros((len_w, 1)), -alpha*w.reshape(-1,1))))

    f_reg = f + lambda_ * np.sum((1/alpha) * (lse + neg_lse))
    return f_reg
def grad_smoothL1Reg(w, gradFunc, lambda_, *args):
    g = gradFunc(w, *args)
    len_w = w.shape[0]
    alpha = 5e4

    lse = mylogsumexp(np.hstack((np.zeros((len_w, 1)), alpha*w.reshape(-1,1))))
    g_reg = g + (lambda_ * (1 - 2 * np.exp(-lse)))
    return g_reg.ravel(order='F')
def hess_smoothL1Reg(w, hessFunc, lambda_, *args):
    H = hessFunc(w, *args)
    len_w = w.shape[0]
    alpha = 5e4

    lse = mylogsumexp(np.hstack((np.zeros((len_w, 1)), alpha*w.reshape(-1,1))))
    diag_terms = lambda_ * 2 * alpha * np.exp(alpha * w - 2 * lse.squeeze())
    H_reg = H + np.diag(diag_terms)
    return H_reg

# return loss function
def get_loss_func(X, y, task_type, reg_type, lambda_, k=None):
    if task_type == 'logistic_regression':
        if k == None:
            raise ValueError("k must be specified for logistic regression")
        func = partial(func_logisticRegression, X=X, y=y, k=k)
        grad = partial(grad_logisticRegression, X=X, y=y, k=k)
        hess = partial(hess_logisticRegression, X=X, y=y, k=k)
        if reg_type == 'none':
            loss_func = partial(func_noneReg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_noneReg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_noneReg, hessFunc=hess, lambda_=lambda_)
        elif reg_type == 'l2':
            loss_func = partial(func_L2Reg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_L2Reg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_L2Reg, hessFunc=hess, lambda_=lambda_)
        elif reg_type =='smoothl1':
            loss_func = partial(func_smoothL1Reg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_smoothL1Reg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_smoothL1Reg, hessFunc=hess, lambda_=lambda_)
        else:
            raise ValueError("Invalid regularization type")
    elif task_type == 'linear_regression':
        func = partial(func_linearRegression, X=X, y=y)
        grad = partial(grad_linearRegression, X=X, y=y)
        hess = partial(hess_linearRegression, X=X, y=y)
        if reg_type == 'none':
            loss_func = partial(func_noneReg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_noneReg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_noneReg, hessFunc=hess, lambda_=lambda_)
        elif reg_type == 'L2':
            loss_func = partial(func_L2Reg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_L2Reg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_L2Reg, hessFunc=hess, lambda_=lambda_)
        elif reg_type =='smoothl1':
            loss_func = partial(func_smoothL1Reg, objFunc=func, lambda_=lambda_)
            grad_func = partial(grad_smoothL1Reg, gradFunc=grad, lambda_=lambda_)
            hess_func = partial(hess_smoothL1Reg, hessFunc=hess, lambda_=lambda_)
        else:
            raise ValueError("Invalid regularization type")
    else:
        raise ValueError("Invalid task type")
    
    return loss_func, grad_func, hess_func