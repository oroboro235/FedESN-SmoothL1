import numpy as np
import scipy
import scipy.sparse
from L1General_python.misc import mylogsumexp

def noProgress(td, f, f_old, opt_tol, verbose):
    x = 0
    if np.abs(f - f_old) < opt_tol:
        x = 1
        if verbose:
            print('Change in Objective below optTol')
    elif sum(np.abs(t) for t in td) < opt_tol:
        x = 1
        if verbose:
            print('Step Size below optTol')
    return x

def solveNewton(g, H, Hmodify=0, verbose=0):
    nVars = H.shape[0]
    
    if Hmodify == 0:
        try:
            R = scipy.linalg.cholesky(H, lower=False)
            y = scipy.linalg.solve(R.T, g)
            d = -scipy.linalg.solve(R, y)
        except np.linalg.LinAlgError:
            # adjust H matrix to make it positive definite
            # eigenvalues = scipy.linalg.eigvals(H).real
            eigenvalues = scipy.sparse.linalg.eigsh(H)
            min_eig = np.min(eigenvalues)
            adjust = max(0, 1e-12 - min_eig)
            H_modified = H + np.eye(nVars) * adjust
            d = -np.linalg.solve(H_modified, g)
    else:

        raise NotImplementedError('No implementation for modified Cholesky decomposition')
    
    return d

def sigmoidL1(w, alpha, gradFunc, lambda_, return_H=True, *args):
    # Handle gradFunc output based on number of return values
    [nll, g, H] = gradFunc(w=w, return_H=return_H, *args)

    if g.shape[1] == 1:

        p = w.shape[0]
        
        # e^0 = 1, that's way add 0 cols
        lse = mylogsumexp(np.hstack([np.zeros((p, 1)), alpha * w]))
        neg_lse = mylogsumexp(np.hstack([np.zeros((p, 1)), -alpha * w]))
        
        # Update negative log likelihood
        # lse here for calculating one single vector for one output weight, sum outside for all outputs' weights.
        nll += np.sum((lambda_.squeeze() * (1/alpha)) * (lse + neg_lse))
        
        # Calculate gradient if needed
        
        g += (lambda_.squeeze() * (1 - 2 * np.exp(-lse))).reshape(-1, 1)
        
        # Calculate Hessian if needed
        if H is not None:
            diag_terms = lambda_.squeeze() * 2 * alpha * np.exp(alpha * w.squeeze() - 2 * lse.squeeze())
            H += np.diag(diag_terms)
        
        return nll, g.reshape(-1, 1), H
    elif g.shape[1] > 1:
        # case 2: multiple output weights
        
        return nll, g, H
    else:
        raise ValueError('Invalid shape of gradient')


def L2_reg(w, alpha, gradFunc, lambda_, return_H=True, *args):
    # Handle gradFunc output based on number of return values
    [nll, g, H] = gradFunc(w=w, return_H=return_H, *args)
    
    # Update negative log likelihood
    w = w.ravel(order='F')
    nll += np.sum(lambda_.T @ w**2)
    
    # Calculate gradient if needed
    # g += lambda_.squeeze() * w
    g += 2 * lambda_.reshape(-1, 1) * w.reshape(-1, 1)
    
    # Calculate Hessian if needed
    if H is not None:
        H += 2 * np.diag(lambda_.squeeze())
    
    return nll, g.reshape(-1, 1), H

def no_reg(w, alpha, gradFunc, lambda_, return_H=True, *args):
    # Handle gradFunc output based on number of return values
    [nll, g, H] = gradFunc(w=w, return_H=return_H, *args)
    
    return nll, g.reshape(-1, 1), H

def initialStepLength(i, adjustStep, order, f, g, gtd, t, f_prev):
    if i == 1 or adjustStep == 0:
        t = 1
    else:
        t = np.min([1, 2*(f - f_prev) / gtd])

    if i == 1 and order < 2:
        t = np.min([1, 1/np.sum(np.abs(g))])

    f_prev = f

    return t, f_prev