import numpy as np
import pandas as pd
from functools import partial

from L1General_python.L1General_sub import solveNewton, sigmoidL1, initialStepLength, noProgress, L2_reg, no_reg
from L1General_python.minFunc import ArmijoBacktrack

def L1GeneralUnconstrainedApx_sub(grad_func, w, params, mode, *args):
    """
    Python implementation of l1GeneralSmooth_sub with continuation strategy
    
    Parameters:
    grad_func - function handle for gradient/Hessian computation
    w - initial weight vector
    params - dictionary of parameters
    mode - regularization mode
    *args - additional arguments for grad_func
    
    Returns:
    w - optimized weights
    f_evals - function evaluation count
    """
    
    # process parameters
    verbose = params.get('verbose', 1)
    threshold = params.get('threshold', 1e-4)
    opt_tol = params.get('optTol', 1e-6)
    prog_tol = params.get("progTol", 1e-9)
    max_iter = params.get('maxIter', 250)
    alpha_max = params.get('alpha', 5e4)
    update1 = params.get('update1', 1.25)
    update2 = params.get('update2', 1.5)
    adjust_step = params.get('adjustStep', 1)
    predict = params.get('predict', 0)

    if verbose:
        print("{:>10} {:>10} {:>15} {:>15} {:>15} {:>8} {:>15}".format(
            'Iteration', 'FunEvals', 'Step Length', 'Function Val', 
            'Opt Cond', 'Non-Zero', 'Alpha'))

    i = 0
    alpha_init = 1
    curr_param = alpha_init
    f_evals = 0

    _grad_func = partial(grad_func, return_H=True)
    f, g, H = _grad_func(w, curr_param, *args)

    f_evals += 1

    t = 1
    f_prev = f

    while i < max_iter:
        i += 1
        f_old = f

        # calculate search direction
        d = solveNewton(g, H)


        gtd = np.dot(g.squeeze(), d.squeeze())
        if gtd > -prog_tol:
            if verbose:
                print("Directional Derivative too small")
            break

        # initial step length selection
        t, f_prev = initialStepLength(i, adjust_step, 2, f, g, gtd, t, f_prev)

        # Armijo backtracking line search
        
        t, w, f, g, ls_evals, _ = ArmijoBacktrack(w, t, d, f, f, g, gtd,
                                                    1e-4, 2, 0, opt_tol,
                                                    max(verbose-1,0), 0, 1, 0,
                                                    grad_func, curr_param, *args)
        f_evals += ls_evals

        if verbose:
            opt_cond = np.sum(np.abs(g[np.abs(w) >= threshold]))
            non_zero = np.sum(np.abs(w) > threshold)
            print("{:10d} {:10d} {:15.5e} {:15.5e} {:15.5e} {:8d} {:15.5e}".format(
                i, f_evals, t, f, opt_cond, non_zero, curr_param))

        # update regularization parameter
        old_param = curr_param
        if ls_evals == 1:
            curr_param = min(curr_param * update2, alpha_max)
        else:
            curr_param = min(curr_param * update1, alpha_max)

        if verbose == 2 and curr_param >= alpha_max:
            print("At max alpha")

        # check convergence
        opt_condition = np.sum(np.abs(g[np.abs(w) >= threshold]))
        if opt_condition < opt_tol and old_param == alpha_max:
            if verbose:
                print("Solution Found")
            break

        if noProgress(t*d, f, f_old, prog_tol, verbose):
            break
        elif f_evals > max_iter:
            break

    w[np.abs(w) < threshold] = 0

    return w, f_evals

def L1GeneralUnconstrainedApx(gradFunc, w, _lambda, params, *args):
    """
    Computes argmin_w: gradFunc(w, *args) + sum lambda.*abs(w)

    Parameters:
        gradFunc - function of the form gradFunc(w, *args)
        w - initial guess (numpy array)
        lambda_ - scale of L1 penalty on each variable (numpy array)
        params - dictionary of user-modifiable parameters
        args - parameters of gradFunc

    Returns:
        w - optimized weights
        fEvals - number of function evaluations
    """
    # Process parameters
    verbose = params.get('verbose', 1)
    maxIter = params.get('maxIter', 250)
    optTol = params.get('optTol', 1e-6)
    mode = params.get('mode', 0)
    cont = params.get('cont', 1)
    alpha = params.get('alpha', 1e5)
    order = params.get('order', 2)
    adjustStep = params.get('adjustStep', 1)

    # Set parameters for optimization
    options = {
        'maxiter': maxIter,
        'ftol': optTol,
        'disp': verbose
    }

    # Choose unconstrained approximation function
    if mode == 0:
        apxFunc = sigmoidL1
    elif mode == 1:
        apxFunc = L2_reg
    elif mode == 2:
        apxFunc = no_reg
    else:
        raise NotImplementedError('Invalid mode for L1GeneralUnconstrainedApx')

    # Optimize
    if cont:
        # Continuation method
        w, funcCount = L1GeneralUnconstrainedApx_sub(apxFunc, w, params, order, gradFunc, _lambda, *args)
    else:
        raise NotImplementedError('Continuation method not implemented for L1GeneralUnconstrainedApx')


    return w, funcCount

