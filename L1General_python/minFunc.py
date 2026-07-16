import numpy as np
from functools import partial

def polyinterp(points, doPlot=False, xminBound=None, xmaxBound=None):
    points = np.array(points, dtype=np.complex128)
    nPoints = points.shape[0]
    
    # Calculate order
    real_flags = np.imag(points[:, 1:3]) == 0
    order = np.sum(real_flags) - 1
    
    xmin = np.min(points[:, 0].real)
    xmax = np.max(points[:, 0].real)
    
    # Set default bounds
    if xminBound is None:
        xminBound = xmin
    if xmaxBound is None:
        xmaxBound = xmax
    
    # Handle special case for cubic interpolation
    if nPoints == 2 and order == 3 and not doPlot:
        minPos_idx = np.argmin(points[:, 0].real)
        notMinPos_idx = 1 - minPos_idx
        x1 = points[minPos_idx, 0].real
        x2 = points[notMinPos_idx, 0].real
        f1 = points[minPos_idx, 1].real
        f2 = points[notMinPos_idx, 1].real
        g1 = points[minPos_idx, 2].real
        g2 = points[notMinPos_idx, 2].real
        
        d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
        d2_sq = d1**2 - g1 * g2
        
        if d2_sq >= 0:
            d2 = np.sqrt(d2_sq)
            t = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            minPos = np.clip(t, xminBound, xmaxBound)
        else:
            minPos = (xminBound + xmaxBound) / 2
        
        return minPos.real, None # fmin is not defined for cubic interpolation

    raise NotImplementedError('Polynomial interpolation not implemented for this order.')


def isLegal(v):
    return np.all(np.isreal(v)) and not np.any(np.isnan(v)) and not np.any(np.isinf(v))

def ArmijoBacktrack(x, t, d, f, fr, g, gtd, c1, LS_interp, LS_multi, progTol, debug, doPlot, saveHessianComp, return_H, funObj, *args):
    """
    Backtracking linesearch to satisfy Armijo condition

    Inputs:
        x: starting location
        t: initial step size
        d: descent direction
        f: function value at starting location
        fr: reference function value (usually funObj(x))
        g: gradient at starting location
        gtd: directional derivative at starting location
        c1: sufficient decrease parameter
        LS_interp: type of interpolation
        LS_multi: whether to use multi-point interpolation
        progTol: minimum allowable step length
        debug: display debugging information
        doPlot: do a graphical display of interpolation
        saveHessianComp: whether to save Hessian computation
        funObj: objective function
        *args: parameters of objective function

    Outputs:
        t: step length
        x_new: new location after step
        f_new: function value at x + t*d
        g_new: gradient value at x + t*d
        funEvals: number of function evaluations performed by line search
        H: Hessian at initial guess (only computed if requested)
    """

    # Evaluate the Objective and Gradient at the Initial Step
    _funObj = partial(funObj, return_H=True)
    f_new, g_new, H = _funObj(x + t * d, *args)


    funEvals = 1

    g_newLegal = isLegal(g_new)
    while f_new > fr + c1 * t * gtd or not isLegal(f_new) or not g_newLegal:
        temp = t

        if LS_interp == 0 or not isLegal(f_new):
            # Ignore value of new point
            if debug:
                print('Fixed BT')
            t = 0.5 * t
        elif LS_interp == 1 or not g_newLegal:
            # Use function value at new point, but not its derivative
            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ quadratic interpolation based on two points
                if debug:
                    print('Quad BT')
                t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, np.nan]]), doPlot, 0, t)
            else:
                # Backtracking w/ cubic interpolation based on three points
                if debug:
                    print('Cubic BT')
                t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, np.nan], [t_prev, f_prev, np.nan]]), doPlot, 0, t)
        else:
            # Use function value and derivative at new point
            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ cubic interpolation w/ derivative
                if debug:
                    print('Grad-Cubic BT')
                t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, np.dot(g_new.squeeze(), d.squeeze())]]), doPlot, 0, t)
            elif not isLegal(g_prev):
                # Backtracking w/ quartic interpolation 3 points and derivative of two
                if debug:
                    print('Grad-Quartic BT')
                t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, np.dot(g_new.squeeze(), d.squeeze())], [t_prev, f_prev, np.nan]]), doPlot, 0, t)
            else:
                # Backtracking w/ quintic interpolation of 3 points and derivative of two
                if debug:
                    print('Grad-Quintic BT')
                t, _ = polyinterp(np.array([[0, f, gtd], [t, f_new, np.dot(g_new.squeeze(), d.squeeze())], [t_prev, f_prev, np.dot(g_prev, d)]]), doPlot, 0, t)

        # Adjust if change in t is too small/large
        if t < temp * 1e-3:
            if debug:
                print('Interpolated Value Too Small, Adjusting')
            t = temp * 1e-3
        elif t > temp * 0.6:
            if debug:
                print('Interpolated Value Too Large, Adjusting')
            t = temp * 0.6

        # Store old point if doing three-point interpolation
        if LS_multi:
            f_prev = f_new
            t_prev = temp
            if LS_interp == 2:
                g_prev = g_new

        # f_new, g_new, H = _funObj(x + t * d, *args)
        _funObj = partial(funObj, return_H=True)
        f_new, g_new, H = _funObj(x + t * d, *args)

        funEvals += 1
        g_newLegal = isLegal(g_new)

        # Check whether step size has become too small
        if np.max(np.abs(t * d)) <= progTol:
            if debug:
                print('Backtracking Line Search Failed')
            t = 0
            f_new = f
            g_new = g
            break

    # Evaluate Hessian at new point
    if return_H and funEvals > 1 and saveHessianComp:
        _funObj = partial(funObj, return_H=True)
        f_new, g_new, H = _funObj(x + t * d, *args)
        funEvals += 1

    x_new = x + t * d

    return t, x_new, f_new, g_new, funEvals, H