import numpy as np
from scipy.sparse import diags
from utils import logsumexp


# ===========================================================================
#                          Objective functions
# ===========================================================================

class SumSquaredError:
    """
    Docstring for SumSquaredError
    """
    def __init__(
            self,
            X: np.ndarray, 
            y: np.ndarray, 
            if_hess: bool = False
        ):
        if X is None or y is None:
            raise ValueError("X and y must be provided")
        
        # check ndim of y
        if y.ndim == 1:
            y.reshape(-1, 1)
        

        # check if X and y have the same number of samples
        if X.shape[0]!= y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.X = X
        self.y = y
        self.if_hess = if_hess
        
        # pre-cal
        if self.if_hess:
            self.XX = np.dot(X.T, X)

    
    def cal_res(self, w):
        return np.dot(self.X, w) - self.y

    def cal_func_val(self, w):
        res = self.cal_res(w)
        return np.sum(res**2)

    def cal_grad_val(self, w):
        res = self.cal_res(w)
        return 2*np.dot(self.X.T, res)

    def cal_hess_val(self, w):
        if self.if_hess:
            return 2*self.XX
        else:
            return None
    
    def cal_val(self, w):
        """Calculate all values at once"""
        res = self.cal_res(w)
        func_val = np.sum(res**2)
        grad_val = 2*np.dot(self.X.T, res)
        hess_val = 2*self.XX if self.if_hess else None
        return func_val, grad_val, hess_val
        

class SoftmaxLoss:
    """
    Softmax loss function for multi-class classification
    """
    def __init__(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            k: int = None,
            if_hess: bool = False
        ):
        if X is None or y is None:
            raise ValueError("X and y must be provided")
        
        # check ndim of y
        if y.ndim > 1:
            raise ValueError("y must be a 1D array")
        
        # check if X and y have the same number of samples
        if X.shape[0]!= y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.y = y
        if k is None:
            self.k = np.unique(y).shape[0]
        else:
            self.k = k

        self.if_hess = if_hess

    def check_w_dim(self, w):
        shape = w.shape
        if len(shape)!= 1:
            raise ValueError("w must be a 1D array")
        if shape[0]!= self.n_features:
            raise ValueError(f"w must have {self.n_features} rows (features)")
        if shape[1]!= self.k-1:
            raise ValueError(f"w must have {self.k} columns for {self.k} classes")

    # def cal_logits_Z(self, w):
    #     self.check_w_dim(w)
    #     logits = np.dot(self.X, w)
    #     return logits - np.max(logits, axis=1, keepdims=True), np.sum(np.exp(logits), axis=1) # prevent overflow

    def cal_logits_Z(self, w):
        """Calculate logits efficiently"""
        self.check_w_dim(w)
        logits = np.dot(self.X, w)
        # Numerical stability: subtract max along each row
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits = logits - logits_max
        return logits

    def cal_func_val(self, w):
        logits, Z = self.cal_logits_Z(w)
        correct_logits = logits[np.arange(self.n_samples), self.y]
        return -np.sum(np.log(Z)) + np.sum(correct_logits)


    def cal_grad_val(self, w):
        logits, Z = self.cal_logits_Z(w)
        grad_val = np.zeros((self.n_samples, self.k-1))
        for c in range(self.k-1):
            indicator = (self.y.squeeze() == c).astype(float)
            prob_c = np.exp(logits[:, c]) / Z
            grad_term = indicator - prob_c
            grad_val[:, c] = np.dot(-self.X.T, grad_term)
        return grad_val.flatten()

    def cal_hess_val(self, w):
        if not self.if_hess:
            return None
        else:
            logits, Z = self.cal_logits_Z(w)
            SM = np.exp(logits[:, :self.k-1]) / Z[:, None]  # tile Propagate exp(logits) to SM
            hess_val = np.zeros((self.n_features*(self.k-1), self.n_features*(self.k-1)))
            for c1 in range(self.k-1):
                for c2 in range(self.k-1):
                    delta = 1.0 if c1 == c2 else 0.0
                    D = SM[:, c1] * (delta - SM[:, c2])
                    H_block = np.dot(self.X.T, (self.X * D[:, np.newaxis]))  # dense matrix multiplication
                    hess_val[self.n_features*c1:self.n_features*(c1+1), self.n_features*c2:self.n_features*(c2+1)] = H_block
            return hess_val

    def cal_val(self, w):
        logits, Z = self.cal_logits_Z(w)
        
        # func val
        func_val = -np.sum(np.log(Z)) + np.sum(logits[np.arange(self.n_samples), self.y])

        # grad val
        grad_val = np.zeros((self.n_samples, self.k-1))
        for c in range(self.k-1):
            indicator = (self.y.squeeze() == c).astype(float)
            prob_c = np.exp(logits[:, c]) / Z
            grad_term = indicator - prob_c
            grad_val[:, c] = np.dot(-self.X.T, grad_term)
        grad_val = grad_val.flatten()

        # hess val
        if self.if_hess:
            SM = np.exp(logits[:, :self.k-1]) / Z[:, None]  # tile Propagate exp(logits) to SM
            hess_val = np.zeros((self.n_features*(self.k-1), self.n_features*(self.k-1)))
            for c1 in range(self.k-1):
                for c2 in range(self.k-1):
                    delta = 1.0 if c1 == c2 else 0.0
                    D = SM[:, c1] * (delta - SM[:, c2])
                    H_block = np.dot(self.X.T, (self.X * D[:, np.newaxis]))  # dense matrix multiplication
                    hess_val[self.n_features*c1:self.n_features*(c1+1), self.n_features*c2:self.n_features*(c2+1)] = H_block
        else:
            hess_val = None

        return func_val, grad_val, hess_val

        

# ===========================================================================
#                         Regularization functions  
# ===========================================================================

class SmoothL1_reg:
    """
    Docstring for SmoothL1
    """
    def __init__(
            self, 
            initial_alpha: float = 1.0, 
            update1: float = 1.5, 
            update2: float = 1.25, 
            max_alpha: float = 5e4, 
            reg_param: float = 1.0, 
            if_bias: bool = False, 
            if_hess: bool = False
        ):
        self.initial_alpha = initial_alpha
        self.alpha = initial_alpha # the smoothness parameter
        self.if_bias = if_bias
        self.if_hess = if_hess

        self.reg_param = reg_param

        self.round = 0
        self.update1 = update1
        self.update2 = update2
        self.max_alpha = max_alpha

    def update_alpha(self):
        if self.round == 0:
            self.alpha *= self.update1
        elif self.round >= 1:
            self.alpha *= self.update2
        
        if self.alpha > self.max_alpha:
            self.alpha = self.max_alpha

        self.round += 1

    def reset_round(self):
        self.round = 0
        self.alpha = self.initial_alpha

    def get_lambda_vec(self, w_shape):
        # return lambda_vec in dim (p, )
        p, q = w_shape
        if self.if_bias:
            lambda_vec = np.hstack((0.0, self.reg_param * np.ones(p-1)))
        else:
            lambda_vec = self.reg_param * np.ones(p)
        return lambda_vec.squeeze()
    
    def cal_func_val(self, w, alpha : float = None):
        n_features = w.shape[0]

        if alpha is None:
            alpha = self.alpha

        lse = logsumexp(np.hstack([np.zeros((n_features, 1)), alpha * w]), axis=1)
        neg_lse = logsumexp(np.hstack([np.zeros((n_features, 1)), -alpha * w]), axis=1)

        # Update negative log likelihood
        # lse here for calculating one single vector for one output weight 
        # sum outside for all outputs' weights.
        lambda_vec = self.get_lambda_vec(w.shape)
        func_val = np.sum((lambda_vec * (1.0 / alpha)) * (lse + neg_lse))

        return func_val

    def cal_grad_val(self, w, alpha : float = None):
        n_features = w.shape[0]

        if alpha is None:
            alpha = self.alpha

        lse = logsumexp(np.hstack([np.zeros((n_features, 1)), alpha * w]))
    
        lambda_vec = self.get_lambda_vec(w.shape)
        
        grad_val = (lambda_vec * (1.0 - 2.0 * np.exp(-lse))).reshape(-1, 1)

        return grad_val

    def cal_hess_val(self, w, alpha : float = None):
        if not self.if_hess:
            return None
        else:
            n_features = w.shape[0]

            if alpha is None:
                alpha = self.alpha

            lse = logsumexp(np.hstack([np.zeros((n_features, 1)), alpha * w]))
        
            lambda_vec = self.get_lambda_vec(w.shape)

            diag_terms = lambda_vec * 2.0 * alpha * np.exp(alpha * w.squeeze() - 2.0 * lse.squeeze())

            hess_val = np.diag(diag_terms)

            return hess_val

    def cal_val(self, w, alpha : float = None):
        n_features = w.shape[0]

        if alpha is None:
            alpha = self.alpha

        lse = logsumexp(np.hstack([np.zeros((n_features, 1)), alpha * w]))
        neg_lse = logsumexp(np.hstack([np.zeros((n_features, 1)), -alpha * w]))

        # Update negative log likelihood
        # lse here for calculating one single vector for one output weight 
        # sum outside for all outputs' weights.
        lambda_vec = self.get_lambda_vec(w.shape)
        func_val = -np.sum((lambda_vec * (1.0 / alpha)) * (lse + neg_lse))

        grad_val = (lambda_vec * (1.0 - 2.0 * np.exp(-lse))).reshape(-1, 1)

        if self.if_hess:
            diag_terms = lambda_vec * 2.0 * alpha * np.exp(alpha * w.squeeze() - 2.0 * lse.squeeze())
            hess_val = np.diag(diag_terms)
        else:
            hess_val = None

        self.update_alpha()

        return func_val, grad_val, hess_val

# ===========================================================================
#                         Combination functions  
# ===========================================================================

class Combinator:
    """
    class for combining the objective function and regularization function
    """
    obj_func = None
    reg_func = None

    def __init__(
            self,
            obj_func,
            reg_func,
            if_hess: bool = False
        ):
        self.obj_func = obj_func
        self.reg_func = reg_func
        self.if_hess = if_hess

    def cal_func_val(self, w):
        obj_val = self.obj_func.cal_func_val(w)
        reg_val = self.reg_func.cal_func_val(w)
        return obj_val + reg_val

    def cal_grad_val(self, w):
        obj_grad = self.obj_func.cal_grad_val(w)
        reg_grad = self.reg_func.cal_grad_val(w)
        return obj_grad + reg_grad

    def cal_hess_val(self, w):
        if not self.if_hess:
            return None
        else:
            obj_hess = self.obj_func.cal_hess_val(w)
            reg_hess = self.reg_func.cal_hess_val(w)
            return obj_hess + reg_hess

    def cal_val(self, w):
        obj_val, obj_grad, obj_hess = self.obj_func.cal_val(w)
        reg_val, reg_grad, reg_hess = self.reg_func.cal_val(w)
        func_val = obj_val + reg_val
        grad_val = obj_grad + reg_grad
        if self.if_hess:
            hess_val = obj_hess + reg_hess
        else:
            hess_val = None
        return func_val, grad_val, hess_val
    