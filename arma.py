# Author: Alex Rubinsteyn 
# Email: alex ~dot~ rubinsteyn ~at~ gmail ~dot~ see-oh-em 
# License: LGPL  (http://www.gnu.org/licenses/lgpl.html)


import numpy as np 

class ARMA:
    """Autoregressive Moving Average 1D time series model with parameter
    estimation by stochastic gradient descent. 
    This is just an experiment!
    """
    def __init__(self, p = 2, q = 2, learning_rate = 10.0 ** -5, tol = 10.0 ** -5, verbose = True ):
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.tol = tol 
        self.verbose = verbose
        
        self.w_ma = None
        self.w_ar = None 
        
    def predict(self, X):
        p = self.p 
        q = self.q 
        k = max(p,q)
        n = X.shape[0]
        Y = np.zeros(n)
        w_ma = self.w_ma
        w_ar = self.w_ar 
        for i in np.arange(n - k):
            curr_idx = i + k
            x_prev = X[i + k - p : curr_idx]
            y_prev = Y[i + k - q : curr_idx]
            Y[curr_idx] = np.dot(x_prev, w_ma) + np.dot(y_prev, w_ar)
        return Y

    def fit(self, X, n_epochs = 100, plot=True):
        p = self.p
        q = self.q 
        w_ma = np.random.randn(p)
        w_ar = np.random.randn(q)
        k = max(p,q)
        n = X.shape[0]
        p_offset = k - p
        q_offset = k - q 
        Y = np.random.randn(n)
        ma_changes = [] 
        ar_changes = [] 
        errors = [] 
        learning_rate = self.learning_rate 
        for i in xrange(n_epochs):
            if self.verbose: print "Epoch ", i 
            old_ma = w_ma.copy()
            old_ar = w_ar.copy()
            for j in np.random.permutation(n - k):    
                curr_idx = j+k
                x_prev = X[j+p_offset : curr_idx]
                y_prev = Y[j+q_offset : curr_idx]
                pred = np.dot(x_prev, w_ma) + np.dot(y_prev, w_ar)
                Y[curr_idx] = pred 
                err = X[curr_idx] - pred
                w_ma += err * x_prev * learning_rate 
                w_ar += err * y_prev * learning_rate 
            ma_change = np.linalg.norm(old_ma - w_ma)
            ma_changes.append(ma_change)
            ar_change = np.linalg.norm(old_ar - w_ar)
            ar_changes.append(ar_change)
            mean_abs_error = np.mean(np.abs(Y[k:] - X[k:])) 
            errors.append(mean_abs_error)
            if self.verbose:
                print "MA weight change", ma_change
                print "AR weight change", ar_change 
                print "Mean abs. error:", mean_abs_error
            
            if ar_change < self.tol and ma_change < self.tol: break 
        if plot:
            import pylab
            pylab.plot(ma_changes)
            pylab.plot(ar_changes)
            pylab.legend(['MA', 'AR'])
            pylab.twinx()
            pylab.plot(errors, 'r')
        self.w_ma = w_ma
        self.w_ar = w_ar 

