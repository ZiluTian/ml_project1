import numpy as np
import itertools as it

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    N = y.shape[0]
    Nt = int(ratio*N)
    it = np.random.choice(N-1, Nt, replace=False)
    xt = x[it]
    yt = y[it]
    xtest = np.delete(x, it, 0)
    ytest = np.delete(y, it, 0)
    return xt, yt, xtest, ytest


def compute_mse(y, tx, w):
    """Compute the loss using MSE."""
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient for MSE."""
    
    N=y.shape[0]
    e = y - tx @ w
    gradient = -1/N * tx.T @ e
    return gradient


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_poly_plus(x, degree):
    """
    Builds polynomial basis function of a certain degree combining all features.
    """
    poly = np.ones((len(x), 1))

    for deg in range(1, degree+1):
        if deg == 1:
            poly = np.c_[poly, x]
        else:
            for i in it.combinations_with_replacement(range(x.shape[1]),deg):
                poly = np.c_[poly, np.prod(x[:,i],1)]
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def find_desired_var(var_space, y, tx, k_fold, model_name, *func_args): 
    """Return the value of variable var that has least rmse_te error over var_space for the given model
    i.e. find lambda for ridge regression, or find gamma for logistic regression, 
    (note that var has to be the last argument of the function parameters)
    along with the rmse_tr and rmse_te which can be used for visualization purpose"""
    rmse_tr = []
    rmse_te = []
    k_indices = build_k_indices(y, k_fold, 1)
    
    # var: lambda for ridge regression, or gamma for other iterative methods 
    for var in var_space:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te, w = cross_validation_helper(y, tx, k_indices, k, model_name, *func_args, var)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(np.sqrt(2*rmse_tr_tmp)))
        rmse_te.append(np.mean(np.sqrt(2*rmse_te_tmp)))
        
    opt_var = var_space[np.argmin(rmse_te)]
    return opt_var, rmse_tr, rmse_te
      
    
def find_weight(y, tx, k_fold, model_name, *func_args): 
    """Return averaged weight across runs of the given model"""
    k_indices = build_k_indices(y, k_fold, 1)
    rmse_tr_tmp = []
    w_init = np.zeros(tx.shape[1])
    
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation_helper(y, tx, k_indices, k, model_name, *func_args)
        w = w_init + w
        rmse_tr_tmp.append(loss_tr)
    return w/k_fold, np.mean(np.sqrt(2*rmse_tr_tmp))
        
    
def cross_validation_helper(y, tx, k_indices, k, model_name, *func_args): 
    """Return the loss of given function
    valid func_name: ridge_regression, logistic_regression, least_squares_GD, least_squares_SGD, least_squares, and reg_logistic_regression
    *func_args take arguments to the function aside from y and tx"""
    # ***************************************************
    
    y_te = y[k_indices[k]]    
    tx_te = tx[k_indices[k]]
               
    y_tr = np.delete(y, k_indices[k], axis=0)
    tx_tr = np.delete(tx, k_indices[k], axis=0)
               
    # ***************************************************
    w, _ = model_name(y_tr, tx_tr, *func_args)
    # ***************************************************
    loss_te = compute_mse(y_te, tx_te, w)
    loss_tr = compute_mse(y_tr, tx_tr, w)
    # ***************************************************
    return loss_tr, loss_te, w
               
               
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    
    # It returns a generator object relative to a batch of size batch_size (tuple of (y,tx)) over which you can iterate only once
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
def compute_score(y_test, y_pred):
    if len(y_pred)== len(y_test):
        ones_aux = np.ones(len(y_pred))
        correct = np.sum(ones_aux[np.equal(y_test, y_pred)])
        incorrect = len(y_pred)-correct
        perc = correct / len(y_pred) * 100
        print("Total correct:", correct, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")
    else:
        print("Data have different sizes.")
    
    

