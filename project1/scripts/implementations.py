import numpy as np
from proj1_helpers import load_csv_data  

def get_missing_index(input_data):
    """Retrieve the indices of missing values in input_data. Missing values are denoted by -999"""
    missing_ind = (input_data[:,4]==-999)
    col_num = input_data.shape[1]
    for col in range(col_num): 
        missing_ind = (missing_ind | (input_data[:,col]==-999))
    return missing_ind 

def load_clean_csv(data_path, sub_sample=False, missing_val="ignore", normalized=True): 
    """Load clean csv, specify data_path, sub_sample(True/False), missing_val(ignore, avg, median), normalized(True/False)
    Return yb, input_data, and ids"""
    yb, input_data, ids = load_csv_data(data_path, sub_sample)
    missing_ind = get_missing_index(input_data)
    
    incomplete_features = np.unique(np.where(input_data == -999.0)[1])
    print('Features with missing data: ', incomplete_features)
    
    if (missing_val=="avg"): 
        mean = np.mean(input_data[~missing_ind], 0)
        print(mean)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, mean[i])
    elif (missing_val=="median"): 
        median = np.median(input_data[~missing_ind], 0)
        print(median)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, median[i])
    else:  
        yb = yb[~missing_ind]
        input_data = input_data[~missing_ind]
        ids = ids[~missing_ind]
    # if not ignore missing val, replace it with average 
        
    if normalized: 
        input_m = np.mean(input_data,0)
        input_std = np.std(input_data, 0)
        input_data = (input_data - input_m)/input_std
    
    return yb, input_data, ids


        
def compute_mse(y, tx, w):
    """Compute the loss using MSE."""
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    
    return mse


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


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


def compute_gradient(y, tx, w):
    """Compute the gradient for MSE."""
    
    N=y.shape[0]
    e = y - tx @ w
    gradient = -1/N * tx.T @ e
    
    return gradient


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

            
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent algorithm."""
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient for MSE
        g = compute_gradient(y, tx, w)        
        loss = compute_mse(y, tx, w)
        # update parameters vector
        w = w - gamma*g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # create one minibatch of size batch_size
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            # compute stochastic gradient and loss function for the batch
            g = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y,tx,w)
        # update parameters vector
        w = w - gamma*g
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares using normal equations."""
    
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression using normal equations."""
    
    N = y.shape[0]
    D = tx.shape[1]
    a = tx.T @ tx + lambda_* 2 * N * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss
