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
    
    if (missing_val=="avg"): 
        mean = np.mean(input_data[~missing_ind], 0)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, mean[i])
    elif (missing_val=="median"): 
        median = np.median(input_data[~missing_ind], 0)
        for i in incomplete_features:
            np.place(input_data[:,i], input_data[:,i] == -999, median[i])
    else:  
        yb = yb[~missing_ind]
        input_data = input_data[~missing_ind]
        ids = ids[~missing_ind]
        
    if normalized: 
        input_m = np.mean(input_data,0)
        input_std = np.std(input_data, 0)
        input_data = (input_data - input_m)/input_std
    
    return yb, input_data, ids

def pairwise_correlation(y, input_data): 
    """Compute pairwise correlation coefficient"""
    coef_vec = []
    for column in range(input_data.shape[1]): 
        coef_vec.append(np.corrcoef(y, input_data[:,column])[0,1])
    return np.abs(coef_vec)

def correlation_matrix(input_data, feature_list): 
    """Compute the correlation matrix"""
    feature_num = len(feature_list)
    corr_matrix = np.array([[0.1 for i in range(feature_num)] for j in range(feature_num)])
    for i in range(feature_num): 
        foo = feature_list[i]
        for j in range(feature_num): 
            bar = feature_list[j]
            corr_matrix[i][j]=np.corrcoef(input_data[:,foo], input_data[:,bar])[0,1]
       
    return np.abs(corr_matrix)

def feature_select(coef_vec, feature_threshold): 
    """Return a list of features with correlation coefficient above feature threshold"""
    return [i for i, x in enumerate(coef_vec) if x>feature_threshold]

def feature_extract(feature_list, corr_matrix, duplicate_threshold):
    """Extract from feature list features that are NOT highly correlated based on duplicate threshold"""
    # Filter out the indices of highly correlated features in corr_matrix 
    repeated_ind = np.where(corr_matrix>duplicate_threshold)
    dup_ind = np.unique(list(np.array(repeated_ind[1])[repeated_ind[0]-repeated_ind[1]!=0]))
    dup_features = [feature_list[i] for i in dup_ind]
    return [feature for feature in feature_list if feature not in dup_features]
    
def find_ridge_lambda(lambdas, y, tx, k_indices, k_fold, degree): 
    """Return the optimal lambda along with rmse_tr and rmse_te. Based on cross_validation using ridge regression"""
    rmse_tr = []
    rmse_te = []
    w_init = np.zeros(tx.shape[1])
        
    for l in lambdas:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te, w = cross_validation(y, tx, k_indices, k, l, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(np.sqrt(2*rmse_tr_tmp)))
        rmse_te.append(np.mean(np.sqrt(2*rmse_te_tmp)))
        
    opt_lambda = lambdas[np.argmin(rmse_te)]
    return opt_lambda, rmse_tr, rmse_te
      
    
def find_weight(y, tx, k_indices, opt_lambda, k_fold, degree): 
    """Return averaged weight using ridge regression"""
    w_init = np.zeros(tx.shape[1])
    rmse_tr_tmp = []
    
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation_ridge_helper(y, tx, k_indices, k, opt_lambda, degree)
        w = w_init + w
        rmse_tr_tmp.append(loss_tr)
    return w/k_fold, np.mean(np.sqrt(2*rmse_tr_tmp))
    
    
def compute_score(y_test, y_pred):
    if len(y_pred)== len(y_test):
        ones_aux = np.ones(len(y_pred))
        correct = np.sum(ones_aux[np.equal(y_test, y_pred)])
        incorrect = len(y_pred)-correct
        perc = correct / len(y_pred) * 100
        print("Total correct:", correct, "\nTotal incorrect:", incorrect, "\nCorrect percentage:", perc, "%")
    else:
        print("Data have different sizes.")


        
def compute_mse(y, tx, w):
    """Compute the loss using MSE."""
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    
    return mse

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_ridge_helper(y, tx, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    y_te = y[k_indices[k]]    
    tx_te = tx[k_indices[k]]
    
    y_tr = np.delete(y, k_indices[k], axis=0)
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    # ***************************************************
    w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    # ***************************************************
    loss_te = compute_mse(y_te, tx_te, w)
    loss_tr = compute_mse(y_tr, tx_tr, w)
    # ***************************************************
    return loss_tr, loss_te, w

def cross_validation_logistic_helper(y, tx, k_indices, k, max_it, gamma, degree):
    """return the loss of logistic regression."""
    # ***************************************************
    y_te = y[k_indices[k]]    
    tx_te = tx[k_indices[k]]
    
    y_tr = np.delete(y, k_indices[k], axis=0)
    tx_tr = np.delete(tx, k_indices[k], axis=0)
    w0 = np.zeros(tx_tr.shape[1])
    
    # ***************************************************
    w, loss_tr = logistic_regression(y_tr, tx_tr, w0, max_it, gamma)
    # ***************************************************
    loss_te = compute_mse(y_te, tx_te, w)
    loss_tr = compute_mse(y_tr, tx_tr, w)
    # ***************************************************
    return loss_tr, loss_te, w


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
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss

#================================
def logistic_sigmoid ( x ):
    large_number = 1e2
    small_number = 1e-15

    if np.abs(x) > large_number:
        x = np.sign(x) * large_number

    value = 1.0 / (1 + np.exp(-x))

    value = np.clip(value, small_number, 1 - small_number)
    return value


def l1_prox_operator(x, gamma, lambda_):
    return np.sign(x) * np.maximum(np.zeros(x.shape[0]), np.abs(x) - lambda_ * gamma)

def logistic_entropy_loss(y, tx, w):
    loss = 0

    for i in range(0, len(y)):
        x = tx[i, :]
        sigma = logistic_sigmoid(np.dot(x, w))
        loss += y[i] * np.log(sigma) + (1 - y[i]) * np.log(1 - sigma)

    return -loss

def reg_logistic_entropy_loss(y, tx, w, lambda_, reg='L2'):
    loss = logistic_entropy_loss(y, tx, w)
    if reg == 'L2':
        loss += lambda_ * np.linalg.norm(w, 2)
    elif reg == 'L1':
        loss += lambda_ * np.linalg.norm(w, 1)
    else:
        raise ValueError('Unknown regularisation method')

    return loss

def logistic_gradient(y, tx, w):
    r, c = tx.shape
    grad = np.zeros(c)

    for i in range(0, r):
        x = tx[i, :]
        sigma = logistic_sigmoid(np.dot(x, w))
        grad += (sigma - y[i]) * x.T

    return grad


def reg_logistic_gradient(y, tx, w, lambda_, reg='L2'):
    grad = logistic_gradient(y, tx, w)
    grad += 2 * lambda_ * w

    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for i in range(0, max_iters):
        gradient = logistic_gradient(y, tx,  w)
        loss = logistic_entropy_loss(y ,tx, w)

        w -= gamma * gradient

    return w, loss


def reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma, reg='L2'):
    w = initial_w

    for i in range(0, max_iters):
        if reg == 'L2':
            gradient = reg_logistic_gradient(y, tx, w, lambda_)
            loss = logistic_entropy_loss(y, tx, w)

            w -= gamma * gradient
        elif reg == 'L1':
            gradient = logistic_gradient(y, tx, w)
            loss = reg_logistic_entropy_loss(y, tx, w, lambda_, reg)
            gd_step = w - gamma * gradient
            w = l1_prox_operator(gd_step, gamma, lambda_)
        else:
            raise ValueError('Unknown regularization method')

    return w, loss
