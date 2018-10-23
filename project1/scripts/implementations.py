import numpy as np
from proj1_preprocessing import * 
from proj1_feature_selection import * 
from proj1_utils import * 

    
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression using normal equations."""
    
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss

            
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
#         print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#               bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    #zt double check batch size 
    batch_size = 20
    
    for n_iter in range(max_iters):
        # create one minibatch of size batch_size
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute stochastic gradient and loss function for the batch
            g = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(y,tx,w)
        # update parameters vector
        w = w - gamma*g
        ws.append(w)
        losses.append(loss)
#         print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#               bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares using normal equations."""
    
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss


# ==== Logistic Regression and its variants ====
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
