# -*- coding: utf-8 -*-

import numpy as np


def calculate_mse(e):
   
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    
    e = y - (tx @ w)
    return calculate_mse(e)
   
