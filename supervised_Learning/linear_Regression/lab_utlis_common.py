import math,copy
import numpy as np 
import matplotlib.pyplot as plt



# different routines for Linear Regression of multiple features input 

def compute_cost_matrix( X , y , w, b , verbose = False):
    """
    Computes the gradient for linear regression
     Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns
      cost: (scalar)
    """
    m,n = X.shape
    f_wb = X @ w + b
    total_cost = (1/(2*m)) * (np.sum((f_wb - y)**2))

    return total_cost

def compute_gradient_matrix(X , y , w, b):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      dj_dw (ndarray (n,1)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):        The gradient of the cost w.r.t. the parameter b.

    """
    # we know that dj_dw_j = (1/m)* ((sum ( f_wb - y)) * xj )
    m,n = X.shape

    f_wb = X @ w + b
    e = f_wb - y

    dj_dw = (1/m) * (X.T @ e)
    dj_db = (1/m) * np.sum(e)


def Z_score_normalize(X , rtn_ms = False):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X , axis = 0)

    X_norm = (X - mu)/sigma

    if(rtn_ms):
        return X_norm,mu , sigma
    else :
        return X_norm
    
def graident_descent( X ,y , w_in , b_in , alpha , num_iters, cost_funciton,gradient_function):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
    """
    m,n = X.shape 

    #make a dictionary of histories 
    #
    hist = {}   
    hist["cost "] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw , dj_db = gradient_function(X,y,w,b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        # we have update our w and b 
        if i% math.ceil(num_iters/10) ==0:
            hist["cost "].append(cost_funciton(X,y,w,b))
            hist["params"].append([w,b])
            hist["grads"].append([dj_dw,dj_db])
            hist["iter"].append(i)
            cst = cost_funciton(X,y,w,b)
            print(f"the cost for the following w and b : is : {cst:0.5e}")
    return w,b

def run_gradient_descent(X,y,interations = 10000,alpha = 1e-6):
    m,n = X.shape
    initial_w = np.zeros(n)
    initial_b =  0

    w_final, b_final = graident_descent(X,y,initial_w,initial_b,alpha=1e-6,num_iters=interations,cost_funciton= compute_cost_matrix,gradient_function= compute_gradient_matrix)
    return w_final,b_final


def compute_cost(X,y,w,b):
    """
      for one feature 
      X  : array like shape (1,)  : single data point 
      y : array like shape(1,)   : target value 
      w : the gradient coffecient 
      b : the intercept value
    """
    m = X.shape[0]
    f_wb = X * w  + b
    e = f_wb - y

    total_cost = (1/(2*m))* np.sum(e**2)
    return total_cost

def compute_gradient(X,y,w,b):
    """
      for one feature 
      X  : array like shape (1,)  : single data point 
      y : array like shape(1,)   : target value 
      w : the gradient coffecient 
      b : the intercept value
    """
    #  here we have the ith sample 
    m = X.shape[0]

    f_wb = X*w + b
    e = f_wb - y
    dj_db = (1/m) * np.sum(e)
    dj_dw = (1/m) * (X @ e)

    return dj_dw,dj_db

def gradient_desc(X,y,w_in,b_in,alpha , num_iters, cost_function,gradient_function):
    m = X.shape[0]

    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        
        dj_dw , dj_db = gradient_function(X,y,w,b)

        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w,b



        
      


