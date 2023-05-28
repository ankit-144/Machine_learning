# library function used in logistic regression 

import numpy as np



def output_vector(X,w,b):
    f_wb = X @w + b
    f_x = 1/(1 +np.exp(-f_wb))
    return f_x

def compute_logistic_cost(X, y , w ,b):
    """
        X : matrix of shape ( N , N)
        y : array of shape (N)
        w : array of shape (N) 
        b : array of shape (N)

    """
    f_x = output_vector(X,w,b)
    m,n = X.shape

    print(f"X is : {X} and the obtained value of f_x is  : {f_x}")
    cost = (-y)*np.log(f_x) - (1 - y) * np.log(1 - f_x)
    total_cost = np.sum(total_cost)/(m)
    return total_cost
    

