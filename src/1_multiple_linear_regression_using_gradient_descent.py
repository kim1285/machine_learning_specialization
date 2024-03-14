"""
1. import libraries
2. define X_train, y_train
3. define compute_cost function
4. define compute_gradient function
5. define gradient_descent function
6. initialize model parameters
7. initialize gradient descent settings
8. run gradient descent
9. print result
"""


import numpy as np
import math
import copy

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
w_in = np.zeros(X_train.shape[1])
b_in = 0.0


def compute_cost(X, y, w, b):
    """"
    X          : ndarray, 2-D array input variable
    y          : ndarray, 1-D array target
    w          : ndarray, 1-D array model weight
    b          : float,   model bias
    return     : float,   total_cost
    """
    m, n = np.shape(X)
    total_cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        total_cost += (f_wb_i - y[i]) ** 2
    total_cost /= 2 * m
    return total_cost


"""
testing compute_cost function

total_cost_test = compute_cost(X_train, y_train, w_in, b_in)
print(total_cost_test)
"""


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.0

    for i in range(m):
        err_i = (np.dot(X[i], w) + b - y[i])
        for j in range(n):
            dj_dw[j] = err_i * X[i, j]
        dj_db = err_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


"""
testing compute_gradient function

dj_dw_test, dj_db_test = compute_gradient(X_train, y_train, w_in, b_in)
print(dj_dw_test)
print(dj_db_test)
"""


def gradient_descent(X, y, initial_w, initial_b, cost_function, gradient_function, alpha,
                     num_iters):
    J_history = []
    w = copy.deepcopy(initial_w)
    b = initial_b

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d} : cost : {J_history[-1]:8.2f}")

    return w, b, J_history


"""
testing gradient_descent function

alpha_test = 1.0e-9
iterations_test = 10000
w_test, b_test, J_history_test = gradient_descent(X_train, y_train, w_in,
                                                  b_in, compute_cost,
                                                  compute_gradient,
                                                  alpha_test, iterations_test)
print(w_test, b_test)
"""

"""
initialize parameters, gradient descent settings, and run gradient descent
"""

m, n = X_train.shape
initial_w = np.zeros(n,)
initial_b = 0.0
alpha = 6.0e-6
num_iters = 1000000

w_final, b_final, J_history_final = gradient_descent(X_train, y_train, initial_w,
                                                     initial_b, compute_cost,
                                                     compute_gradient,
                                                     alpha, num_iters)

print(f"final w: {w_final}, final b: {b_final}")


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


y_pred = np.dot(X_train, w_final) + b_final
mse = mean_squared_error(y_train, y_pred)
print(f"Mean Squared Error (MSE) : {mse:.2f}")



