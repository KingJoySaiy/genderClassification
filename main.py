import readFile
import numpy as np


readFile.getTrainData()







'''
# given X[n_x][m], Y[m][1], calculate W[n_x][1], b
# z = W^T X + b = W[1][n_x]*X[n_x][m] + b[1][m]
# a = y hat = sigma(z)
# J = average{-ylog(a) - (1-y)log(1-a)}

# n_x -> dimension, m -> count of test examples
n_x = 10
m = 10


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# given m-examples X-Y & updating step, update W & b, make J(W, b) smaller
# X[n_x][m], Y[m][1], step, w[n_x][1], b
# w & b need to initialize
def work(X, Y, step, w, b):

    # Z.shape == A.shape == dZ.shape == (1, m)
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    # J = -1.0 / m * (np.dot(Y, np.log(A)) + np.dot((1 - Y), np.log(1 - A)))

    # dJ/dw = average of (a - y) * x
    # dJ/db = average of (a - y)
    dZ = A - Y
    dw = 1.0 / m * np.dot(X, dZ.T)
    db = 1.0 / m * np.sum(dZ)

    # update w & b, using Gradient Descent
    w -= step * dw
    b -= step * db
'''
