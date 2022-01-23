"""
Reviewed: In-Progress
Running: In-Progress
Refactor: In-Progress
Code comments: In-Progress
"""
from numpy import genfromtxt

def load_data(distribution_type, K):
    X_train = genfromtxt(f"./datasets/{distribution_type}/{K}_users/X_train_{distribution_type}.csv", delimiter=',')
    X_train = X_train.T
    num_H = X_train.shape[1]
    #print(X_train.shape)

    Y_train = genfromtxt(f"./datasets/{distribution_type}/{K}_users/Y_train_{distribution_type}.csv", delimiter=',')
    Y_train = Y_train.T
    #print(Y_train.shape)

    X_test = genfromtxt(f"./datasets/{distribution_type}/{K}_users/X_test_{distribution_type}.csv", delimiter=',')
    X_test = X_test.T
    #print(X_test.shape)

    Y_test = genfromtxt(f"./datasets/{distribution_type}/{K}_users/Y_test_{distribution_type}.csv", delimiter=',')
    Y_test = Y_test.T
    #print(Y_test.shape)
    return X_train, Y_train, X_test, Y_test
