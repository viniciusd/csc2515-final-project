# ###############################################
# This file is written for 
# "Who is Who: Reinforcement Learning vs Deep Neural Network
#   For Power Allocation in Wireless Networks"
# IML Fall 2021 Project, University of Toronto
# version 1.0 -- December 2021.
# Based on "Learning to Optimize" [1] and [2],
# see main.py for more details.
# ==============================================
"""
Reviewed: Ok
Running: Ok
Refactor: In-Progress
Code comments: In-Progress
"""

import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import re

def objective_sum_rate(H, p, var_noise, K):
    #print("sum_rate:")
    #print("H", H.shape)
    #print("p", p.shape)
    #print("noise value:",var_noise)
    """
    Returns the weighted sum-rate (WSR)
    
    Input:
        H: channel (type: ndarray, shape: (10,10))
        p: pk (type: ndarray)
        var_noise: noise (type: int, default: 1)
        K: number of transceivers (type: int, default: 10)
    Output:
        y: weighted sum-rate (WSR), (type: float)
    """
    y = 0.0
    for i in range(K):
        s = var_noise
        for j in range(K):
            if j!=i: # Only for the interference
                s = s+H[i,j]**2*p[j]
        y = y+math.log2(1+H[i,i]**2*p[i]/s)
    return y

def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    """"
    WMMSE optimization algoritrhm [3].
    Returns the weighted sum-rate (WSR) based on the
    This function is used to compute the labels of the data.
    
    Input
        p_int: number of single-antenna transceivers pairs (type: ndarray, shape: (10,), default: 10)
        H: sample of the data (type: ndarray, shape: (10,10))
        Pmax: power budget of each transmitter (type: int, default:1)
        var_noise: noise (type: float, default:1.0)
    Output
        p_opt: weighted system throughput (type: ndarray, shape: (10,))
    """
    ########## t = 0
    K = np.size(p_int) # recreate number of transceivers?
    vnew = 0 # v_k_0
    b = np.sqrt(p_int) # maximum range of values for v_k_0
    f = np.zeros(K) # u_k_0
    w = np.zeros(K) # w_k_0
    for i in range(K): # if k = 10
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise) # u_k_0
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i]) # w_k_0
        vnew = vnew + math.log2(w[i])

    ########## t = t + 1
    VV = np.zeros(100) # Why 100?
    for iter in range(100):
        vold = vnew
        # v_k_t is updated using all the values computed in the last loop
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp # This is the output

        vnew = 0
        # u_k_t and w_j_t are computed
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        # v_k_t is stored
        VV[iter] = vnew
        # Convergence condition 
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt


def gab_generate_Gaussian(K, num_H, seed, distribution, var_noise):
    # original: def X_generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=15):
    # new: def generate_CSI(K, num_H, seed, distribution, var_noise):
    # All these values are given in the generate_data.py file
    # K = 10, default=10, type=int, number of single-antenna receivers pairs
    # num_H, default=num_train, type=str, number of samples of the training data
    # seed, from np.random.RandomState(seed)
    # distribution, default=distribution, can be Rayleigh, Rince, Geometry
    # var_noise, default=1.0, type=float
    #########

    rng = np.random.RandomState(seed)

    out = re.split(r'(\d+)', distribution)
    #name, num = (out[0], 1) if len(out) == 1 else (out[0], out[1])
    name = "Rayleigh"
    num = 20000
    print('Generate Data: %s, %s, K = %d, num = %d, seed = %d' %
          (name, num, K, num_H, seed))

    if name == "Rayleigh":
        abs_H = generate_rayleigh_CSI(K, num_H, rng, int(num))
    #elif name == "Rice":
    #    abs_H = generate_rice_CSI(K, num_H, rng)
    #elif name == "Geometry":
    #    abs_H = generate_geometry_CSI(K, num_H, rng, area_length=int(num))
    else:
        print("Invalid CSI distribution.")
        exit(0)

    Pmax = 1
    Pini = np.ones(K)
    # num_H is the number of samples, K is the number of single-antenna transceivers pairs
    Y = np.zeros((num_H, K)) # Y is an array of zeros to store the ouput of WMMSE, num_H is the # samples in set
    for loop in range(num_H): # loop is an index representing every sample (given in num_H)
        # abs_H[loop, :] selects from the distribution stored in abs_H the row number given by the loop index
        
        #print("OLD abs_H: "+str(abs_H.shape)+" , "+str(type(abs_H)))
        #print("OLD abs_H row: "+str(abs_H[loop,:]))
        H = np.reshape(abs_H[loop, :], (K, K)) # is a reshaped array with dimensions (# transceivers, # transceivers)
        # Parameters of WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        # 1. Pini is an array of ones depending on K
        #  K (default 10) is the number of single-antenna receivers pairs
        # 2. H is an array containing the distribution for the loop sample
        # 3. Pmax (default 1) is the power budget of each transmitter
        # 4. var_noise is the noise
        Y[loop, :] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
    print("generate data ouput ",name)
    print("abs_H: "+str(abs_H.shape)+" , "+str(type(abs_H)))
    print("p_opt: "+str(Y.shape)+" , "+str(type(Y)))
    return abs_H, Y

# Functions for data generation, Gaussian IC case
def generate_Gaussian(K, num_H, Pmax=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pini = Pmax*np.ones(K)
    var_noise = 1
    X=np.zeros((K**2,num_H)) # (100, 10) if K = 10
    Y=np.zeros((K,num_H)) # (10, 10) if K = 10
    print("Definition inside Gaussian:")
    print("X ->",X.shape)
    print("Y ->",Y.shape)
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = WMMSE_sum_rate(Pini, H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    # print("wmmse time: %0.2f s" % total_time)
    print("Output inside Gaussian:")
    print("X ->",X.shape)
    print("Y ->",Y.shape)
    return X, Y, total_time