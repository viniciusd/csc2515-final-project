# ###############################################
# This file is written for 
# "Who is Who: Reinforcement Learning vs Deep Neural 
# Network For Power Allocation in Wireless Networks"
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
import scipy.io as sio
import matplotlib.pyplot as plt

import wmmse

def evaluate(H, Py_p, NN_p, K, var_noise=1):
    """
    Evaluates the performance of the model using 
    the weighted sum-rate (WSR)
    
    Input
        H: channel
        p: pk 
        # Py_p are the Y labels in the test set
        var_noise: noise
        K: number of transceivers
    Output
        y: weighted sum-rate (WSR)
    """
    print("Evaluating model")
    #print("H -> ",H.shape,type(H))
    #print("Py_p -> ",Py_p.shape,type(Py_p))
    #print("NN_p -> ",NN_p.shape,type(NN_p))
    #print("K -> ",type(K))
    num_sample = H.shape[2]
    pyrate = np.zeros(num_sample)
    nnrate = np.zeros(num_sample)
    mprate = np.zeros(num_sample)
    rdrate = np.zeros(num_sample)
    for i in range(num_sample):
        pyrate[i] = wmmse.objective_sum_rate(H[:, :, i], Py_p[:, i], var_noise, K)
        nnrate[i] = wmmse.objective_sum_rate(H[:, :, i], NN_p[i, :], var_noise, K)
        mprate[i] = wmmse.objective_sum_rate(H[:, :, i], np.ones(K), var_noise, K)
        rdrate[i] = wmmse.objective_sum_rate(H[:, :, i], np.random.rand(K,1), var_noise, K)
    print('Sum-rate: WMMSE: %0.3f, DNN: %0.3f, Max Power: %0.3f, Random Power: %0.3f'%(sum(pyrate)/num_sample, sum(nnrate)/num_sample, sum(mprate)/num_sample, sum(rdrate)/num_sample))
    print('Ratio: DNN: %0.3f%%, Max Power: %0.3f%%, Random Power: %0.3f%%\n' % (sum(nnrate) / sum(pyrate)* 100, sum(mprate) / sum(pyrate) * 100, sum(rdrate) / sum(pyrate) * 100))

    plt.figure('%d'%K)
    plt.style.use('seaborn-deep')
    data = np.vstack([pyrate, nnrate]).T
    bins = np.linspace(0, max(pyrate), 50)
    plt.hist(data, bins, alpha=0.7, label=['WMMSE', 'DNN'])
    plt.legend(loc='upper right')
    plt.xlim([0, 8])
    plt.xlabel('sum-rate')
    plt.ylabel('number of samples')
    plt.savefig(f'./store_plots/histogram_{K}.eps')
    plt.show()
    return 0
