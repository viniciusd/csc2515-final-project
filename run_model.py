# ###############################################
# This file is written for 
# "Who is Who: Reinforcement Learning vs Deep Neural Network 
#   For Power Allocation in Wireless Networks"
# IML Fall 2021 Project, University of Toronto
# version 1.0 -- December 2021.
# ==============================================
# 
# Based on "Learning to Optimize" [1] and [2].
# Also includes functions to perform the WMMSE algorithm.
#
# References:
# [1] Haoran Sun, Wenqiang Pu, Minghe Zhu, Xiao Fu, Tsung-Hui Chang,
# Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In
# Episodically Dynamic Environment",
# arXiv preprint arXiv:2011.07782 (2020).
#
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong,
# Xiao Fu and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep
# Neural Networks for Wireless Resource Management”, IEEE Transactions on
# Signal Processing 66.20 (2018): 5438-5453.
#
# [3] Qingjiang Shi, Meisam Razaviyayn, Zhi-Quan Luo, and Chen He.
# "An iteratively weighted MMSE approach to distributed sum-utility
# maximization for a MIMO interfering broadcast channel."
# IEEE Transactions on Signal Processing 59.9 (2011): 4331-4340.
# ###############################################
import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import wmmse
import neuralNetwork as nn
import evaluation as ev
import distributions

def main():
    # Problem Setup
    K = 10                          # Number of users
    training_epochs = 100           # Number of training epochs
    distribution_type = "Rayleigh"  # Rayleigh or Rice

    """
    Load data
    """

    X_train, Y_train, X_test, Y_test = distributions.load_data(distribution_type, K)

    num_H = X_train.shape[1]
    print(X_train.shape)

    print("Distribution:", distribution_type)
    print('K=%d, Total Samples: %d, Total Iterations: %d\n'%( K, num_H, training_epochs))
    print('Training  Deep Neural Network ...')

    """
    Training
    """
    model_location = "./store_models/dnn_model.ckpt"
    nn.train(X_train, Y_train, model_location, training_epochs=training_epochs, traintestsplit = 0.2, batch_size=200)


    """
    Testing
    """
    nn.test(X_test, model_location, "./store_files/Prediction_%d" % K , K * K, K, binary=1)


    """
    Evaluate
    """
    H = np.reshape(X_test, (K, K, X_test.shape[1]), order="F")
    NNVbb = sio.loadmat('./Prediction_%d' % K)['pred']
    evaluate(H, Y_test, NNVbb, K)

    print("Plot figures")
    train_loss = sio.loadmat('./MSETime_%d_%d_%d'%(K, 200, 10))['train_loss']
    val_loss = sio.loadmat('./MSETime_%d_%d_%d'%(K, 200, 10))['validation_loss']
    fig = plt.figure(figsize=(7, 4))
    plt.subplot(1, 2, 1)
    plt.plot(val_loss.T,label='validation')
    plt.plot(train_loss.T,label='train')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')

    train_acc = sio.loadmat('./MSETime_%d_%d_%d'%(K, 200, 10))['train_acc']
    val_acc = sio.loadmat('./MSETime_%d_%d_%d'%(K, 200, 10))['validation_acc']
    plt.subplot(1, 2, 2)
    plt.plot(val_acc.T,label='validation')
    plt.plot(train_acc.T,label='train')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"DNN training for channel modelled as a {distribution_type} distribution for {K} users")
    plt.savefig(f'./store_plots/dnn_train_{distribution_type}_k{K}.eps')
    plt.show()

if __name__ == '__main__':
    main()
