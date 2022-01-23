# ###############################################
# This file is written for 
# "Who is Who: Reinforcement Learning vs Deep Neural 
# Network For Power Allocation in Wireless Networks"
# IML Fall 2021 Project, University of Toronto
# version 1.0 -- December 2021.
# Based on "Learning to Optimize"
# ==============================================
import numpy as np
import scipy.io as sio
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Functions for deep neural network structure construction
def multilayer_perceptron(input_keep_prob,hidden_keep_prob,n_hidden_1,n_hidden_2, n_hidden_3, n_input, n_output):
    output_activation = lambda x: keras.activations.relu(x, max_value=6)/6

    model = keras.models.Sequential([

    layers.InputLayer(input_shape=(n_input,)),
    layers.Dropout(1 - (input_keep_prob)),                        # dropout layer
    
    tf.keras.layers.Dense(n_hidden_1, activation='gelu'),
    layers.Dropout(1 - (hidden_keep_prob)),            # dropout layer

    tf.keras.layers.Dense(n_hidden_2, activation='gelu'),
    layers.Dropout(1 - (hidden_keep_prob)),

    tf.keras.layers.Dense(n_hidden_3, activation='gelu'),
    layers.Dropout(1 - (hidden_keep_prob)),

    tf.keras.layers.Dense(n_output, activation=output_activation),
    ])

    return model


# Functions for deep neural network training
def train(X, Y,location, training_epochs=300, batch_size=1000, LR= 0.001, n_hidden_1 = 200,n_hidden_2 = 80,n_hidden_3 = 80, traintestsplit = 0.01, LRdecay=0):
    n_input = X.shape[0]                          # input size
    n_output = Y.shape[0]                         # output size

    X, Y, = np.transpose(X), np.transpose(Y)
    print("X ->", X.shape)
    print("Y ->", Y.shape)

    input_keep_prob = tf.constant(1, dtype=tf.float32)
    hidden_keep_prob = tf.constant(1, dtype=tf.float32)
    cost = keras.losses.MeanSquaredError()    # cost function: MSE
    start_time = time.time()
    optimizer = keras.optimizers.Adam(learning_rate=LR) # training algorithms: Adam
    model = multilayer_perceptron(input_keep_prob, hidden_keep_prob,n_hidden_1, n_hidden_2,n_hidden_3, n_input, n_output)
    model.compile(optimizer=optimizer,
              loss=cost,
              metrics=['accuracy', cost])
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=location,
                                                 save_weights_only=True,
                                                 verbose=1)
    history = model.fit(X, Y,
              batch_size=batch_size,
              validation_split = traintestsplit,
              callbacks=[cp_callback],
              epochs=training_epochs)

    sio.savemat('./MSETime_%d_%d_%d' % (n_output, batch_size, LR*10000) ,
                {'train_acc': history.history["accuracy"], 'validation_acc': history.history["val_accuracy"],
                 'train_loss': history.history["loss"], 'validation_loss': history.history["val_loss"]
                 })


    print("training time: %0.2f s" % (time.time() - start_time))
    return model

# Functions for deep neural network testing
def test(X, model_location, save_name, n_input, n_output, n_hidden_1 = 200, n_hidden_2 = 80, n_hidden_3 = 80, binary=0):    
    input_keep_prob = tf.constant(1, dtype=tf.float32)
    hidden_keep_prob = tf.constant(1, dtype=tf.float32)
    model = multilayer_perceptron(input_keep_prob, hidden_keep_prob,n_hidden_1, n_hidden_2,n_hidden_3, n_input, n_output)
    #loss, acc = model.evaluate(X, Y, verbose=2)
    #print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    model.load_weights(model_location)
    start_time = time.time()
    # loss, acc = model.evaluate(X, Y, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    y_pred = model.predict(np.transpose(X))
    testtime = time.time() - start_time
    if binary==1:
      y_pred[y_pred >= 0.5] = 1
      y_pred[y_pred < 0.5] = 0
    sio.savemat(save_name, {'pred': y_pred})
    
    return testtime
