# THIRD PARTY IMPORTS
import numpy as np
import math
from numba import cuda
from enum import Enum
import torch
from torch import nn

# FIRST PARTY IMPORTS
from Models.Neural_Nets.cuda_functions import *




class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        samples = input_data.shape[0]
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, batch_size, epochs, learning_rate):
        samples = x_train.shape[0]
        num_batches = np.floor(samples/batch_size)
        x_batches = np.array_split(x_train, num_batches)
        y_batches = np.array_split(y_train, num_batches)
        
        for e in range(epochs):
            for b in range(len(x_batches)):
                err = 0
                outputs = np.zeros( y_batches[b].shape )
                for s in range(x_batches[b].shape[0]):
                    #forward prop
                    output = x_batches[b][s]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    # outputs.append( output )
                    outputs[s] = output
                        
                #compute loss
                err += self.loss(y_batches[b], outputs)
                
                #backward prop
                error = self.loss_prime(y_batches[b], outputs)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                    
            err /= num_batches
            print('epoch %d/%d   error=%f' % (e+1, epochs, err))
        
        # for i in range(epochs):
        #     err = 0
        #     for j in range(samples):
        #         # forward propagation
        #         output = x_train[j]
        #         for layer in self.layers:
        #             output = layer.forward_propagation(output)

        #         # print(output)
        #         # compute loss (for display purpose only)
        #         # err += self.loss(y_train[j], output)
        #         err += self.loss(output, x_train[j-1], x_train[j])

        #         # backward propagation
        #         # error = self.loss_prime(y_train[j], output)
        #         error = self.loss_prime(output, x_train[j-1], x_train[j])
        #         for layer in reversed(self.layers):
        #             error = layer.backward_propagation(error, learning_rate)

        #     # calculate average error on all samples
        #     err /= samples
        #     print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        
    



















        








