import math
import numpy as np
from numba import cuda, float32






@cuda.jit
def neural_matrix_operation(input_weights, input_biases, added_biases, output):
    i, j = cuda.grid(2)
    if (i < output.shape[0]) & (j < output.shape[1]):
        val = 0.0
        for k in range(input_weights.shape[1]):
            val += input_weights[i,k] * input_biases[k,j]
        output[i,j] = math.tanh(val + added_biases[i,j])
        
        







