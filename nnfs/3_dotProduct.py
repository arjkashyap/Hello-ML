#!/usr/bin/python3.6

import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, -.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ]

# we have 3 nuerons
biases = [2, 3, 0.5]
"""
Dot product simply multiplies the two vectors, or in this case
a vector and a matrix. The order of parameters is of prime importance here 
if you want to avoid weird shape erros. 
"""
output = np.dot(weights, inputs) + biases;

print(output)
