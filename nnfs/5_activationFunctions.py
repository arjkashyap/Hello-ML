#!/usr/bin/python3.6
import numpy as np
import nnfs
from nnfs.datasets import spiral_data


"""
In this file, we will try to summarise the activation functions.
The function used here is relu.
relu outputs:
        0 if x < 0
        x if x >= 0
"""

nnfs.init()

# This gives us x and y co-ordinates as features
# class to which the feature belongs is output y
X, y = spiral_data(100, 3)


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize a random numpy array with matching dimmensions
        Here the dimmensions must be paid attention for avoiding shape errors
        Matrix multipication rules to be followed
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Function responsible for passing the features or the outputs from another
        layer through the layer.
        """
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationRelu:
    def forward(self, inputs):
        """
        Returns the array where input: if smaller than 0 has been replaced
        by 0. Which is exactly what a relu does
        """
        self.output = np.maximum(0, inputs)

# The layer in this case takes two inputs namely the x and y co-ordinates
layer_one = DenseLayer(2, 5)
act1= ActivationRelu()
layer_one.forward(X)


act1.forward(layer_one.output)

print(act1.output)
