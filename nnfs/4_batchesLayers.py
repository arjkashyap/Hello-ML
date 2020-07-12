#!/usr/bin/python3.6
import numpy as np

#Features
X = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

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

layer_one = DenseLayer(4, 5)
layer_two = DenseLayer(5, 2)

layer_one.forward(X)
layer_two.forward(layer_one.output)
print(layer_two.output)
