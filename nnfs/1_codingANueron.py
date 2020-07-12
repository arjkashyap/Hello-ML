#!/usr/bin/python3.6

"""
We will create the most basic form of a nueraon in raw non abstract method
Using 3 inputs
"""

"""
These are the value of inputs which are fed to the nueron
An input can come as an op from another layer or the first layer
"""
inputs = [1.2, 5.1, 2.1]

"""
 Every input has a unique weight associated with it. We cannot 
 tweak the input value but we can change the weights
"""
weights = [3.1, 2.1, 8.7]

"""
Bias is just like an intercept added in a linear equation. 
It is an additional parameter in the Neural Network which is used 
to adjust the output along with the weighted sum of the inputs to the neuron
"""
bias = 3.0

"""
Output formula from a single nueron is
op = sum(weights * inputs) + bias
i;e sum(w1*i1 + w2*i2 + w3*i3 .... wn*in) + bias 
This is also called activation function
"""

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
