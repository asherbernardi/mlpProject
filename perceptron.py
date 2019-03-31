# Adapted from code by Thomas Van Drunen (tvandrun)

import numpy as np

# The step function
def step(x) :
    return 0 if x < 0 else 1

# The "logistic" function, often called "sigmoid"
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

# We can adjust the sigmoid to make it range from -1 to 1
def sigmoid_adjusted(x) :
    return 2 / (1 + np.exp(-x)) -1

# A class that represents a single perceptron
class Perceptron :
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation
    def dimension(self) :
        return len(self.weights)-1
    def initialize(num_weights, activation):
        return Perceptron([uniform(-1,1) for i in range(num_weights)], activation)
    def __call__(self, inputs) :
        #print(np.append([1],inputs))
        return self.activation(np.dot(self.weights, np.append([1],inputs)))
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

from random import uniform
