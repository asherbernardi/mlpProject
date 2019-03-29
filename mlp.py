import numpy as np
from random import uniform

import perceptron

# a class that represents one hidden layer in a neural network
class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
    def initialize(num_perceptrons, num_weights):
        return PerceptronLayer([Perceptron.initialize(num_weights) for i in range(num_perceptrons)])

# a class that represents a neural network
class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
    def feed_forward(self, inputs):
        for layer in self.layers:
            for perceptron in layer.perceptrons

# p is an UnthreasholdedPerceptron
def stochastic_gradient_descent(data, targets, termination = 100, activation = perceptron.sigmoid):
    p = init_unthreasholded_perceptron(activation)
    for i in range(termination):
        for x,t in zip(data, targets):
            perc_train_step(p, x, t)

# Backpropagation with only one hidden layer
def train(M, data, targets, termination = 100):
    for k in range(M):
        for

# feed forward
def classify(mlp, inputs):
