import numpy as np
from random import uniform

import perceptron

Perceptron = perceptron.Perceptron

# a class that represents one hidden layer in a neural network
class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
    def initialize(num_perceptrons, num_weights, activation):
        return PerceptronLayer([Perceptron.initialize(num_weights, activation)
                                for i in range(num_perceptrons)])

# a class that represents a neural network
class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
    def initialize(layer_sizes, num_weights, activation):
        return MultiLayerPerceptron([PerceptronLayer.initialize(num_perceptrons,
                                     num_weights, activation) for num_perceptrons in layer_sizes])
    def feed_forward(self, inputs):
        outputs = []
        for layer in self.layers:
            output = [perceptron(inputs)
                      for perceptron in layer.perceptrons]
            outputs.append(output)
            inputs = output
        return outputs
    def single_layer_backprop(self, inputs, targets, eta=0.1):
        outputs = self.feed_forward(inputs)
        delta_y = [yl * (1 - yl) * (ti - yl) for yl,ti in zip(outputs[1], targets)]
        delta_z = [zk * (1 - zk) *
                    np.dot(delta_y, [self.layers[1].perceptrons[l].weights[k]
                    for l in range(len(self.layers[1].perceptrons))])
                    for k,zk in enumerate(outputs[0])]

# p is an UnthreasholdedPerceptron
def stochastic_gradient_descent(data, targets, termination = 100,
                                activation = perceptron.sigmoid):
    p = init_unthreasholded_perceptron(activation)
    for i in range(termination):
        for x,t in zip(data, targets):
            perc_train_step(p, x, t)

# Backpropagation with only one hidden layer
def train(M, data, targets, termination = 100):
    for k in range(M):
        pass

# feed forward
def classify(mlp, inputs):
    return mlp.feed_forward(inputs)[-1]

# xor_network = MultiLayerPerceptron([PerceptronLayer([Perceptron([-30,20,20],perceptron.sigmoid),Perceptron([-10,20,20],perceptron.sigmoid)]),PerceptronLayer([Perceptron([-30,-60,60],perceptron.sigmoid)])])

# for x in [0,1]:
#     for y in [0,1]:
#         print x, y, xor_network.feed_forward([x,y])[-1]
