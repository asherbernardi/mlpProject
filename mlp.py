import numpy as np
from random import uniform

import perceptron

Perceptron = perceptron.Perceptron

# a class that represents one hidden layer in a neural network
class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
    def __str__(self):
        return " | ".join([str(p) for p in self.perceptrons])
    def initialize(num_perceptrons, num_weights, activation):
        return PerceptronLayer([Perceptron.initialize(num_weights, activation)
                                for i in range(num_perceptrons)])

# a class that represents a neural network
class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
    def __str__(self):
        ret = ""
        for i, l in enumerate(self.layers):
            ret += "Layer " + str(i) + ":\n" + str(l) + "\n"
        ret += "\n"
        return ret
    # The last layer (layers_sizes[-1]) will be the number of outputs of the mlp
    def initialize(layer_sizes, dimensionality, activation):
        # this holds an array that is used to determine how many weights should
        # be in each perceptron, since it's meant to be equal to the number of
        # perceptrons in the previous layer
        previous_layer_sizes = np.append(dimensionality, layer_sizes[:-1])
        return MultiLayerPerceptron([PerceptronLayer.initialize(num_perc, num_perc_prev + 1, activation)
                                     for num_perc, num_perc_prev in zip(layer_sizes, previous_layer_sizes)])
    def feed_forward(self, inputs):
        outputs = []
        for layer in self.layers:
            output = [p(inputs)
                      for p in layer.perceptrons]
            outputs.append(output)
            inputs = output
        return outputs
    def single_layer_backprop(self, inputs, targets, eta=0.1):
        assert(len(targets) == len(self.layers[1].perceptrons))
        outputs = self.feed_forward(inputs)
        delta_y = [y_out * (1 - y_out) * (t - y_out) for y_out, t in zip(outputs[1], targets)]
        delta_z = [z_out * (1 - z_out) *
                    np.dot(delta_y, [y.weights[k]
                    for y in self.layers[1].perceptrons])
                    for k, z_out in enumerate(outputs[0])]
        for l, y in enumerate(self.layers[1].perceptrons):
            y.weights = [weight + eta * delta_y[l] * outs for weight, outs in zip(y.weights, np.append([1],outputs[0]))]
        for k, z in enumerate(self.layers[0].perceptrons):
            z.weights = [weight + eta * delta_z[k] * ins for weight, ins in zip(z.weights, np.append([1],inputs))]

# train an mlp with only one hidden layer using backpropagation
def train(M, data, targets, termination = 10):
    mlp = MultiLayerPerceptron.initialize([M, 1], len(data[0]), perceptron.sigmoid)
    for i in range(termination):
        for d, t in zip(data, targets):
            mlp.single_layer_backprop(d, [t])
    return mlp

# feed forward for each input
def classify(mlp, inputs):
    return [np.round(mlp.feed_forward(i)[-1]) for i in inputs]
