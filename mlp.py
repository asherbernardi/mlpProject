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
        #print([str(self.layers[0].perceptrons[i]) for i in range(2)])
        # delta_y = [output * (1 - output) * (output - target) for output, target in zip(outputs[1], targets)]
        # delta_z = [h_output * (1 - h_output) * np.dot(delta_y, [n.weights[i] for n in self.layers[-1].perceptrons]) for i, h_output in enumerate(outputs[0])]
        delta_y = [y_out * (1 - y_out) * (t - y_out) for y_out, t in zip(outputs[1], targets)]
        delta_z = [z_out * (1 - z_out) *
                    np.dot(delta_y, [y.weights[k]
                    for y in self.layers[1].perceptrons])
                    for k, z_out in enumerate(outputs[0])]
        # for i, output in enumerate(self.layers[-1].perceptrons):
        #     for j, hidden_out in enumerate([1] + outputs[0]):
        #         output.weights[j] -= eta * delta_y[i] * hidden_out
        # for i, hidden in enumerate(self.layers[0].perceptrons):
        #     for j, input in enumerate([1] + inputs):
        #         hidden.weights[j] -= eta * delta_z[i] * input
        for l, y in enumerate(self.layers[1].perceptrons):
            y.weights = [weight + eta * delta_y[l] * outs for weight, outs in zip(y.weights, [1] + outputs[0])]
        for k, z in enumerate(self.layers[0].perceptrons):
            z.weights = [weight + eta * delta_z[k] * ins for weight, ins in zip(z.weights, [1] + inputs)]
        # print(self)

# p is an UnthreasholdedPerceptron
def stochastic_gradient_descent(data, targets, termination = 100,
                                activation = perceptron.sigmoid):
    p = init_unthreasholded_perceptron(activation)
    for i in range(termination):
        for x,t in zip(data, targets):
            perc_train_step(p, x, t)

# train an mlp with only one hidden layer using backpropagation
def train(M, data, targets, termination = 100):
    mlp = MultiLayerPerceptron.initialize([M, 1], len(data[0]), perceptron.sigmoid)
    for i in range(termination):
        for d, t in zip(data, targets):
            mlp.single_layer_backprop(d, [t])
    return mlp

# feed forward for each input
def classify(mlp, inputs):
    return [mlp.feed_forward(i)[-1] for i in inputs]

# test XOR
clsfyr = train(2, [[0,0],[0,1],[1,0],[1,1]], [0,1,1,0])
print(clsfyr)
results = classify(clsfyr, [[1,1],[0,0],[1,0],[0,1]])
print(results) # should be -1, -1, 1, 1

# xor_network = MultiLayerPerceptron([PerceptronLayer([Perceptron([-30,20,20],perceptron.sigmoid),Perceptron([-10,20,20],perceptron.sigmoid)]),PerceptronLayer([Perceptron([-30,-60,60],perceptron.sigmoid)])])

# for x in [0,1]:
#     for y in [0,1]:
#         print x, y, clsfyr.feed_forward([x,y])[-1]
