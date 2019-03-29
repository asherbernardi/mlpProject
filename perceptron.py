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
        return [self.activation(j) for j in np.dot(self.weights, [1]+inputs)]
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

# A class that represents a single unthreasholded perceptron
class UnthreasholdedPerceptron :
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation
    def dimension(self) :
        return len(self.weights)-1
    def __call__(self, inputs) :
        return np.dot(self.weights, [1]+inputs)
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

from random import uniform

def initialize_perceptron(n, activation) :
    return Perceptron([uniform(-1,1) for n in range(n)], activation)

def init_unthreasholded_perceptron(n, activation) :
    return UnthreasholdedPerceptron([uniform(-1,1) for n in range(n)], activation)

def perc_train_step(p, x, t, eta=.1) :
    xx = [1] + x
    assert len(xx) == len(p.weights)
    z = p(x)
    p.weights = [p.weights[j] + eta*(t-z)*xx[j] for j in range(len(xx))]



line = np.linspace(-5, 5, 100)
import matplotlib.pyplot as plt

# The other activation functions are already in numpy

# Let's see the 0/1 activation functions

plt.plot(line, [step(x) for x in line], label="step")
plt.plot(line, [sigmoid(x) for x in line], label="sigmoid")
plt.legend(loc="best")
plt.show()

# Now let's see the -1/1 activation functions

plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.sign(line), label="sign")
plt.plot(line, [sigmoid_adjusted(x) for x in line], label="sigmoid")
plt.legend(loc="best")
plt.show()


and_gate = Perceptron([-1, 1, 1], np.tanh)

[and_gate([1,1]),and_gate([1,-1]),and_gate([-1,-1])]

or_gate = Perceptron([1, 1, 1], np.tanh)

[or_gate([1,1]), or_gate([1,-1]), or_gate([-1,-1])]





# train an and gate
data = [([1,1],1), ([1,-1],-1), ([-1,1],-1), ([-1,-1],-1)]
p = initialize_perceptron(3)
for i in range(10) :
    print("iteration " + str(i))
    print(str(p))
    print(",".join([str(p(d[0])) for d in data]))
    for d in data:
        perc_train_step(p, d[0], d[1])
