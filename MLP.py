# Creating Multilayer Perceptron using random function

import numpy as np


class MLP(object):
    def __init__(self, num_inputs=2, hidden_layers=[3, 5],
                 num_outputs=2):  # defining number of inputs, outputs and hidden layers
        self.num_inputs = num_inputs
        self.num_hidden = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # Initiate Random Weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":
    # create MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagate(inputs)

    # print the results
    print("The network inputs are:", format(inputs))
    print("The network outputs are:", format(output))
