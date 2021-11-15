# Creating Artificial Neuron with Sigmoid Activation Function

import math
def sigmoid(x):         #defining activation function Sigmoid    
    y = 1.0 / (1+ math.exp(-x))
    return y

def activate(inputs, weights):      #defining neuron calculations
    #  perform summation
        h=0
        for x,w in zip(inputs,weights):
            h+=x*w
    # perform activation
        return sigmoid(h)

inputs = [0.32, 0.3, 0.2]
weights = [0.4, 0.45, 0.2]
output = activate(inputs, weights)

print(output)       #artificial neuron generated
