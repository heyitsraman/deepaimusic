import math
def sigmoid(x):
    y = 1.0 / (1+ math.exp(-x))
    return y

def activate(inputs, weights):
    #  perform summation
        h=0
        for x,w in zip(inputs,weights):
            h+=x*w
    # perform activation
        return sigmoid(h)

inputs = [0.5, 0.3, 0.2]
weights = [0.4, 0.7, 0.2]
output = activate(inputs, weights)

print(output)
