import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Sample input and output data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

# Random initialization of weights
weights0 = 2 * np.random.random((2, 4)) - 1
weights1 = 2 * np.random.random((4, 1)) - 1

# Learning speed parameter
learning_rate = 0.1

# The learning process
for j in range(60000):
    # forward pass
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))
    # Error in the output layer
    layer2_error = y - layer2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(layer2_error))))

    # backpropagation
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Weight update
    weights1 += learning_rate * layer1.T.dot(layer2_delta)
    weights0 += learning_rate * layer0.T.dot(layer1_delta)

print("After training:")
print(layer2)
