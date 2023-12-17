import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Dane wejściowe i wyjściowe dla problemu XOR
entrance = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Inicjalizacja wag i obciążeń
np.random.seed(1)
weights_hidden = np.random.uniform(-1, 1, (2, 2))
bias_hidden = np.random.uniform(-1, 1, (1, 2))
weights_output = np.random.uniform(-1, 1, (2, 1))
bias_output = np.random.uniform(-1, 1, (1, 1))


print("Wagi przed treningiem - warstwa ukryta: \n", weights_hidden)
print("Obciążenia przed treningiem - warstwa ukryta: \n", bias_hidden)
print("Wagi przed treningiem - warstwa wyjściowa: \n", weights_output)
print("Obciążenia przed treningiem - warstwa wyjściowa: \n", bias_output)


# Parametry uczenia
learning_rate = 0.1
n_epochs = 10000

for epoch in range(n_epochs):
    # Propagacja w przód
    entrance_input = np.dot(entrance, weights_hidden) + bias_hidden
    hidden_output = sigmoid(entrance_input)

    output_input = np.dot(hidden_output, weights_output) + bias_output
    predicted_output = sigmoid(output_input)

    # Obliczanie błędu i propagacja wsteczna
    error = expected_output - predicted_output
    delta_output = error * sigmoid_derivative(predicted_output)

    error_hidden = delta_output.dot(weights_output.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Aktualizacja wag i obciążeń
    weights_output += hidden_output.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

    weights_hidden += entrance.T.dot(delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate


print("\nWagi po treningu - warstwa ukryta: \n", weights_hidden)
print("Obciążenia po treningu - warstwa ukryta: \n", bias_hidden)
print("Wagi po treningu - warstwa wyjściowa: \n", weights_output)
print("Obciążenia po treningu - warstwa wyjściowa: \n", bias_output)

print("\nWyjście z sieci neuronowej: \n", end='')
print(predicted_output)