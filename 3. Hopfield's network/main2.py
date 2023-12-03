import numpy as np


def train_hopfield(example):
    size = len(example) * len(example[0])
    flat_example = np.array(example).flatten()

    Wij = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            if i != j:
                Wij[i][j] = (2 * flat_example[i] - 1) * \
                    (2 * flat_example[j] - 1)

    return Wij


def hopfield(Wij, testExample):
    size = len(testExample) * len(testExample[0])
    result = np.zeros((len(testExample), len(testExample[0])), dtype=int)

    flat_test = np.array(testExample).flatten()
    updated_flat_test = np.zeros(size, dtype=int)

    for i in range(size):
        weighted_sum = np.sum(Wij[i] * flat_test)
        if weighted_sum > 0:
            updated_flat_test[i] = 1
        elif weighted_sum < 0:
            updated_flat_test[i] = 0
        else:
            updated_flat_test[i] = flat_test[i]

    for i in range(len(testExample)):
        for j in range(len(testExample[0])):
            result[i][j] = updated_flat_test[i * len(testExample[0]) + j]

    return result


def inputExample():
    pattern = np.zeros((8, 8), dtype=int)
    for i in range(8):
        rowInput = input(
            f"Enter the line {i + 1} (a string of characters of length 8 consisting of either 0 or 1): ")
        pattern[i] = [int(bit) for bit in rowInput]
    return pattern


example = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

# testExample = inputExample()
testExample = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1]])

Wij = train_hopfield(example)
result = hopfield(Wij, testExample)

print("\nRecognized image:")
for row in result:
    print(" ".join(map(str, row)))
