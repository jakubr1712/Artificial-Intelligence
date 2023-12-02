def train_hopfield(example):
    size = len(example) * len(example[0])
    Wij = [[0 for _ in range(size)] for _ in range(size)]
    flat_example = [bit for row in example for bit in row]

    for i in range(size):
        for j in range(size):
            if i != j:
                Wij[i][j] = (2 * flat_example[i] - 1) * \
                    (2 * flat_example[j] - 1)

    return Wij


def hopfield(Wij, testExample):
    size = len(testExample) * len(testExample[0])
    result = [[0 for _ in range(len(testExample[0]))]
              for _ in range(len(testExample))]

    flat_test = [bit for row in testExample for bit in row]
    updated_flat_test = [0 for _ in range(size)]

    for i in range(size):
        weighted_sum = sum(Wij[i][j] * flat_test[j] for j in range(size))
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
    pattern = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        rowInput = input(
            f"Enter the line {i + 1} (a string of characters of length 8 consisting of either 0 or 1): ")
        pattern[i] = [int(bit) for bit in rowInput]
    return pattern


example = [[1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]]


# testExample = inputExample()
testExample = [[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]]

Wij = train_hopfield(example)
result = hopfield(Wij, testExample)

print("\Recognized image:")
for row in result:
    print(" ".join(map(str, row)))
