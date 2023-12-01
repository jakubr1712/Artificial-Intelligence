def train_hopfield(example):
    sizeExample = len(example)
    Wij = [[0 for _ in range(8)] for _ in range(8)]

    for k in range(sizeExample):
        row = example[k]
        sizeRow = len(row)
        for i in range(sizeRow):
            for j in range(sizeRow):
                if i != j:
                    Wij[i][j] += ((2 * row[i] - 1) * (2 * row[j] - 1))

    return Wij


def hopfield(Wij, testExample):
    sizeTestExample = len(testExample)
    result = [[0 for _ in range(8)] for _ in range(8)]

    for k in range(sizeTestExample):
        rowTest = testExample[k]

        row2 = []
        for i in range(len(Wij)):
            row2Item = 0
            rowWage = Wij[i]
            for j in range(len(rowWage)):
                row2Item += rowWage[j] * rowTest[j]

            if row2Item == 0:
                row2.append(rowTest[i])
            elif row2Item > 0:
                row2.append(1)
            elif row2Item < 0:
                row2.append(0)

        result[k] = row2

    return result


def inputExample():
    pattern = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        rowInput = input(
            f"Enter the line {i + 1} (a string of characters of length 8 consisting of either 0 or 1): ")
        pattern[i] = [int(bit) for bit in rowInput]
    return pattern


example = [[1, 1, 1, 0, 0, 1, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 1, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0, 0]]


testExample = inputExample()

Wij = train_hopfield(example)
result = hopfield(Wij, testExample)

print("\Recognized image:")
for row in result:
    print(" ".join(map(str, row)))

input("Press Enter to end the program.")
