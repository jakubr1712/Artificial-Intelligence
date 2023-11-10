import random


def threshold(sum):
    if sum > 0:
        return 1
    else:
        return 0


def displayResults(we, weights, d, y, delta, count):
    print(f"Step: {count}")
    print(
        f"Input {we}, expected result: {d}, obtained result: {y}, delta: {delta}")
    print(f"Weights after update: {weights}\n")


def deltaRule():
    we = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    d = [0, 1, 1, 1]
    deltas = []
    wu = 0.3
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    w3 = random.uniform(-1, 1)
    weights = [w1, w2, w3]
    print(f"Starting weights: {weights}\n")
    count = 0

    while True:
        count += 1

        for i in range(len(we)):
            s = 0

            for j in range(len(weights)):
                s += (we[i][j] * weights[j])

            y = threshold(s)
            delta = d[i] - y
            deltas.append(delta)

            weights = [weights[0] + wu * delta*we[i][0], weights[1] +
                       wu * delta*we[i][1], weights[2] + wu * delta*we[i][2]]

            displayResults(we[i], weights, d[i], y, delta, count)

        all_zero = all(element == 0 for element in deltas)

        if all_zero:
            break
        else:
            deltas = []

    print(f"Steps: {count}")


deltaRule()
