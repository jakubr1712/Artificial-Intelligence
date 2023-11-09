import math
import random


def threshold(sum):
    if sum > 0:
        return 1
    else:
        return 0


def sigmoid(sum):
    return 1 / (1 + math.exp(-sum))


def nonlinear_neuron(x1, x2, x3, choice):
    random_number1 = random.uniform(-1, 1)
    random_number2 = random.uniform(-1, 1)
    random_number3 = random.uniform(-1, 1)

    sum = random_number1*x1+random_number2*x2+random_number3*x3

    if choice == '1':
        return threshold(sum)
    elif choice == '2':
        return sigmoid(sum)


x1 = float(input("Enter a value 1: "))
x2 = float(input("Enter a value 2: "))
x3 = float(input("Enter a value 3: "))

choice = input(
    "Select the activation function: \n 1 - threshold \n 2 - sigmoidal \n")

result = nonlinear_neuron(x1, x2, x3, choice)

print(f"Result: {result}")
