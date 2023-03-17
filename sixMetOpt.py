import random
import numpy as np


def function(n, k):

    if k == 0:
        return [0]*n

    a = random.uniform(k + 1, 100)
    b = a / k

    mass = list()
    mass.append(a)
    mass.append(b)

    for i in range(2, n):
         mass.append(random.uniform(b + 1, a - 1))

    random.shuffle(mass)
    return mass

dimension = random.sample(range(2, 13), 10)
conditionality = random.sample(range(1, 13), 10)

step = 0.01
startApp = 2
epsilon = 2e-4
chart = []

print(dimension)
print(conditionality)

test = 0

for n in range(len(dimension)):
    for i in range(len(conditionality)):
        test += 1

        mass = function(dimension[n], conditionality[i])
        massNew = np.array(mass) * 2

        for j in range(len(mass)):
            print("прошел")
            col = 1
            mass[j + 1] = mass[j] - step * massNew[j]

            while abs(mass[j + 1] - mass[j]) > epsilon:

                mass[j + 1] = mass[j] - col * step * massNew[j]
                col += 1
            chart.append(col)

        # проверка на то что функция убывает - f(xk - ak * f'(xk)) < f(xk)
print(test)
print(chart)