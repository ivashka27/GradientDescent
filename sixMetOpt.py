from generatorRandomQuadraticFunctions import function
import random
import numpy as np




n = random.sample(range(2, 10), 10)
k = random.sample(range(1, 10), 10)

step = 0.01
startApp = 2
epsilon = 2e-4
chart = []

x = [1] # мжет пойти нахуй


for n in range(len(n)):
    for k in range(len(k)):

        col = 0

        mass = function(n, k)
        massNew = np.array(mass) * 2
        for k in range(len(mass)):
            mass[k + 1] = mass[k] - step * massNew[k]
            while abs(mass[k + 1] - mass[k]) >= epsilon:
                mass[k + 1] = mass[k] - step * massNew[k]

        # massNew = 2 * np.array(mass)

        # проверка на то что функция убывает - f(xk - ak * f'(xk)) < f(xk)
        while ():