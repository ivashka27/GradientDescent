import time

import numpy as np
import matplotlib.pyplot as plt
import psutil as psutil

start = time.time()
step = 0.25
countPoint = 30
eps = 1e-4
max_iter = 1000
w_start = np.array([0, 0, 0, 0])


# start function
def start_func(x):
    return 1 + (x - 5) * 5 + 1 * (x - 5) ** 2 + 4 * (x - 5) ** 3


# function
def func(p):
    return lambda x: p[0] + (x - 5) * p[1] + p[2] * (x - 5) ** 2 + p[3] * (x - 5) ** 3


def grad_calculator(x, func, dim):
    h = 1e-5
    res = []
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)

def grad_func(f, dim):
    return lambda x: grad_calculator(x, f, dim)


expected = [1, 5, 1, 4] # начальные параметры нашей функции



#generaet many points
def generate_sample(total):
    x = 0
    while x < total * step:
        yield start_func(x) + np.random.uniform(-1, 1) * np.random.uniform(2, 8)
        x += step


# sets points
X = np.arange(0, countPoint * step, step)
Y = np.array([round(y, 2) for y in generate_sample(countPoint)])


# генерация функций ошибок
def mse_func(f, X, y):
    def mse(w):
        diff = 0
        for i in range(len(X)):
            diff += (f(w)(X[i]) - y[i]) ** 2
        return diff
    return mse


f = mse_func(func, X, Y)
grad = grad_func(f, 4)


def wolfe_conditions(f, fx, grad, grx, x, d, t, c1=1e-4, c2=0.9):
    grxd = np.dot(grx, d)
    ft = f(x + t * d)
    gxt = np.dot(grad(x + t * d), d)
    armijo = ft <= fx + c1 * t * grxd
    curvature = gxt <= - c2 * grxd
    return armijo and curvature


def l_bfgs(f, gradF, start, e, max_iter, m = 10):
    point = [start]
    grad_calc = 1
    func_calc = 0
    gx = gradF(start)
    I = np.eye(len(start))
    Hk = I

    #new
    s_list = []
    y_list = []

    i = 0
    while i < max_iter:
        pk = -np.dot(Hk, gx)
        lr = 1

        while not wolfe_conditions(f, f(start), gradF, gx, start, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2
        xn = start + lr * pk

        start = xn
        gradn = gradF(xn)

        s = xn - start
        y = gradn - gx

        s_list.append(s)
        y_list.append(y)

        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)

        gx = gradn
        ro_list = [1.0 / (np.dot(y_i, s_i) + e) for y_i, s_i in zip(y_list, s_list)]

        for ro_i, s_i, y_i in zip(ro_list, s_list, y_list):
            Hk = np.dot(I - ro_i * s_i[:, np.newaxis] * y_i[np.newaxis, :],
                        np.dot(Hk, I - ro_i * y_i[:, np.newaxis] * s_i[np.newaxis, :])) + (ro_i * s_i[:, np.newaxis] * s_i[np.newaxis, :])
        point = start
        if np.linalg.norm(gx) < e:
            break
        i += 1
    return np.asarray(point), grad_calc, func_calc

ans = l_bfgs(f, grad, w_start, eps, max_iter)

print(ans)
x = np.linspace(0, 10, 50)


# Вычисление значений y
y = start_func(x)


x2 = np.linspace(0, 10, 50)
y2 = func(ans[0])(x2)
print(psutil.virtual_memory())
print("Используемое количество итераций:", ans[1])
print("Использованная память в байтах:", psutil.virtual_memory().used)
print("Использованная память в %:", psutil.virtual_memory().percent)
end = time.time()
print("Время в сеундах ", end - start)
time.time()
plt.plot(x, y)
plt.plot(x2, y2)
plt.xlabel('x')
plt.ylabel('y')
# plt.ylim(-20, 20)
plt.grid(True)
plt.show()