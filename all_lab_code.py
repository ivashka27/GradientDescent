import numpy as np
import matplotlib.pyplot as plt
import random

# f(x, y) = ax^2 + by^2 + cxy + dx + ey + f
# f(x_1, x_2, ..., x_n) = c_1*x1^2 + c_2*x_2^2 + ... + c_n*x_n^2


def make_2_function(a, b, c, d, e, f):
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f}


def calculate_two_gradient(coordinates, coefficient):
    return np.array([
        2.0 * coefficient['a'] * coordinates[0] + coefficient['c'] * coordinates[1] + coefficient['d'],
        2.0 * coefficient['b'] * coordinates[1] + coefficient['c'] * coordinates[0] + coefficient['e']
    ])


def calculate_n_gradient(coordinates, coefficient):
    return np.array(2.0 * coordinates * coefficient)


def get_2_function_value(coordinates, function):
    return function['a'] * coordinates[0] ** 2 + \
        function['b'] * coordinates[1] ** 2 + \
        function['c'] * coordinates[0] * coordinates[1] + \
        function['d'] * coordinates[0] + \
        function['e'] * coordinates[1] + \
        function['f']


def get_n_function_value(coordinates, coefficient):
    return np.sum(np.dot(np.square(coordinates), coefficient))


def get_function_points_value(coordinates, function, get_value):
    res = []
    for i in range(len(coordinates)):
        res.append(get_value(coordinates, function))
    return np.array(res)


# dichotomy возвращает [шаг, количество вычислений функции]


def dichotomy(coordinates, grad, function, get_value):
    left = 0.0001
    right = 0.09
    eps = 0.00001
    cnt_function = 0
    while abs(right - left) > eps:
        a = (left * 2 + right) / 3
        b = (left + right * 2) / 3
        f_a = get_value(coordinates - a * grad, function)
        f_b = get_value(coordinates - b * grad, function)
        cnt_function += 2
        if f_a < f_b:
            right = b
        else:
            left = a
    return [(left + right) / 2, cnt_function]


# constant_learning_rate -- возвращает [шаг, количество вычислений функции(всегда 0)] -- костыль, времени исправлять нет


def constant_learning_rate(coordinates, grad, function, get_value):
    return [0.06, 0]


# gradient_descent возвращает [массив точек спуска, количество вычислений функции, количество вычислений градиента]


def gradient_descent(start_point, function, gradient, get_value, learning_rate, eps):
    condition = 1
    cnt_function = 0
    cnt_gradient = 0
    res = [start_point[:]]
    prev_point = start_point[:]

    while condition:
        grad = gradient(prev_point, function)
        cnt_gradient += 1
        alpha = learning_rate(prev_point, grad, function, get_value)
        cnt_function += alpha[1]
        new_point = prev_point - alpha[0] * grad
        res += [new_point]

        for i in range(len(prev_point)):
            if abs(prev_point[i] - new_point[i]) <= eps:
                condition = 0
        prev_point = new_point
    return [np.array(res), cnt_function, cnt_gradient]


# generate_function -- генератор случайных квадратичных функций


def generate_function(n, k):
    if k == 0:
        return np.array([0 for i in range(n)])

    a = random.uniform(k + 1, 100)
    b = a / k

    coefficients = [a, b]

    for i in range(2, n):
        coefficients.append(random.uniform(b + 1, a - 1))

    random.shuffle(coefficients)
    return np.array(coefficients)


def tnk():
    dimension = random.sample(range(1, 1000), 10)
    conditionality = random.sample(range(1, 1000), 10)
    e = 0.00000001
    for n in dimension:
        for k in conditionality:
            start_point = np.array([0 for i in range(n)])
            res = gradient_descent(start_point, generate_function(n, k), calculate_n_gradient,
                                   get_n_function_value, dichotomy, e)[:]
            print("N = ", n)
            print("K = ", k)
            print("function calls: ", res[1])
            print("gradient calls: ", res[2])
            print("-------------------------")


def wolfe(coordinates, grad, function, gradient, get_value, t, d, c1, c2):
    fx = get_value(coordinates, function)
    x_grad = np.dot(grad, d)
    f_new_x = get_value(coordinates + t * d, function)
    grad_new_x = np.dot(gradient(coordinates + t * d, function), d)
    function_calls = 2
    gradient_calls = 1
    while not (f_new_x <= fx + c1 * t * x_grad and grad_new_x <= -c2 * x_grad):
        function_calls += 1
        gradient_calls += 1
        t /= 2
        f_new_x = get_value(coordinates + t * d, function)
        grad_new_x = gradient(coordinates + t * d, function) * d
    return [t, function_calls, gradient_calls]


def wolfe_gradient(start_point, function, gradient, get_value, eps, c1, c2, sample_alpha):
    condition = 1
    cnt_function = 0
    cnt_gradient = 0
    res = [start_point[:]]
    prev_point = start_point[:]

    while condition:
        grad = gradient(prev_point, function)
        alpha = wolfe(prev_point, grad, function, gradient, get_value, sample_alpha, -grad, c1, c2)
        cnt_gradient += alpha[2] + 1
        cnt_function += alpha[1]
        new_point = prev_point - alpha[0] * grad
        res += [new_point]

        for i in range(len(prev_point)):
            if abs(prev_point[i] - new_point[i]) <= eps:
                condition = 0
        prev_point = new_point
    return [np.array(res), cnt_function, cnt_gradient]


f1 = make_2_function(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
f2 = make_2_function(2.0, 6.0, 0.0, 15.0, 2.0, 0.0)
epsilon = 0.0001

# res1 = gradient_descent([-5.0, -5.0], f1, calculate_two_gradient, get_2_function_value, constant_learning_rate, epsilon)
# res2 = gradient_descent([-5.0, -5.0], f1, calculate_two_gradient, get_2_function_value, dichotomy, epsilon)

res1 = wolfe_gradient([-5.0, -5.0], f1, calculate_two_gradient, get_2_function_value, epsilon, 0.0001, 0.09, 0.1)
print("Points:", res1[0])
print("function calls:", res1[1], ", gradient calls:", res1[2])
# print("Points:", res2[0])
# print("function calls:", res2[1], ", gradient calls:", res2[2])

f3 = [1, 1, 1]
#res3 = gradient_descent([5, 5, 5], f3, calculate_n_gradient, get_n_function_value, dichotomy, epsilon)
#print("Points:", res3[0])
#print("function calls:", res3[1], ", gradient calls:", res3[2])

tnk()
