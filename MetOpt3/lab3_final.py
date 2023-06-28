import math

import numpy as np
from matplotlib import pyplot as plt
import psutil as psutil
import time


def generate_func(x):
    return math.sin(x) + 0.5 * x ** 1.2


def generate_sample(start, count, step):
    x = start
    i = 0
    while i < count:
        i += 1
        yield generate_func(x) + np.random.uniform(-0.4, 0.4) * np.random.uniform(0, 0.8)
        x += step


count = 100
step = 0.1
start = 0.2
X = np.arange(start, count * step + start, step)
Y = np.array([round(y, 2) for y in generate_sample(start, count, step)])


def loss(X, Y, w):
    # loss = 0
    # for i in range(len(Y)):
    #     loss += get_vector(X[i], Y[i], w) ** 2
    # return loss
    return w[0] ** 2 - w[0] * w[1] + w[1] ** 2 + 9 * w[0] - 6 * w[1] + 20


def get_vector(x, y, w):
    return y - regression(x, w)


def get_vectors(X, Y, w):
    vectors = np.zeros(len(Y))
    for i in range(len(Y)):
        vectors[i] = get_vector(X[i], Y[i], w)
    return vectors


def regression(x, w):
    return math.sin(x) + w[0] * x ** w[1]
    # return w[0] * math.sin(x) + w[1] * x ** w[2]
    # return w[0] * x / (w[1] + x)
    # return (1 - w[0]) ** 2 + 100 * (w[1] - x[0] ** 2) ** 2


def get_grad_for_vector(x, y, w):
    return np.array([(-math.sin(x)),
                     (-x ** w[2]),
                     (-w[1] * np.log(x) * x ** w[2])])
    # return np.array([(-1), (-x)])
    # return np.array([-2 * (y - regression(x, w)), 2 * (y - regression(x, w)) * x])
    # return np.array((-x / (w[1] + x),
    #               (w[0] * x) / ((w[1] + x) ** 2)))


def jacobian_gauss_newton(X, Y, w):
    jacobian = np.zeros((len(Y), len(w)))
    for i in range(len(Y)):
        jacobian[i] = get_grad_for_vector(X[i], Y[i], w)
    return jacobian


def gauss_newton(X, Y, w_start, learning_rate, max_iters):
    start = time.time()
    iters = 0
    jacobi_calc = 0
    w = w_start
    points = []
    while iters < max_iters:
        j = jacobian_gauss_newton(X, Y, w)
        jacobi_calc += 1
        pseudo_j = np.linalg.pinv(j)
        jj = np.dot(pseudo_j, get_vectors(X, Y, w))
        w = w - learning_rate * jj
        points.append(w)
        iters += 1
        print("Error:", loss(X, Y, w))
        print("Weights:", w)
        if np.linalg.norm(jj) < 10 ** -3:
            break
    end = time.time()
    print("gauss_newton")
    print("Врямя запуска: ", end - start)
    print("Использованная память в процентах: ", psutil.virtual_memory().used)
    print("Использованно памяти в процентах: ", psutil.virtual_memory().percent)
    return [w, iters, jacobi_calc, points]


def jacobian_powell_dog_leg(X, Y, w):
    jacobian = np.zeros((len(w)))
    for i in range(len(X)):
        jacobian[0] += 2 * X[i] ** w[1] * (w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
        jacobian[1] += 2 * w[0] * X[i] ** w[1] * np.log(X[i]) * (w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
    return jacobian


def hessian_powell_dog_leg(X, Y, w):
    hessian = np.zeros((len(w), len(w)))
    for i in range(len(X)):
        hessian[0][0] += 2 * X[i] ** (2 * w[1])
        hessian[0][1] += 2 * X[i] ** w[1] * np.log(X[i]) * (2 * w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
        hessian[1][0] += 2 * X[i] ** w[1] * np.log(X[i]) * (2 * w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
        hessian[1][1] += 2 * w[0] * X[i] ** w[1] * np.log(X[i]) ** 2 * (2 * w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
    return hessian


def powell_dog_leg(H_inv, J, H, trust_radius):
    deltaB = -np.dot(H_inv, J)
    norm_deltaB = np.linalg.norm(deltaB)

    if norm_deltaB <= trust_radius:
        return deltaB

    deltaU = - (np.dot(J, J) / np.dot(J, np.dot(H, J))) * J
    norm_deltaU = np.linalg.norm(deltaU)

    if norm_deltaU >= trust_radius:
        return trust_radius * deltaU / norm_deltaU

    diff_B_U = deltaB - deltaU
    diff_square_B_U = np.dot(diff_B_U, diff_B_U)
    dot_U_diff_B_U = np.dot(deltaU, diff_B_U)

    fact = dot_U_diff_B_U ** 2 - diff_square_B_U * (np.dot(deltaU, deltaU) - trust_radius ** 2)
    tau = (-dot_U_diff_B_U + math.sqrt(fact)) / diff_square_B_U
    return deltaU + tau * diff_B_U


def trust_region_powell_dog_leg(X, Y, w_start, max_iters):
    start = time.time()
    eta = 0.2
    eps = 1e-3
    start_trust_radius = 1
    max_trust_radius = 10

    w = w_start
    trust_radius = start_trust_radius

    iters = 0
    jacobian_calc = 0
    hessian_calc = 0
    points = []
    while iters < max_iters:
        J = jacobian_powell_dog_leg(X, Y, w)
        jacobian_calc += 1

        H = hessian_powell_dog_leg(X, Y, w)
        hessian_calc += 1

        H_inv = np.linalg.inv(H)
        delta = powell_dog_leg(H_inv, J, H, trust_radius)

        actual_reduction = loss(X, Y, w) - loss(X, Y, w + delta)

        predicted_reduction = -(np.dot(J, delta) + 0.5 * np.dot(delta, np.dot(H, delta)))
        if predicted_reduction == 0.0:
            ratio = 1e11
        else:
            ratio = actual_reduction / predicted_reduction

        delta_norm = np.linalg.norm(delta)
        if ratio < 0.2:
            trust_radius = 0.2 * delta_norm
        else:
            if ratio > 0.8 and delta_norm == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if ratio > eta:
            w = w + delta
        else:
            w = w
            points.append(w)

        if np.linalg.norm(J) < eps:
            break

        if iters >= max_iters:
            break

        iters += 1
    end = time.time()
    print("powell_dog_leg")
    print("Врямя запуска: ", end - start)
    print("Использованная память в процентах: ", psutil.virtual_memory().used)
    print("Использованно памяти в процентах: ", psutil.virtual_memory().percent)
    return [w, iters, jacobian_calc, hessian_calc, points]


def gradient_loss(X, Y, w):
    # grad = np.zeros((len(w)))
    # for i in range(len(X)):
    #     grad[0] += 2 * X[i] ** w[1] * (w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
    #     grad[1] += 2 * w[0] * X[i] ** w[1] * np.log(X[i]) * (w[0] * X[i] ** w[1] + math.sin(X[i]) - Y[i])
    # return grad
    return np.array([2 * w[0] - w[1] + 9, -w[0] + 2 * w[1] - 6])


def wolfe_conditions(X, Y, w, gradient, lr):
    c1, c2 = 1e-4, 0.9
    if loss(X, Y, w - lr * gradient) > (loss(X, Y, w) - c1 * lr * np.dot(gradient, gradient)):
        return False
    if np.dot(gradient, gradient_loss(X, Y, w - lr * gradient)) > c2 * np.dot(gradient, gradient):
        return False
    return True


def bfgs(X, Y, w_start, eps, max_iters):
    start = time.time()
    grad_calc = 0
    func_calc = 0

    w = w_start
    grad = gradient_loss(X, Y, w)

    I = np.eye(len(w))
    H = I
    iters = 0

    while iters < max_iters and np.linalg.norm(grad) >= eps:
        p = -np.dot(H, grad)
        lr = 1
        while not wolfe_conditions(X, Y, w, grad, lr):
            lr = lr / 2

        w_next = w + lr * p
        s = w_next - w
        w = w_next

        grad_next = gradient_loss(X, Y, w_next)
        y = grad_next - grad
        grad = grad_next

        ro = 1.0 / (np.dot(y, s) + eps)

        H = np.dot(I - ro * s[:, np.newaxis] * y[np.newaxis, :],
                   np.dot(H, I - ro * y[:, np.newaxis] * s[np.newaxis, :])) + \
            (ro * s[:, np.newaxis] * s[np.newaxis, :])

        iters += 1
    end = time.time()
    print("bfgs")
    print("Врямя запуска: ", end - start)
    print("Использованная память в процентах: ", psutil.virtual_memory().used)
    print("Использованно памяти в процентах: ", psutil.virtual_memory().percent)
    return [list(w), iters, grad_calc, func_calc]


def l_bfgs(X, Y, w_start, eps, max_iter, m=10):
    start = time.time()
    grad_calc = 1
    func_calc = 0

    w = w_start
    grad = gradient_loss(X, Y, w_start)

    I = np.eye(len(w_start))
    H = I
    s_list = []
    y_list = []

    iters = 0
    while iters < max_iter and np.linalg.norm(grad) > eps:
        pk = -np.dot(H, grad)
        lr = 1
        while not wolfe_conditions(X, Y, w, grad, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2

        w_next = w + lr * pk
        s = w_next - w
        w = w_next

        grad_next = gradient_loss(X,Y,w_next)
        y = grad_next - grad
        grad = grad_next

        s_list.append(s)
        y_list.append(y)

        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)

        ro_list = [1.0 / (np.dot(y_i, s_i) + eps) for y_i, s_i in zip(y_list, s_list)]

        for ro_i, s_i, y_i in zip(ro_list, s_list, y_list):
            H = np.dot(I - ro_i * s_i[:, np.newaxis] * y_i[np.newaxis, :],
                        np.dot(H, I - ro_i * y_i[:, np.newaxis] * s_i[np.newaxis, :])) + (
                             ro_i * s_i[:, np.newaxis] * s_i[np.newaxis, :])

        iters += 1
    end = time.time()
    print("bfgs")
    print("Врямя запуска: ", end - start)
    print("Использованная память в процентах: ", psutil.virtual_memory().used)
    print("Использованно памяти в процентах: ", psutil.virtual_memory().percent)
    return [list(w), iters, grad_calc, func_calc]


w_start = [5, 5]
max_iters = 1000
learning_rate = 1
eps = 0.001
#
# test_gauss_newton = gauss_newton(X, Y, w_start, learning_rate, max_iters)
# print("function:", test_gauss_newton[0][0], 'w[0]', str(test_gauss_newton[0][1]) + "w[1]",
#       str(test_gauss_newton[0][2]) + "w[2]"
#                                      "\nepoch:", test_gauss_newton[1],
#       "\njacobi_calculations:", test_gauss_newton[2],
#       "\npoints:")

# test_powell_dog_leg = trust_region_powell_dog_leg(X, Y, w_start, max_iters)
# print("function:", test_powell_dog_leg[0][0], '+', str(test_powell_dog_leg[0][1]) + " * x",
#       "\nepoch:", test_powell_dog_leg[1],
#       "\njacobi_calculations:", test_powell_dog_leg[2],
#       "\npoints:")

# test_bfgs = bfgs(X, Y, w_start, eps, max_iters)
# print("function:", test_bfgs[0][0], '+', str(test_bfgs[0][1]) + " * x",
#       "\nepoch:", test_bfgs[1],
#       "\npoints:")

test_lbfgs = l_bfgs(X, Y, w_start, eps, max_iters)
print("function:", test_lbfgs[0][0], '+', str(test_lbfgs[0][1]) + " * x",
      "\nepoch:", test_lbfgs[1],
      "\npoints:")

plt.scatter(X, Y, alpha=0.4)
plt.plot(X, Y, 'g', linewidth=2.0)
Y_current1 = np.array([regression(x, test_lbfgs[0]) for x in X])
plt.plot(X, Y_current1, 'b', linewidth=2.0)
plt.show()
