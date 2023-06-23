import math

import numpy as np
from matplotlib import pyplot as plt


def generate_func(x):
    return math.sqrt(x) + math.sin(x)


def generate_sample(start, count, step):
    x = start
    i = 0
    while i < count:
        i += 1
        yield generate_func(x) + np.random.uniform(-0.8, 0.8) * np.random.uniform(0, 0.8)
        x += step


count = 100
step = 0.1
start = 0.2
X = np.arange(start, count * step + start, step)
Y = np.array([round(y, 2) for y in generate_sample(start, count, step)])


def error(X, Y, w):
    value = 0
    for i in range(len(Y)):
        value += get_vector(X[i], Y[i], w) ** 2
    return value


def get_vector(x, y, w):
    return y - regression(x, w)


def get_vectors(X, Y, w):
    vectors = np.zeros(len(Y))
    for i in range(len(Y)):
        vectors[i] = get_vector(X[i], Y[i], w)
    return vectors


def regression(x, w):
    return w[0] * x / (w[1] + x)


def get_grad(x, y, w):
    return np.array((- x / (w[1] + x),
                     (w[0] * x) / ((w[1] + x) ** 2)))


def jacobian(X, Y, w):
    jacobian = np.zeros((len(Y), 2))
    for i in range(len(Y)):
        jacobian[i] = get_grad(X[i], Y[i], w)
    return jacobian


def get_second_diff(x, y, w):
    return np.array([(0, -2 * (w[0] * x) / ((w[1] + x) ** 3))])


def hessian(X, Y, w):
    hessian = np.zeros((len(Y), 2))
    for i in range(len(Y)):
        hessian[i] = get_second_diff(X[i], Y[i], w)
    return hessian


def gauss_newton(X, Y, w_start, learning_rate, max_iters):
    iters = 0
    jacobi_calc = 0
    w = w_start
    points = []
    while iters < max_iters:
        j = jacobian(X, Y, w)
        jacobi_calc += 1
        pseudo_j = np.linalg.pinv(j)
        jj = np.dot(pseudo_j, get_vectors(X, Y, w))
        w = w - learning_rate * jj
        points.append(w)
        iters += 1
        print("Error:", error(X, Y, w))
        print("Weights:", w)
        if np.linalg.norm(jj) < 10 ** -3:
            break
    return [w, iters, jacobi_calc, points]


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
    eta = 0.2
    eps = 1e-3
    start_trust_radius = 1
    max_trust_radius = 10
    trust_radius = start_trust_radius
    iters = 0
    jacobian_calc = 0
    hessian_calc = 0
    w = w_start
    points = []
    while iters < max_iters:
        J = jacobian(X, Y, w)
        jacobian_calc += 1
        H = hessian(X, Y, w)
        hessian_calc += 1
        delta = powell_dog_leg(np.linalg.inv(H), J, H, trust_radius)
        actual_reduction = error(X, Y, w) - error(X, Y, w + delta)
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
    return [w, iters, jacobian_calc, hessian_calc, points]


def function(p):
    return lambda x: p[0] + p[1] * x


def grad_calculate(x, func, dim):
    h = 1e-5
    res = []
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)


def grad_function(f, dim):
    return lambda x: grad_calculate(x, f, dim)


def error_function(f, X, y):
    def error(w):
        error = 0
        for i in range(len(X)):
            error += (f(w)(X[i]) - y[i]) ** 2
        return error

    return error


f = error_function(function, X, Y)
grad = grad_function(f, 2)


def wolfe_conditions(f, fx, gradient, grx, x, d, t):
    c1 = 1e-4
    c2 = 0.9
    grxd = np.dot(grx, d)
    ft = f(x + t * d)
    gxt = np.dot(gradient(x + t * d), d)
    first_condition = ft <= fx + c1 * t * grxd
    second_condition = gxt <= - c2 * grxd
    return first_condition and second_condition


def bfgs(f, grad_f, w_start, eps, max_iters):
    w = [w_start]
    grad_calc = 0
    func_calc = 0
    grad_f = grad_f(w_start)
    I = np.eye(len(w_start))
    H = I
    i = 0
    while i < max_iters:
        pk = -np.dot(H, grad_f)
        lr = 1
        while not wolfe_conditions(f, f(w_start), grad_f, grad_f, w_start, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2
        x_n = w_start + lr * pk
        s = x_n - w_start
        w_start = x_n
        grad_n = grad_f(x_n)
        y = grad_n - grad_f
        grad_f = grad_n
        ro = 1.0 / (np.dot(y, s) + eps)
        H = np.dot(I - ro * s[:, np.newaxis] * y[np.newaxis, :],
                   np.dot(H, I - ro * y[:, np.newaxis] * s[np.newaxis, :])) + \
            (ro * s[:, np.newaxis] * s[np.newaxis, :])
        w = w_start
        if np.linalg.norm(grad_f) < eps:
            break
        i += 1
    return [list(w), grad_calc, func_calc]


w_start = [0, 0]
max_iters = 20
learning_rate = 1

test_gauss_newton = gauss_newton(X, Y, w_start, learning_rate, max_iters)
print("function:", test_gauss_newton[0][0], '+', str(test_gauss_newton[0][1]) + " * x",
      "\nepoch:", test_gauss_newton[1],
      "\njacobi_calculations:", test_gauss_newton[2],
      "\npoints:")

# test_powell_dog_leg = trust_region_powell_dog_leg(X, Y, w_start, max_iters)
# print("function:", test_powell_dog_leg[0][0], '+', str(test_powell_dog_leg[0][1]) + " * x",
#       "\nepoch:", test_powell_dog_leg[1],
#       "\njacobi_calculations:", test_powell_dog_leg[2],
#       "\njacobi_calculations:", test_powell_dog_leg[3],
#       "\npoints:")
#
# test_bfgs = bfgs(X, Y, w_start, max_iters)
# print("function:", test_bfgs[0][0], '+', str(test_bfgs[0][1]) + " * x",
#       "\nepoch:", test_bfgs[1],
#       "\njacobi_calculations:", test_bfgs[2],
#       "\njacobi_calculations:", test_bfgs[3],
#       "\npoints:")

plt.scatter(X, Y, alpha=0.4)
plt.plot(X, Y, 'g', linewidth=2.0)
Y_current1 = np.array([regression(x, test_gauss_newton[0]) for x in X])
plt.plot(X, Y_current1, 'b', linewidth=2.0)
plt.show()
