import numpy as np
from scipy.optimize import minimize
import time
import psutil as psutil
from scipy.optimize import least_squares


def fun(x):
    return np.array([x[0] + 0.5 * np.exp(-x[0] / x[1]), x[1] * np.exp(-x[0] / x[1]) - 1])


def f(x):
    return x ** 2 + 2 * x + 1

def minimizezf(f, x, segment):
    start = time.time()
    x0 = x
    bounds = segment
    result_with_bounds = minimize(f, x0, bounds=bounds)
    end = time.time()
    print("Number of iterations: ", result_with_bounds.nit,
          ' time in seconds: ', end - start,
          ' used memery in bytes: ', psutil.virtual_memory().used,
          ' used memery in percent: ', psutil.virtual_memory().percent)
    # print(result_with_bounds) # это вывод всех получившихся результатов


def least_squaresf(f, x, segment):
    start = time.time()
    x0 = x
    bounds = segment
    result_with_bounds = least_squares(f, x0, bounds=bounds)
    end = time.time()
    print("Number of iterations: ", result_with_bounds.nfev,
          ' time in seconds: ', end - start,
          ' used memery in bytes: ', psutil.virtual_memory().used,
          ' used memery in percent: ', psutil.virtual_memory().percent)
    # print(result_with_bounds) # это вывод всех получившихся результатов


for i in range(1, 5):
    x0 = np.array([0.5 * i])
    bounds2 = [(-2 * i, 2 * i)]
    minimizezf(f, x0, bounds2)


for i in range(1, 5):
    x0 = np.array([1.0 * i, 1.0 * i])
    bounds = ([1 * (-i), 1 * (-i)], [2 * i, 2 * i])
    least_squaresf(fun, x0, bounds)