import numpy as np
from scipy.optimize import minimize
import time
import psutil as psutil


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


for i in range(1, 5):
    x0 = np.array([0.5 * i])
    bounds2 = [(-2 * i, 2 * i)]
    minimizezf(f, x0, bounds2)