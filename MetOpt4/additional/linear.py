import numpy as np
from scipy.optimize import minimize, LinearConstraint
import psutil as psutil
import time


start = time.time()

def objective(x):
    return 0.5 * x ** 4 + x + x ** 3

x0 = np.array([100])

A_linear = np.array([[3]])
ub_linear = np.array([-1.68])
lb_linear = np.array([-10000])
linear_constraint = LinearConstraint(A_linear, lb_linear, ub_linear)


result_linear = minimize(objective, x0, constraints=linear_constraint)

end = time.time()


print("Результаты с линейными ограничениями:")
print("Время в секундах: ", end - start)
print("Использованная память в байтах:", psutil.virtual_memory().used)
print("Использованная память в %:", psutil.virtual_memory().percent)
print("Оптимальное значение x:", result_linear.x)
print("Количество итераций:", result_linear.nit)
