import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import psutil as psutil
import time


start = time.time()


def objective(x):
    return 0.5 * x ** 4 + x + x ** 3


x0 = np.array([100])
ub_linear = np.array([-1.68])
lb_linear = np.array([-10000])


def constraint(x):
    return 3 * x
nonlinear_constraint = NonlinearConstraint(constraint, lb_linear, ub_linear)


# Минимизация с нелинейными ограничениями
result_nonlinear = minimize(objective, x0, constraints=nonlinear_constraint)

end = time.time()

print("Результаты с нелинейными ограничениями:")
print("Время в секундах: ", end - start)
print("Использованная память в байтах:", psutil.virtual_memory().used)
print("Использованная память в %:", psutil.virtual_memory().percent)
print("Оптимальное значение x:", result_nonlinear.x)
print("Количество итераций:", result_nonlinear.nit)
