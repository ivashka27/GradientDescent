import numpy as np

#initial approximation
x0 = np.array([-1.2, 1.0])

#step limit
# radius = 1

accuracy = 1e-6

max_iter = 1000

#Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x)**2


#rosenbrock gradient
def gradient_rosenbrock(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0]**2)])

#hessian rosenbrock
def hessian(x):
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                     [-400 * x[0], 200]])

#Finding the Optimal Gradient Direction
def linalgNorm(gradient):
    grad_norm = np.sqrt(np.sum(gradient * gradient))
    if grad_norm == 0:
        return np.zeros_like(gradient)

    direction = -gradient / grad_norm

    return direction


def optimalDirection(grad, hessian, radius): # redo late
    p_u = -grad / linalgNorm(grad) #optimal gradient direction
    p_b = -np.linalg.solve(hessian, grad) #Newton's optimal direction

    if linalgNorm(p_b) <= radius:
        p = p_b
    elif linalgNorm(p_u) >= radius:
        p = (radius / linalgNorm(p_u)) * p_u
    else:
        p_opt = p_u + ((linalgNorm(p_b) - radius) /
                       linalgNorm(p_b - p_u)) * (p_b - p_u)
        p = min(linalgNorm(p_u), radius) * p_opt / linalgNorm(p_opt)

    return p

def minimumFunction(step, xy1, max_iter, radius):
    i = 0
    while i < max_iter:
        f1 = rosenbrock(xy1)
        gradF1 = gradient_rosenbrock(xy1)
        hessianF1 = hessian(xy1)
        directionStep = optimalDirection(gradF1, hessianF1, radius)
        xy1_new = xy1 + directionStep # maybe minStep is not np.array
        f2 = rosenbrock(xy1_new)
        delta = f1 - f2
        quadraticModel = -(np.dot(gradF1, directionStep) + 0.5 * np.dot(directionStep, np.dot(hessianF1, directionStep)))
        if quadraticModel == 0:
            rho =  accuracy
        else:
            rho = delta / quadraticModel
        euclideanNorm = np.sqrt(np.dot(directionStep, directionStep))
