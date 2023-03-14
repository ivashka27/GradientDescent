import numpy as np
import matplotlib.pyplot as plt


# f(x, y) = ax^2 + by^2 + cxy + dx + ey + f
# make_function() returns f(x, y) as a dictionary
def make_function(a, b, c, d, e, f):
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f}


# return list of function values
def function_value_list(X, Y, function):
    res = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        res += [get_current_value(x, y, function)]
    return res


# return current value of function
def get_current_value(x, y, function):
    return function['a'] * x ** 2 + function['b'] * y ** 2 + function['c'] * x * y + \
           function['d'] * x + function['e'] * y + function['f']


# calculate_gradient() returns gradient of the function
def calculate_gradient(x, y, coefficient):
    return [
        2.0 * coefficient['a'] * x + coefficient['c'] * y + coefficient['d'],
        2.0 * coefficient['b'] * y + coefficient['c'] * x + coefficient['e']
    ]


'''

gradient_descent

'''


def gradient_descent(start_x, start_y, function, learning_rate, eps):
    prev_x = start_x
    prev_y = start_y
    condition = 1
    res = []
    res += [(start_x, start_y)]

    while condition:
        grad = calculate_gradient(prev_x, prev_y, function)
        new_x = prev_x - learning_rate * grad[0]
        new_y = prev_y - learning_rate * grad[1]
        res += [[new_x, new_y]]
        if abs(prev_x - new_x) <= eps and abs(prev_y - new_y) <= eps:
            break
        prev_x = new_x
        prev_y = new_y
    return res


f1 = make_function(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
f2 = make_function(2.0, 6.0, 0.0, 15.0, 2.0, 0.0)

# draw function graph
t = np.linspace(-20, 10, 100)
X, Y = np.meshgrid(t, t)
Z = np.array(function_value_list(X, Y, f2))
fig, ax1 = plt.subplots(1)
fig = plt.figure().add_subplot(projection='3d')
fig.plot_surface(X, Y, Z)
##
step = 0.01
epsilon = 0.001

res1 = gradient_descent(-5.0, -5.0, f1, step, epsilon)
res2 = gradient_descent(-5.0, -5.0, f2, step, epsilon)
points = np.array(res2)
# draw gradient move
ax1.plot(points[:, 0], points[:, 1], 'o-')
ax1.grid()
ax1.contour(X, Y, Z, levels=sorted([get_current_value(p[0], p[1], f2) for p in points]))
##

# print(res1)
# print(res2)
plt.show()
