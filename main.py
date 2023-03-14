import numpy as np

# f(x, y) = ax^2 + by^2 + cxy + dx + ey + f


def make_function(a, b, c, d, e, f):
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f}


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

    while condition:
        grad = calculate_gradient(prev_x, prev_y, function)
        new_x = prev_x - learning_rate * grad[0]
        new_y = prev_y - learning_rate * grad[1]
        res += [(new_x, new_y)]
        #print(prev_x, prev_y, new_x, new_y)
        if abs(prev_x - new_x) <= eps and abs(prev_y - new_y) <= eps:
            break
        prev_x = new_x
        prev_y = new_y
    return res


f1 = make_function(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
f2 = make_function(-2.0, -6.0, 0.0, 15.0, 2.0, 0.0)
f3 = make_function(-1.0, 3.0, 1.0, 0.0, -5.0, 0.0)

step = 0.06
epsilon = 0.00001

#res1 = gradient_descent(-5.0, -5.0, f1, step, epsilon)
res2 = gradient_descent(0.0, 0.0, f2, step, epsilon)
#res3 = gradient_descent(0.0, 0.0, f3, step, epsilon)

print(res2)
#print(res2)
#print(res3)
