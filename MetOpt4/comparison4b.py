import torch
from autograd import grad
import tensorflow as tf
import time
import psutil as psutil

start = time.time()

x = 2.0


def function(x):
    return x**2 + 2*x + 1


def pyTorch(f, x):
    start = time.time()
    x = torch.tensor(x, requires_grad=True)
    y = f(x)
    y.backward()
    end = time.time()
    print('Gradient: ', x.grad.item(),
          ' time in seconds: ', end - start,
          ' used memery in bytes: ', psutil.virtual_memory().used,
          ' used memery in percent: ', psutil.virtual_memory().percent)


def atutoGrad(f, x):
    start = time.time()
    grad_f = grad(f)
    gradient = grad_f(x)
    end = time.time()
    print('Gradient:', gradient,
          ' time in seconds: ', end - start,
          ' used memery in bytes: ', psutil.virtual_memory().used,
          ' used memery in percent: ', psutil.virtual_memory().percent)


def tensor(f ,x):
    start = time.time()
    x2 = tf.Variable(x)
    with tf.GradientTape() as tape:
        y = f(x2)
    grad = tape.gradient(y, x2)
    end = time.time()
    print('Gradient:', grad.numpy(),
          ' time in seconds: ', end - start,
          ' used memery in bytes: ', psutil.virtual_memory().used,
          ' used memery in percent: ', psutil.virtual_memory().percent)


pyTorch(function, x)
atutoGrad(function, x)
tensor(function, x)
