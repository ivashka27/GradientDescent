import torch
from autograd import grad
import tensorflow as tf
import time
import psutil as psutil


def pyTorch(f, x):
    start = time.time()
    x = torch.tensor(x, requires_grad=True)
    y = f(x)
    y.backward()
    end = time.time()
    return x.grad.item(), end - start, psutil.virtual_memory().used, psutil.virtual_memory().percent



def autoGrad(f, x):
    start = time.time()
    grad_f = grad(f)
    gradient = grad_f(x)
    end = time.time()
    return gradient, end - start, psutil.virtual_memory().used, psutil.virtual_memory().percent



def tensor(f ,x):
    start = time.time()
    x2 = tf.Variable(x)
    with tf.GradientTape() as tape:
        y = f(x2)
    grad = tape.gradient(y, x2)
    end = time.time()
    return grad.numpy(), end - start, psutil.virtual_memory().used, psutil.virtual_memory().percent



def function(x):
    return x**2 + 2*x + 1


x = 2.0

massFun = [pyTorch, autoGrad, tensor]

for i in massFun:
    gradD = 0
    timeF = 0
    ramB = 0
    ramP = 0
    for j in range(1, 5):
        funR = i(function, x)
        gradD += funR[0]
        timeF += funR[1]
        ramB += funR[2]
        ramP += funR[3]
    print(i, ' Gradient:', gradD / 4,
          ' time in seconds: ', timeF / 4)
    print(
        ' used memery in bytes: ', ramB / 4,
        ' used memery in percent: ', ramP / 4)

