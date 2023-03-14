from re import A
import random
import numpy as np
import matplotlib.pyplot as plt

fig, (axs1, axs2) = plt.subplots(2)  # If you need more hists.
plt.subplots_adjust(wspace=1, hspace=1)
axs1.grid(color='k', alpha=0.3)
axs2.grid(color='k', alpha=0.3)
n = int(input("Enter the number of splits:"))
type = input("""Enter equipment(left|right|middle|select|random):""")
print(type)

if type == "left":
    r = 1
if type == "right":
    r = 0
if type == "middle":
    r = 0.5
if type == "select":
    r = float(input("Enter any equipment[0;1]):"))
x = []
y = []
dn = 1 / n
S = float(0)
# create an integral sum.(S)
for i in range(1, n + 1):
    if type == "random":
        r = round(random.random(), 3)
        S += (3 ** (1 + (i - r) / n) * (1 / n))
    else:
        S += (3 ** (1 + (i - r) / n) * (1 / n))
    x += [(1 + (i - 0.5) / n)]
    y += [(3 ** (1 + (i - r) / n))]
print("The resulting value of the integral:", S)
# draw function y=3^x
xmin = 1
xmax = 2
c = 200000
xlist = np.linspace(xmin, xmax, c)
ylist = [3 ** x for x in xlist]
axs1.plot(xlist, ylist, color='k', alpha=1, lw=1.5)
axs2.plot(xlist, ylist, color='k', alpha=1, lw=2.5)
# draw hist.
axs1.bar(x, y, dn, color=(255 / 256, 112 / 256, 13 / 256), alpha=0.5, lw=1.5,
         edgecolor=(255 / 256, 112 / 256, 13 / 256))
axs1.set_title('f(x)=3^x & Integral sum : x∈[1,2]')
axs2.set_title('Trapezoidal method: f(x)=3^x : x∈[1,2]')
axs1.set_xlabel('Argument value - x')
axs2.set_xlabel('Argument value - x')
axs1.set_ylabel('Function value - f(x)')
axs2.set_ylabel('Function value - f(x)')
# trapezoidal method
n = int(input("Trapezoidal method: Enter the number of splits:"))
S1 = float(0);
for i in range(1, n + 1):
    S1 += (3 ** (1 + (i - 1) / n) + 3 ** (1 + (i) / n)) / (2 * n)
    x1 = 1 + (i - 1) / n
    y1 = 3 ** (x1)
    x2 = 1 + (i) / n
    y2 = 3 ** (x2)
    x = [x1, x2]
    y = [y1, y2]
    axs2.plot(x, y, 'o-r', c=(255 / 256, 112 / 256, 13 / 256), alpha=1, lw=1.5, mec='k', mew=1.5, ms=2)
    axs2.fill_between(x, y, color=(255 / 256, 112 / 256, 13 / 256), alpha=0.5)
print("Trapezoidal method: The resulting value of the integral:", S1)
plt.show()
