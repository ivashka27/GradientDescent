import random

def function(n, k):
    if k == 0:
        print(0)
        exit()

    a = random.uniform(k + 1, 100)
    b = a / k

    mass = list()
    mass.append(a)
    mass.append(b)

    for i in range(2, n):
         mass.append(random.uniform(b + 1, a - 1))

    random.shuffle(mass)

    if __name__ == '__main__':
        print(mass[0],'* (X 0 )^2', end="")
        for i in range(1, n):
            print(' +', mass[i],"(X",i,")^2", end="")

    return mass


if __name__ == '__main__':
    n, k = map(int, input().split())
    q = function(n, k)
    x = [1] * n
