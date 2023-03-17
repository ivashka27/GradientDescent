import random


def generate_function(n, k):
    if k == 0:
        return [0 for i in range(n)]

    a = random.uniform(k + 1, 100)
    b = a / k

    coefficients = [a, b]

    for i in range(2, n):
        coefficients.append(random.uniform(b + 1, a - 1))

    random.shuffle(coefficients)

    # if __name__ == '__main__':
    #     print(mass[0],'* (X 0 )^2', end="")
    #     for i in range(1, n):
    #         print(' +', mass[i],"(X",i,")^2", end="")

    return coefficients
