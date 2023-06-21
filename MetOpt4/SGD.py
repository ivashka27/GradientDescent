import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil as psutil


def allmet(x, y, mod, Model, func):

    start = time.time()
    model = Model()
    if (mod == "sgd"):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif (mod == "adam"):
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif (mod == "rmsprop"):
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    elif (mod == "adagrad"):
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif (mod == "momentum"):
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif (mod == "nesterov"):
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    else:
        print("введен не корректный mod")
        return 0

    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(x.unsqueeze(1))

        loss = torch.nn.functional.mse_loss(outputs.squeeze(), y)

        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    x_start = np.linspace(0, 10, 50)
    y_start = func(x_start)
    predicted = model(x.unsqueeze(1)).detach().numpy()
    print("Использованная память в байтах:", psutil.virtual_memory().used)
    print("Использованная память в %:", psutil.virtual_memory().percent)
    end = time.time()
    print("Время в сеундах ", end - start)
    plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
    plt.plot(x.numpy(), predicted, label='Fitted line')
    plt.plot(x_start, y_start)
    plt.legend()

    plt.show()


def func(x):
    return 5 + 5 * x

# точки генятся отдельно, при попытке вставки генерации из 2 лабы возникает несогласованность типов (хуй знает почему)
x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25])  # Входные значения
y = torch.tensor([11.73, 6.75, 4.22, 11.0, 11.14, 9.65, 12.65, 18.1, 15.81, 14.49, 18.0, 14.7, 14.02, 21.6, 22.46, 27.21, 24.37, 27.19, 26.62, 30.67, 25.18, 30.12, 34.17, 35.54, 39.49, 37.09, 42.39, 37.54, 36.13, 43.12])  # Выходные значения

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

allmet(x, y, "sgd", Model, func)