import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil as psutil


def allmet(x, y, mod, Model):

    start = time.time()
    model = Model()
    if (mod == "sgd"):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        num_epochs = 10 ** 4
    elif (mod == "adam"):
        optimizer = optim.Adam(model.parameters(), lr=0.4)
        num_epochs = 16
    elif (mod == "rmsprop"):
        optimizer = optim.RMSprop(model.parameters(), lr=0.1)
        num_epochs = 100
    elif (mod == "adagrad"):
        optimizer = optim.Adagrad(model.parameters(), lr=0.1)
        num_epochs = 10 ** 4
    elif (mod == "momentum"):
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        num_epochs = 100
    elif (mod == "nesterov"):
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        num_epochs = 100
    else:
        print("введен не корректный mod")
        return 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = model(x.unsqueeze(1))

        loss = torch.nn.functional.mse_loss(outputs.squeeze(), y)

        loss.backward()
        optimizer.step()

    predicted = model(x.unsqueeze(1)).detach().numpy()
    end = time.time()
    print("Время в сеундах ", end - start)
    return psutil.virtual_memory().used, psutil.virtual_memory().percent, end - start, predicted




class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)


def func(x):
    return 5 + 3 * x



x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25])
y = torch.tensor([4.75, 5.72, 6.36, 7.15, 8.02, 8.75, 9.22, 10.36, 10.96, 11.99, 12.37, 13.42, 13.96, 14.75, 15.53, 16.29, 17.12, 17.65, 18.45, 19.23, 20.18, 20.57, 21.59, 22.13, 23.16, 23.74, 24.83, 24.92, 25.91, 26.96])
predictedSum = np.array([[0]] * 30)
ramB = 0
ramP = 0
timeP = 0
for i in range (1, 5):
    ans = allmet(x, y, "sgd", Model) # меняем название
    ramB += ans[0]
    ramP += ans[1]
    timeP += ans[2]
    predictedSum += np.array(ans[3]).astype(np.int64)



print("Использованная память в байтах:", ramB / 4)
print("Использованная память в %:", ramP / 4)
print("Время в сеундах ", timeP / 4)
x_start = np.linspace(0, 10, 50)
y_start = func(x_start)
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original data')
plt.plot(x.numpy(), predictedSum / 4, label='Fitted line')
plt.plot(x_start, y_start)
plt.legend()
plt.show()
