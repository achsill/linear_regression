import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import math


df = pd.read_csv('data.csv')
x = df.iloc[:,0]
y = df.iloc[:,1]
min_x = min(x)
max_x = max(x)

def normalization(value):
    return (value - min_x) / (max_x - min_x)

def denormalization(value):
    return (value * (max_x - min_x) + min_x)

m = 0
b = 0
iterations = 100000
n = float(len(x))
learning_rate = 0.01
x = x / max(x)
tmp_cost = 0
for i in range(iterations):
    y_predicted = m * x + b
    cost = (1/n) * sum([val ** 2 for val in (y - y_predicted)])
    if (abs(tmp_cost - cost) < 0.001):
        break
    tmp_m = (1/n) * sum((y_predicted - y) * x)
    tmp_b = (1/n) * sum(y_predicted - y)
    m = m - learning_rate * tmp_m
    b = b - learning_rate * tmp_b
    tmp_cost = cost

x = x * max_x
m = m / max(x)

print(m, m / max(x), max(x), b)
np.save("train_result", [m, b])


xDraw = np.linspace(min(x),max(x),1000)
yDraw = m * xDraw + b
plt.scatter(x, y)
plt.plot(xDraw, yDraw)
plt.show()
