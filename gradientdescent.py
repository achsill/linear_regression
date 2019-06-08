import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import math
from sklearn.preprocessing import normalize


df = pd.read_csv('data.csv')
x = df.iloc[:,0]
y = df.iloc[:,1]
print(x)
tmp_x = x
def normalization(value):
    return (value - min(tmp_x)) / (max(tmp_x) - min(tmp_x))

def denormalization(value):
    return (value * (max(tmp_x) - min(tmp_x)) + min(tmp_x))

m_curr = b_curr = 0
iterations = 50000
n = len(x)
learning_rate = 0.01
x = normalization(x)
for i in range(iterations):
    y_predicted = m_curr * x + b_curr
    cost = (1/n) * sum([val ** 2 for val in (y - y_predicted)])
    md = (1/(2*n))*sum((y_predicted - y)*x)
    bd = (1/(2*n))*sum(y_predicted - y)
    m_curr = m_curr - learning_rate * md
    b_curr = b_curr - learning_rate * bd

# print(m_curr, b_curr)
print(x)

testX = normalization(230000)

testY = m_curr * testX + b_curr
print(testY)

xDraw = np.linspace(min(x),max(x),1000)
yDraw = m_curr * xDraw + b_curr
plt.scatter(x, y)
plt.plot(xDraw, yDraw)
plt.show()
