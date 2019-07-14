import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import math
import argparse


def read_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--print', action="store_true", help="Print the graph")
    args = parser.parse_args()
    df = pd.read_csv('data.csv')

    m, b = gradient_descent(df.iloc[:,0], df.iloc[:,1])
    if (args.print):
        print_line(df.iloc[:,0], df.iloc[:,1], m, b, df.columns.values)

def gradient_descent(x, y):
    max_x = max(x)
    m = b = tmp_cost = 0
    iterations = 100000
    n = float(len(x))
    learning_rate = 0.01
    x = x / max(x)

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
    np.save("train_result", [m, b])
    return m, b

def print_line(x, y, m, b, header):
    xDraw = np.linspace(min(x),max(x),1000)
    yDraw = m * xDraw + b
    plt.scatter(x, y, c = 'red')
    plt.plot(xDraw, yDraw)
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    plt.show()

def main():
    read_file()

if __name__ == "__main__":
	main()
