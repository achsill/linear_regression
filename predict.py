import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')

input = raw_input('Mileage: ')
train_result = np.load("train_result.npy")
print(train_result[0])
print(train_result[1])

to_predict = normalization(float(input))


print(train_result[0] * to_predict + train_result[1])
# print(asked_mileage)


(value - min(tmp_x)) / (max(tmp_x) - min(tmp_x))
