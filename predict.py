import numpy as np
import pandas as pd

train_result = np.load("train_result.npy")

while(True):
    mileage = input("Mileage: ")
    if not mileage.lower() == "exit":
        print(train_result[0] * float(mileage) + train_result[1])
    else:
        break
