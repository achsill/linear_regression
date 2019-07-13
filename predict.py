import numpy as np
import pandas as pd
import math as Math

train_result = np.load("train_result.npy")

while(True):
    mileage = input("Mileage: ")
    if not mileage.lower() == "exit":
        print("{}{}".format(round(train_result[0] * float(mileage) + train_result[1], 2), "$"))
    else:
        break
