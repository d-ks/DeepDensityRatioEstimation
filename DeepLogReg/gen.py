import numpy as np
import os

# dataset generator

#is there "data" directory?
if os.path.isdir("./data/")!=True:
    os.mkdir("./data/")

#1st data ... expert dist. mean, std dev, n of data
exp = np.random.normal(0, 1, 10000)

#2nd data ... baseline dist.
base = np.random.normal(0, 5, 10000)
print(exp.shape)

np.savetxt("./data/exp.csv", exp)
np.savetxt("./data/base.csv", base)
