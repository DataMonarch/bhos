import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(10, 5, 1000)
y = np.random.normal(0, 1, 1000)

plt.plot(y, x)

plt.plot(x, y)

This is a buggy line

def blah(x):
    return x*1000
