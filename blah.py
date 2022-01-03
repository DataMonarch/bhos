import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(10, 5, 1000)
y = np.random.normal(0, 1, 1000)
xx, yy = np.meshgrid(x, y)
z = xx * yy
z.shape
plt.plot(y, x)

plt.plot(x, y)

plt.hist(z.flatten())
plt.imshow(z)
