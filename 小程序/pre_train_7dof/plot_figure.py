import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("loss.npy")
print(data1)
plt.plot(data1)
plt.show()