import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("result_ep_reward_data.npy")
data2 = np.load("result_total_data.npy")
print(data2)
plt.plot(data1)
plt.show()