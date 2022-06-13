import numpy as np
import random

input_data = []
output_data = []
for i in range(10000):
    theta1 = random.uniform(-np.pi / 2, np.pi / 2)
    theta2 = random.uniform(-np.pi / 2, np.pi / 2)
    theta3 = random.uniform(-np.pi / 2, np.pi / 2)

    J11 = - np.sin(theta1 + theta2 + theta3) / 2 - np.sin(theta1 + theta2) - np.sin(theta1)
    J12 = - np.sin(theta1 + theta2 + theta3) / 2 - np.sin(theta1 + theta2)
    J13 = -np.sin(theta1 + theta2 + theta3) / 2
    J21 = np.cos(theta1 + theta2 + theta3) / 2 + np.cos(theta1 + theta2) + np.cos(theta1)
    J22 = np.cos(theta1 + theta2 + theta3) / 2 + np.cos(theta1 + theta2)
    J23 = np.cos(theta1 + theta2 + theta3) / 2

    J = [[J11, J12, J13], [J21, J22, J23]]
    target = [[random.uniform(-1, 1)], [random.uniform(-1, 1)]]

    del_q = np.dot(np.linalg.pinv(J), target)

    input_data1 = np.array([theta1, theta2, theta3, 0, 0, 0, target[0][0], target[1][0], 0])
    output_data1 = del_q*100
    input_data.append(input_data1)
    output_data.append(output_data1)

output_data = output_data
np.save("input_data", input_data)
np.save("output_data", output_data)

np.load("/home/liujian/桌面/single_arm_3dof_torch/DDPG/pre_train/input_data.npy")
np.load("/home/liujian/桌面/single_arm_3dof_torch/DDPG/pre_train/output_data.npy")
