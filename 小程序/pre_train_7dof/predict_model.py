import numpy as np
from scipy.spatial.transform import Rotation as R

def predict_model(input_data):
    theta1 = input_data[0]
    theta2 = input_data[1]
    theta3 = input_data[2]
    theta4 = input_data[3]
    theta5 = input_data[4]
    theta6 = input_data[5]
    theta7 = input_data[6]

    R0_1L = np.array([[np.cos(theta1), -np.sin(theta1), 0], [np.sin(theta1), np.cos(theta1), 0], [0, 0, 1]])
    R1_2L = np.array([[0, 0, 1], [np.cos(theta2), -np.sin(theta2), 0], [np.sin(theta2), np.cos(theta2), 0]])
    R2_3L = np.array([[-np.cos(theta3), np.sin(theta3), 0], [0, 0, 1], [np.sin(theta3), np.cos(theta3), 0]])
    R3_4L = np.array([[-np.sin(theta4), -np.cos(theta4), 0], [np.cos(theta4), -np.sin(theta4), 0], [0, 0, 1]])
    R4_5L = np.array([[np.cos(theta5), -np.sin(theta5), 0], [np.sin(theta5), np.cos(theta5), 0], [0, 0, 1]])
    R5_6L = np.array([[0, 0, 1], [np.cos(theta6), -np.sin(theta6), 0], [np.sin(theta6), np.cos(theta6), 0]])
    R6_7L = np.array([[-np.cos(theta7), np.sin(theta7), 0], [0, 0, 1], [np.sin(theta7), np.cos(theta7), 0]])
    R7_8L = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    P0_1L = np.array([0.8, 0, 1.1])[:, np.newaxis]
    P1_2L = np.array([0.2, 0, 0.17])[:, np.newaxis]
    P2_3L = np.array([0, 0.2, 0.17])[:, np.newaxis]
    P3_4L = np.array([0, 1.4, 0.32])[:, np.newaxis]
    P4_5L = np.array([1.4, 0, 0.32])[:, np.newaxis]
    P5_6L = np.array([0.2, 0, 0.17])[:, np.newaxis]
    P6_7L = np.array([0, 0.2, 0.17])[:, np.newaxis]
    P7_8L = np.array([0, 0, 0.2])[:, np.newaxis]

    share_vector = np.array([0, 0, 0, 1], dtype=float)[np.newaxis, :]
    R0_1L34 = np.concatenate((R0_1L, P0_1L), axis=1)
    R0_1L44 = np.concatenate((R0_1L34, share_vector), axis=0)
    R1_2L34 = np.concatenate((R1_2L, P1_2L), axis=1)
    R1_2L44 = np.concatenate((R1_2L34, share_vector), axis=0)
    R2_3L34 = np.concatenate((R2_3L, P2_3L), axis=1)
    R2_3L44 = np.concatenate((R2_3L34, share_vector), axis=0)
    R3_4L34 = np.concatenate((R3_4L, P3_4L), axis=1)
    R3_4L44 = np.concatenate((R3_4L34, share_vector), axis=0)
    R4_5L34 = np.concatenate((R4_5L, P4_5L), axis=1)
    R4_5L44 = np.concatenate((R4_5L34, share_vector), axis=0)
    R5_6L34 = np.concatenate((R5_6L, P5_6L), axis=1)
    R5_6L44 = np.concatenate((R5_6L34, share_vector), axis=0)
    R6_7L34 = np.concatenate((R6_7L, P6_7L), axis=1)
    R6_7L44 = np.concatenate((R6_7L34, share_vector), axis=0)
    R7_8L34 = np.concatenate((R7_8L, P7_8L), axis=1)
    R7_8L44 = np.concatenate((R7_8L34, share_vector), axis=0)

    R0_8L44 = R0_1L44.dot(R1_2L44).dot(R2_3L44).dot(R3_4L44).dot(R4_5L44).dot(R5_6L44).dot(R6_7L44).dot(R7_8L44)
    Rotation33 = R0_8L44[:3, :3]

    r = R.from_matrix(Rotation33)
    rot_quat = r.as_quat()
    postion = R0_8L44[:3, 3]

    return postion, rot_quat

if __name__ == '__main__':
    aa, bb = predict_model(np.array([30, 10, 20, 20, 40, 50, 60]) * np.pi / 180)
    print(aa, "\n", bb)
