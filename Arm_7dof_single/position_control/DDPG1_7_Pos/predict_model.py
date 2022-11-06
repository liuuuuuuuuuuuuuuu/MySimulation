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
    # 注意：此时四元数为[q1, q2, q3, q0],需要进一步转换
    rot_quat = np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])
    if rot_quat[0] < 0:
        rot_quat = rot_quat * -1
    postion = R0_8L44[:3, 3]
    pos_quat_total = np.concatenate((postion, rot_quat))

    return postion, rot_quat, pos_quat_total

if __name__ == '__main__':
    aa, bb, cc = predict_model(np.array([0.23, 1.57, 0.66, -2.41, 0.18, -1.34, 0.45]))
    print(aa, "\n", bb,"\n", cc)
