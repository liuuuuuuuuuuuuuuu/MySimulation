import numpy as np
from predict_reward import predict_reward
from predict_model import predict_model
import torch


def action_random(joint_state, pos_Target_EE, attitude_error, action, target):
    action_output1 = np.zeros(7)
    action_output = np.zeros(7)
    distance = np.linalg.norm(pos_Target_EE)
    joint_state1 = joint_state.copy()
    for i in range(7):
        temp = action

        a0 = temp[i]
        joint_state1[i] = joint_state[i] + a0
        _, _, current_EE = predict_model(joint_state1)
        r0 = predict_reward(current_EE, target)

        a1 = a0 - 0.01 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a1 = np.clip(a1, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a1
        _, _, current_EE = predict_model(joint_state1)
        r1 = predict_reward(current_EE, target)

        a2 = a0 + 0.01 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a2 = np.clip(a2, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a2
        _, _, current_EE = predict_model(joint_state1)
        r2 = predict_reward(current_EE, target)

        a3 = a0 - 0.05 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a3 = np.clip(a3, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a3
        _, _, current_EE = predict_model(joint_state1)
        r3 = predict_reward(current_EE, target)

        a4 = a0 + 0.05 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a4 = np.clip(a4, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a4
        _, _, current_EE = predict_model(joint_state1)
        r4 = predict_reward(current_EE, target)

        a5 = a0 - 0.1 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a5 = np.clip(a5, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a5
        _, _, current_EE = predict_model(joint_state1)
        r5 = predict_reward(current_EE, target)

        a6 = a0 + 0.1 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a6 = np.clip(a6, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a6
        _, _, current_EE = predict_model(joint_state1)
        r6 = predict_reward(current_EE, target)

        a7 = a0 - 0.2 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a7 = np.clip(a7, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a7
        _, _, current_EE = predict_model(joint_state1)
        r7 = predict_reward(current_EE, target)

        a8 = a0 + 0.2 / 180 * np.pi * np.min([1, np.max([10 * distance, (1 - attitude_error) * 3])])
        a8 = np.clip(a8, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
        joint_state1[i] = joint_state[i] + a8
        _, _, current_EE = predict_model(joint_state1)
        r8 = predict_reward(current_EE, target)

        _, _, current_EE = predict_model(joint_state)
        r_last = predict_reward(current_EE, target)

        action_initial = np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, 0])
        number = np.argmax([r0, r1, r2, r3, r4, r5, r6, r7, r8, r_last])
        action_output1[i] = action_initial[number]

    joint_state_final = joint_state + action_output1
    _, _, current_EE = predict_model(joint_state_final)
    r_output = predict_reward(current_EE, target)

    a11 = np.zeros(7)
    a11 = np.clip(a11, -0.2/ 180 * np.pi, 0.2/ 180 * np.pi)
    joint_state1 = joint_state + a11
    _, _, current_EE = predict_model(joint_state1)
    r11 = predict_reward(current_EE, target)

    for j in range(100):
        a12 = np.array(np.random.normal(0, scale=0.001, size=7))

        a12 = np.clip(a12, -0.2 / 180 * np.pi, 0.2 / 180 * np.pi)
        joint_state1 = joint_state + a12
        _, _, current_EE = predict_model(joint_state1)
        r12 = predict_reward(current_EE, target)

        if r12 > r11:
            a11 = a12
            r11 = r12

    action_all = np.array([action_output1, a11])
    number1 = np.argmax([r_output, r11])

    action_output = action_all[number1]

    return action_output  # 弧度制
