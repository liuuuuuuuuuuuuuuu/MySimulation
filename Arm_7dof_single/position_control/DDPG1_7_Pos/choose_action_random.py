import numpy as np
from predict_reward import predict_reward
from predict_model import predict_model


def action_random(joint_state, pos_Target_EE, attitude_error, action, target):
    action_output = np.zeros(7)
    distance = np.linalg.norm(pos_Target_EE)
    joint_state1 = joint_state*180/np.pi
    for i in range(7):
        temp = action
        a0 = temp[i]
        # a0=0
        joint_state1[i] = joint_state[i] + a0
        _, _, current_EE = predict_model(joint_state1)
        r0 = predict_reward(current_EE, target)
        a1 = a0 - 0.5/180*np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 50])
        a1 = np.clip(a1, -1 / 180 * np.pi, 1 / 180 * np.pi)
        joint_state1[i] = joint_state[i] + a1
        _, _, current_EE = predict_model(joint_state1)
        r1 = predict_reward(current_EE, target)
        a2 = a0 + 0.5/180*np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 50])
        a2 = np.clip(a2, -1 / 180 * np.pi, 1 / 180 * np.pi)
        joint_state1[i] = joint_state[i] + a2
        _, _, current_EE = predict_model(joint_state1)
        r2 = predict_reward(current_EE, target)

        a3 = a0 - 1 / 180 * np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 50])
        a3 = np.clip(a3, -1 / 180 * np.pi, 1 / 180 * np.pi)
        joint_state1[i] = joint_state[i] + a3
        _, _, current_EE = predict_model(joint_state1)
        r3 = predict_reward(current_EE, target)

        a4 = a0 + 1 / 180 * np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 50])
        a4 = np.clip(a4, -1 / 180 * np.pi, 1 / 180 * np.pi)
        joint_state1[i] = joint_state[i] + a4
        _, _, current_EE = predict_model(joint_state1)
        r4 = predict_reward(current_EE, target)

        action_initial = np.array([a0, a1, a2, a3, a4])
        number = np.argmax([r0, r1, r2, r3, r4])
        action_output[i] = action_initial[number]

    return action_output/180*np.pi