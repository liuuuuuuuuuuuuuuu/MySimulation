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
        a0=0
        joint_state1[i] = joint_state[i] + a0
        _, _, current_EE = predict_model(joint_state1)
        r0 = predict_reward(current_EE, target)
        a1 = a0 - 0.4/180*np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 25])
        joint_state1[i] = joint_state[i] + a1
        _, _, current_EE = predict_model(joint_state1)
        r1 = predict_reward(current_EE, target)
        a2 = a0 + 0.4/180*np.pi * np.min([1, 10 * distance, (1 - attitude_error) * 25])
        joint_state1[i] = joint_state[i] + a2
        _, _, current_EE = predict_model(joint_state1)
        r2 = predict_reward(current_EE, target)

        action_initial = np.array([a0, a1, a2])
        number = np.argmax([r0, r1, r2])
        action_output[i] = action_initial[number]

    return action_output/180*np.pi