import numpy as np
from predict_reward import predict_reward


def action_random(joint_state, pos_Target_EE, attitude_error, action):
    action_output = np.zeros(7)
    distance = np.linalg.norm(pos_Target_EE)
    for i in range(7):
        temp = action
        a0 = temp[i]
        r0 = predict_reward(joint_state + a0)
        a1 = a0 - 0.1 * np.min([1, 10 * distance, (1 - attitude_error) * 25])
        temp[i] = a1
        r1 = predict_reward(joint_state + a1)
        a2 = a0 + 0.1 * np.min([1, 10 * distance, (1 - attitude_error) * 25])
        temp[i] = a2
        r2 = predict_reward(joint_state + a2)

        action_initial = np.array([a0, a1, a2])
        number = np.argmax([r0, r1, r2])
        action_output[i] = action_initial[number]

    return action_output