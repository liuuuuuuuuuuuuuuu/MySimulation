# 预测当前动作可以产生的瞬时奖励
import numpy as np


def predict_reward(current, target):
    pos_Target_EE = target[:3] - current[:3]
    distance = np.linalg.norm(pos_Target_EE)
    attitude_error = np.dot(current[3:7], target[3:7])
    # 是否需要添加其他奖励函数，比如说关节角度最小。

    reward_current = -2 - 10 * distance - 10 * np.log10(1 * distance + 1e-3) - 30 + \
                     50 * attitude_error - 50 - 10 * np.log10((1 - attitude_error) * 0.1 + 1e-2) - 50

    # reward_current = -2 - 10 * distance - np.log10(1 * distance + 1e-3) - 3

    return reward_current
