# 预测当前动作可以产生的瞬时奖励
import numpy as np


def predict_reward(current, target):
    pos_Target_EE = target[:3] - current[:3]
    distance = np.linalg.norm(pos_Target_EE)
    attitude_error = np.dot(current[3:7], target[3:7])
    # 是否需要添加其他奖励函数，比如说关节角度最小。
    reward_current = -distance - np.log10(0.1 * distance + 1e-4) - 5 + attitude_error + 0
    # reward_current = -distance - np.log10(0.1 * distance + 1e-4)
    return reward_current
