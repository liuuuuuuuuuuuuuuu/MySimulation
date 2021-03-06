import gym
import time

import matplotlib.pyplot as plt
import numpy as np
from DDPG import DDPG
import argparse
import torch

# 设置参数变量
# (1) 声明一个parser（解析器）
parser = argparse.ArgumentParser()

# (2) 添加参数
parser.add_argument('--seed', type=int, default=616,
                    help='random seed (default: 616)')
# parser.add_argument('--test_every_steps', type=int, default=5,
#                     help='eval interval (default: 10)')
# parser.add_argument('--train_total_steps', type=int, default=10e7,
#                     help='number of total time steps to train (default: 10e5)')
parser.add_argument('--env', default='Arm-v1',
                    help='environment to train on (default: Arm-v1)(Hopper-v3)')
parser.add_argument('--MAX_EPISODES_TRAIN', default=200, type=int,
                    help='Max number of total train episodes:(default:100)')
parser.add_argument('--MAX_EPISODES_TEST', default=50, type=int,
                    help='Max number of total test episodes:(default:100)')
parser.add_argument('--MAX_EP_STEPS', default=1000, type=int,
                    help='Max number of steps per episode (default: 1000')
parser.add_argument('--explore_noise', default=0.0002, type=int,
                    help='探索随机的方差‘ (default: 0.0002')
parser.add_argument('--MEMORY_CAPACITY', default=40000, type=int,
                    help='Max number of memory_capacity (default: 20000')

parser.add_argument('--mode', default='test', type=str,
                    help="mode='train' or 'test' (default: test)")
parser.add_argument('--load', default=True, type=bool,
                    help="load model False True (default: False)")  # load model False True
parser.add_argument('--log_interval', default=50, type=int,
                    help="每多少episode保存一次神经网络")

# (3) 读取命令行参数并解析
args = parser.parse_args()

if __name__ == '__main__':

    t1 = time.time()
    REPLACEMENT = [
        dict(name='soft', tau=0.00001),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies
    RENDER = False

    # train
    env = gym.make(args.env)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    a_dim = 3
    a_bound = env.action_space.high
    a_bound = 0.01 * np.pi / 180
    agent = DDPG(state_dim=s_dim,
                 action_dim=a_dim,
                 replacement=REPLACEMENT,
                 explore_noise=args.explore_noise,
                 memory_capacity=args.MEMORY_CAPACITY)
    if args.mode == 'test':
        agent.load()
        ep_reward1 = []
        for i in range(args.MAX_EPISODES_TEST):
            s = env.reset()
            ep_reward = 0
            for j in range(args.MAX_EP_STEPS):
                env.render()
                # Add exploration noise
                action = agent.choose_action_test(s)
                # action = np.clip(np.random.normal(action, VAR), -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)  # 在动作选择上添加随机噪声

                action = np.clip(action, -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)
                joint_current = (np.array(s[:3])).ravel()
                action1 = np.zeros(3)
                action1[0] = action[0] + joint_current[0]
                action1[1] = action[1] + joint_current[1]
                action1[2] = action[2] + joint_current[2]

                s_, r, done, info = env.step(action1)

                agent.store_transition(s, action, r, s_)

                s = s_
                ep_reward += r
                if (done == True) or (j == args.MAX_EP_STEPS - 1):
                    # print(j)
                    print('Episode:', i, "done: ", done, ' Reward: %i' % int(ep_reward),
                          'Explore: %.2f' % args.explore_noise, )
                    break

    elif args.mode == 'train':
        if args.load:
            agent.load()
        ep_reward1 = []
        for i in range(args.MAX_EPISODES_TRAIN):
            s = env.reset()
            ep_reward = 0
            for j in range(args.MAX_EP_STEPS):
                # if i > 50:
                #     env.render()
                # env.render()

                # Add exploration noise
                action = agent.choose_action_train(s)
                # action = np.clip(np.random.normal(action, VAR), -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)  # 在动作选择上添加随机噪声

                action = np.clip(action, -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)
                joint_current = (np.array(s[:3])).ravel()
                action1 = np.zeros(3)
                action1[0] = action[0] + joint_current[0]
                action1[1] = action[1] + joint_current[1]
                action1[2] = action[2] + joint_current[2]

                s_, r, done, info = env.step(action1)

                agent.store_transition(s, action, r, s_)

                if agent.pointer > args.MEMORY_CAPACITY:
                    agent.learn()

                s = s_
                ep_reward += r
                if done or j == args.MAX_EP_STEPS - 1:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % args.explore_noise,
                          "done:%s" % done)
                    if ep_reward > -300:
                        RENDER = True
                    break

            ep_reward1.append(ep_reward)
            plt.plot(ep_reward1)

            if i >= 40:
                args.explore_noise *= 0.99

            if i % 50 == 49:
                plt.show()

            if i % args.log_interval == 49:
                agent.save(i)

    else:
        raise NameError("mode wrong!!!")

    print('Running time: ', time.time() - t1)
