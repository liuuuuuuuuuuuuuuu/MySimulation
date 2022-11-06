"""
torch实现PPO算法
"""
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os
from torch.distributions import MultivariateNormal
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)

# current_file = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取上一级目录的名字
current_file = os.getcwd()  # 获取当前目录的名字
date_save = "0925"
date_load = "0925"
directory_save = current_file + '/train_log/' + '/' + date_save
directory_load = current_file + '/train_log/' + '/' + date_load

if not os.path.exists(directory_save):
    os.makedirs(directory_save)
else:
    pass

Transition = namedtuple('Transition', ['state', 'aciton', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # self.action_bound = torch.FloatTensor(action_bound)

        # layer
        self.l1 = nn.Linear(state_dim, 128)
        # 初始化神经网络的参数的两种方式
        # nn.init.normal_(self.l1.weight, -0.0, 0.001)
        # nn.init.constant_(self.l1.bias, 0.001)
        # self.l1.weight.data.normal_(0.,0.001)
        # self.l1.bias.data.fill_(0.001)

        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.mu_head = nn.Linear(128, action_dim)
        self.sigma_head = nn.Linear(128, action_dim)

        # self.l8.weight.data.normal_(-0.0, 0.001)
        # self.l8.bias.data.fill_(0.001)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        mu = torch.tanh(self.mu_head(x))
        sigma = torch.tanh(self.sigma_head(x))

        # x = x*1/180*np.pi
        # 对action进行放缩，实际上a in [-1,1]
        # scaled_a = x * self.action_bound
        return mu, sigma


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_layer=300):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        value = self.l5(x)
        return value


# Deep Deterministic Policy Gradient
class PPO(object):
    def __init__(self, state_dim, action_dim, replacement, clip_param, memory_capacity=1000, gamma=0.9, lr_a=0.001,
                 lr_c=0.001, batch_size=2048):
        super(PPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.clip_param = clip_param
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        # self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.memory = []
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action_train(self, s):
        Pos_Target_EE = s[10:13]
        distance = np.linalg.norm(Pos_Target_EE)
        delta_Pos = Pos_Target_EE / distance
        s_norm = np.concatenate((s[0:7], delta_Pos[:], s[10:]))

        s = torch.FloatTensor(s_norm).to(device)
        with torch.no_grad():
            mu, sigma = self.actor(s)

        dist = MultivariateNormal(mu, np.diag(sigma)).cpu()
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        action = action.detach().cpu()
        action = np.clip(action, -10 / 180 * np.pi, 10 / 180 * np.pi)

        return action, action_log_prob

    def choose_action_test(self, s):
        Pos_Target_EE = s[10:13]
        distance = np.linalg.norm(Pos_Target_EE)
        delta_Pos = Pos_Target_EE / distance
        s_norm = np.concatenate((s[0:7], delta_Pos[:], s[10:]))

        s = torch.FloatTensor(s_norm).to(device)
        with torch.no_grad():
            mu, sigma = self.actor(s)

        dist = MultivariateNormal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        action = action.detach().cpu()
        action = np.clip(action, -10 / 180 * np.pi, 10 / 180 * np.pi)

        return action, action_log_prob

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def learn(self):
        self.training_step += 1

        bm = self.sample()
        state = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
        action = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        reward = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
        next_state = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)
        # old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)


        reward = (reward - reward.mean()) / (reward.std() + 1e-10)

        with torch.no_grad():
            target_v = reward + self.gamma * self.critic_net(next_state)
        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity), self.batch_size, True)):
                # epoch iteration, PPO core!!!
                mu, sigma = self.actor_net(state[index])
                n = MultivariateNormal(mu, sigma)
                action_log_prob = n.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob)

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

            del self.buffer[:]


        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            # for al in a_layers:
            #     a = self.actor.state_dict()[al[0] + '.weight']
            #     al[1].weight.data.mul_((1 - tau))
            #     al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
            #     al[1].bias.data.mul_((1 - tau))
            #     al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            # for cl in c_layers:
            #     cl[1].weight.data.mul_((1 - tau))
            #     cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
            #     cl[1].bias.data.mul_((1 - tau))
            #     cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1

        # 从记忆库中采样bacth data :batch memory
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:]).to(device)

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        a_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        # for param_tensor in self.actor.state_dict():
        #     param_tensor
        #     self.actor.state_dict()[param_tensor].size()
        #     print(param_tensor, "\t", self.actor.state_dict()[param_tensor].size())

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1


    def save(self, i):
        torch.save(self.actor.state_dict(), directory_save + "/" + str(i) + '_actor.pth')
        torch.save(self.critic.state_dict(), directory_save + "/" + str(i) + '_critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor_target.load_state_dict(
            torch.load(directory_load + "/" + "499_actor.pth"))
        self.actor.load_state_dict(
            torch.load(directory_load + "/" + "499_actor.pth"))
        # self.critic.load_state_dict(
        #     torch.load(directory_load + "/" + "7999_critic.pth"))
        # self.critic_target.load_state_dict(
        #     torch.load(directory_load + "/" + "7999_critic.pth"))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")
