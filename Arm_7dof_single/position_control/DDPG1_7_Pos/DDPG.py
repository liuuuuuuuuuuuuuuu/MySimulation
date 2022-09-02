"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

torch实现DDPG算法
"""
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)

# current_file = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取上一级目录的名字
current_file = os.getcwd()  # 获取当前目录的名字
date_save = "0612-1"
date_load = "0617"
directory_save = current_file + '/train_log/' + '/' + date_save
directory_load = current_file + '/train_log/' + '/' + date_load

if not os.path.exists(directory_save):
    os.makedirs(directory_save)
else:
    pass


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # self.action_bound = torch.FloatTensor(action_bound)

        # layer
        self.l1 = nn.Linear(state_dim, 300)
        # 初始化神经网络的参数的两种方式
        # nn.init.normal_(self.l1.weight, -0.0, 0.001)
        # nn.init.constant_(self.l1.bias, 0.001)
        # self.l1.weight.data.normal_(0.,0.001)
        # self.l1.bias.data.fill_(0.001)

        self.l2 = nn.Linear(300, 500)
        self.l3 = nn.Linear(500, 1000)
        self.l4 = nn.Linear(1000, 1500)
        self.l5 = nn.Linear(1500, 1000)
        self.l6 = nn.Linear(1000, 500)
        self.l7 = nn.Linear(500, 300)
        self.l8 = nn.Linear(300, action_dim)
        # self.l8.weight.data.normal_(-0.0, 0.001)
        # self.l8.bias.data.fill_(0.001)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = torch.tanh(self.l8(x))
        x = x*0.1


        # 对action进行放缩，实际上a in [-1,1]
        # scaled_a = x * self.action_bound
        return x


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_layer=300):
        super(Critic, self).__init__()
        self.ls = nn.Linear(state_dim, 300)
        self.la = nn.Linear(action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 500)
        self.l4 = nn.Linear(500, 1000)
        self.l5 = nn.Linear(1000, 500)
        self.l6 = nn.Linear(500, 300)
        self.l7 = nn.Linear(300, 300)
        self.l8 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = self.ls(s)
        y = self.la(a)
        x = F.relu(x + y)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        q_val = self.l8(x)
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, replacement,  explore_noise, memory_capacity=1000, gamma=0.9, lr_a=0.0001,
                 lr_c=0.0001, batch_size=2048):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_noise = explore_noise
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
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
        s = torch.FloatTensor(s).to(device)
        action_ = self.actor(s)
        action_noise = torch.FloatTensor(np.random.normal(0, scale=self.explore_noise, size=7)).to(device)
        action = action_ + action_noise
        action = action.detach().cpu()
        action = np.clip(action, -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)
        return action

    def choose_action_test(self, s):
        s = torch.FloatTensor(s).to(device)
        action = self.actor(s).detach().cpu()
        action = np.clip(action, -0.1 / 180 * np.pi, 0.1 / 180 * np.pi)
        return action

    def learn(self):
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

        # 从记忆库中采样bacth data
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
            torch.load(directory_load + "/" + "7999_actor.pth"))
        self.actor.load_state_dict(
            torch.load(directory_load + "/" + "7999_actor.pth"))
        # self.critic.load_state_dict(
        #     torch.load(directory_load + "/" + "7999_critic.pth"))
        # self.critic_target.load_state_dict(
        #     torch.load(directory_load + "/" + "7999_critic.pth"))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")
