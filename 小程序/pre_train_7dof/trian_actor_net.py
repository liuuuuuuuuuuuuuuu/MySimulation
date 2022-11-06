import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import random
from torch.autograd import Variable

import torch.utils.data as Data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Actor：输入是state，输出的是一个确定性的action
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         # self.action_bound = torch.FloatTensor(action_bound)
#
#         # layer
#         self.l1 = nn.Linear(state_dim, 20)
#         # 初始化神经网络的参数的两种方式
#         # nn.init.normal_(self.l1.weight, -0.0, 0.001)
#         # nn.init.constant_(self.l1.bias, 0.001)
#         # self.l1.weight.data.normal_(0.,0.001)
#         # self.l1.bias.data.fill_(0.001)
#
#         self.l2 = nn.Linear(20, 20)
#         self.l3 = nn.Linear(20, 20)
#         self.l4 = nn.Linear(20, 20)
#         self.l5 = nn.Linear(20, 20)
#         self.l6 = nn.Linear(20, 20)
#         self.l7 = nn.Linear(20, 20)
#         self.l8 = nn.Linear(20, action_dim)
#         # self.l8.weight.data.normal_(-0.0, 0.001)
#         # self.l8.bias.data.fill_(0.001)
#
#     def forward(self, s):
#         x = F.relu(self.l1(s))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         x = F.relu(self.l5(x))
#         x = F.relu(self.l6(x))
#         x = F.relu(self.l7(x))
#         x = torch.tanh(self.l8(x))
#         x = x*1
#
#
#         # 对action进行放缩，实际上a in [-1,1]
#         # scaled_a = x * self.action_bound
#         return x

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
        self.l5 = nn.Linear(128, action_dim)
        # self.l8.weight.data.normal_(-0.0, 0.001)
        # self.l8.bias.data.fill_(0.001)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = torch.tanh(self.l5(x))
        x = x*5

        # 对action进行放缩，实际上a in [-1,1]
        # scaled_a = x * self.action_bound
        return x

    def save(self):
        torch.save(self.actor.state_dict(),'pretrain_actor.pth')
        torch.save(self.critic.state_dict(), 'pretrain_critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        # self.actor.load_state_dict(torch.load(directory + '_actor.pth'))
        # self.critic.load_state_dict(torch.load(directory + '_critic.pth'))
        self.actor.load_state_dict(
            torch.load("pretrain_actor.pth"))
        # self.critic.load_state_dict(
        #     torch.load("pretrain_critic.pth"))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


actor = Actor(state_dim=18, action_dim=7).to(device)
# actor.load_state_dict(
#             torch.load("/home/liujian/桌面/single_arm_3dof_torch/DDPG/pre_train_7dof/pretrain_actor.pth"))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)

# s = np.load("input_data.npy")
# y = np.load("output_data.npy")
# y = y.reshape(200000, 7)
#
# s = torch.FloatTensor(s).to(device)
#
# y = np.clip(y, -0.04, 0.04)
# y = torch.FloatTensor(y).to(device)

input_data = torch.from_numpy(np.load("input_data.npy").reshape(500000,18))
output_data = torch.from_numpy(np.load("output_data.npy").reshape(500000,7))

torch_dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=500000,
                         shuffle=True,
                         num_workers=8)


loss1 = []
for epoch in range(1000):
    for step, data in enumerate(loader):
        # 将数据从 train_loader 中读出来,一次读取的样本数是32个
        inputs, outputs = data
        # 将这些数据转换成Variable类型
        inputs, outputs = Variable(inputs), Variable(outputs)

        # print(input.size())
        # print("epoch：", epoch, "的第", step, "个inputs", inputs.data.size(), "labels", outputs.data.size())
        # print(inputs)

        s = torch.tensor(inputs, dtype=torch.float).to(device)
        y = torch.tensor(outputs, dtype=torch.float).to(device)

        optimizer.zero_grad()
        y_pred = actor(s)

        loss = criterion(y_pred, y)
        print(epoch, loss)

        loss1.append(loss.detach().cpu())
        plt.plot(loss1)
        if (epoch+1) % 200 == 0:
            plt.show()

        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            np.save("loss", loss1)
            torch.save(actor.state_dict(), str(epoch) + 'pretrain_actor.pth')
