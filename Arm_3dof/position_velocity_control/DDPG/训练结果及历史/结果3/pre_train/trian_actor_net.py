import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


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
        self.l4 = nn.Linear(1000, 500)
        self.l5 = nn.Linear(500, 300)
        self.l6 = nn.Linear(300, action_dim)
        # self.l8.weight.data.normal_(-0.0, 0.001)
        # self.l8.bias.data.fill_(0.001)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = torch.tanh(self.l6(x))


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
        self.critic.load_state_dict(
            torch.load("pretrain_critic.pth"))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


actor = Actor(state_dim=5, action_dim=3)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)

s = input_data = np.load("/home/liujian/桌面/single_arm_3dof_torch/DDPG/pre_train/input_data.npy")
y = np.load("/home/liujian/桌面/single_arm_3dof_torch/DDPG/pre_train/output_data.npy")
y = y.reshape(10000, 3)

s = torch.FloatTensor(s)
y = torch.FloatTensor(y)
y = np.clip(y, -0.3*np.pi/180, -0.3*np.pi/180)
# y1 = actor(s)

loss1 = []
for epoch in range(3000):
    optimizer.zero_grad()
    y_pred = actor(s)

    loss = criterion(y_pred, y)
    print(epoch, loss)
    loss1.append(loss.data)
    plt.plot(loss1)
    if epoch % 1000 ==999:
       plt.show()

    loss.backward()
    optimizer.step()

torch.save(actor.state_dict(),'pretrain_actor.pth')


