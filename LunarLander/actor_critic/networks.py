import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        x = F.selu(self.fc3(x), inplace=True)
        x = F.selu(self.fc4(x), inplace=True)
        return F.softmax(self.fc5(x), dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        x = F.selu(self.fc3(x), inplace=True)
        x = F.selu(self.fc4(x), inplace=True)
        return self.fc5(x)