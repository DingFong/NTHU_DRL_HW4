import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # self.linear4 = nn.Linear(hidden_dim*2, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        # self.linear7 = nn.Linear(hidden_dim, hidden_dim*2)
        # self.linear8 = nn.Linear(hidden_dim*2, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.relu(self.linear6(x2))
        
        # x1 = F.relu(self.linear1(xu))
        # x1 = F.relu(self.linear2(x1))
        # x1 = F.relu(self.linear3(x1))
        # x1 = self.linear4(x1)

        # x2 = F.relu(self.linear5(xu))
        # x2 = F.relu(self.linear6(x2))
        # x2 = F.relu(self.linear7(x2))
        # x2 = self.linear8(x2)

        return x1, x2