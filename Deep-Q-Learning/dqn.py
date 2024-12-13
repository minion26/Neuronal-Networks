import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) # hidden layer
        self.fc2 = nn.Linear(hidden_dim, action_dim) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # x = state, activation function
        return self.fc2(x) # set to the output layer for the Q-values



if __name__ == '__main__' :
    state_dim = 12 # 12 informations : last pipe, next pipe, bird position
    action_dim = 2 # no action 0.1 - bird flapping 0.9 reward
    net = DQN(state_dim, action_dim,128)
    state = torch.randn(1, state_dim) # dummy values , [1,12] pytorch uses 1 dim for batching
    output = net(state)
    print(output)