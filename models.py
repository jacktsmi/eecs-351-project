import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        num_moods = 4
        self.main = nn.Sequential(
            nn.Linear(input_dim, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 200, bias=True),
            nn.ReLU(),
            nn.Linear(200, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, num_moods, bias=True)
        )
    
    def forward(self, x):
        out = self.main(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, input_dim):
        super(ConvNet, self).__init__()
        num_moods = 4
        self.main = nn.Sequential(
            nn.Conv1d(input_dim, 100, 10),
            nn.ReLU(),
            nn.Linear(100, num_moods, bias=True)
        )
    
    def forward(self, x):
        out = self.main(x)
        return out

def weights_init(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight.data, 0.0, 0.01)
        nn.init.constant_(layer.bias.data, 0.0)