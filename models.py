import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_dim = 8000 # Unsure
        hidden_dim = 1000
        num_moods = 4
        self.main = nn.Sequential
        (
            nn.Linear(input_dim, hidden_dim ,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_moods, bias=True),
            nn.Softmax()
        )
    
    def forward(self, x):
        out = self.main(x)
        return out