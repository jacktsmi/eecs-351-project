import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(8000, 4000 ,bias=True),# Unsure on 8000 dimension because I think mfcc outputs 2d array
            nn.ReLU(),
            nn.Linear(4000, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 200, bias=True),
            nn.ReLU(),
            nn.Linear(200, 4, bias=True)
        )
    
    def forward(self, x):
        out = self.main(x)
        return out