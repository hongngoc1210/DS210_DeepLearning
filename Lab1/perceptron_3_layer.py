import torch
import torch.nn as nn
from torch.nn import functional as F

class Perceptron3Layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()
        
        w, h = image_size
        self.linear1 = nn.Linear(
            in_features=w*h,
            out_features=128
        )
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=64
        )
        self.linear3 = nn.Linear(
            in_features=64,
            out_features=num_labels
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.log_softmax(x, dim=1)
        return x