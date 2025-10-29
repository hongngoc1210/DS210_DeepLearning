import torch
import torch.nn as nn
from torch.nn import functional as F

class Perceptron1Layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super().__init__()
        
        w, h = image_size
        self.linear = nn.Linear(
            in_features=w*h,
            out_features=num_labels
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x