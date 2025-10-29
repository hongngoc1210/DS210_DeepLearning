import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5,5),  
            stride=1,
            padding=(2,2)
        )
        
        self.pooling1 = nn.AvgPool2d(
            kernel_size = (2,2),
            stride=(2,2),
            padding=(0,0)
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            padding=0
        )
        
        self.pooling2 = nn.AvgPool2d(
            kernel_size=(2,2),
            stride=(2,2),
            padding=(0,0)
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
            padding=(0,0)
        )
        
        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )
        
        self.output = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward (self, images: torch.Tensor): 

        if images.ndim == 3:
            images = images.unsqueeze(1) 
        #input shape: (batch_size, 1, 28, 28)
        features = F.relu(self.conv1(images))   # 32, 6, 28, 28
        features = self.pooling1(features) # 32, 6, 14, 14
        features = F.relu(self.conv2(features)) # 32, 16, 10, 10
        features = self.pooling2(features) # 32, 16, 5, 5
        features = F.relu(self.conv3(features)) # 32, 120, 1, 1
        features = torch.flatten(features, 1) # 32, 120
        features = F.relu(self.fc(features)) # 32, 84
        logits = self.output(features) # 32, 10
        
        return logits
  