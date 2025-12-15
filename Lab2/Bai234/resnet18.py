import torch
from torch.nn import functional as F
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=stride,
            padding=1
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,7),
            stride=2,
            padding=3
        )
        
        self.initial_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        
        self.layer1 = self.make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self.make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self.make_layer(256, 512, num_blocks=2, stride=2)
        
        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(
            in_features=256, 
            out_features=num_classes
        )
        
        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,1),
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResnetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResnetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, images: torch.Tensor):
        out = self.initial_conv(images)
        out = self.initial_bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.average_pool(out)
        out = out.squeeze(-1).squeeze(-1)
        out = self.fc(out)
        
        return out 
        