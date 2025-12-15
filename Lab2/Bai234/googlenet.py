import torch
import torch.nn as nn
from torch.nn import functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()
        # Nhánh 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 2: 1x1 conv -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 3: 1x1 conv -> 5x5 conv
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # Nhánh 4: 3x3 maxpool -> 1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Nối kết quả của 4 nhánh lại với nhau theo chiều kênh (dimension 1)
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
    
class GoogleNet(nn.Module):
    def __init__(self, num_classes: int = 1000): # ImageNet có 1000 lớp
        super().__init__()
        
        # Stem
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Lớp conv2 có depth=2
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Các khối Inception được lặp lại
        # Tham số (in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32) # out: 256
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64) # out: 480
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Nhóm inception 4 (lặp 5 lần)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64) # out: 512
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # out: 512
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64) # out: 512
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64) # out: 528
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128) # out: 832

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Nhóm inception 5 (lặp 2 lần)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128) # out: 832
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128) # out: 1024

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # Sửa in_features thành 1024
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Làm phẳng tensor
        x = self.dropout(x)
        x = self.classifier(x)
        return x