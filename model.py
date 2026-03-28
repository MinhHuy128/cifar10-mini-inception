import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1), nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch3x3_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch3x3), nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch5x5_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5), nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5), nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)

class MiniInceptionNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
        self.incept1a = InceptionBlock(96, 32, 32, 48, 8, 16, 16)  
        self.incept1b = InceptionBlock(112, 64, 48, 64, 8, 16, 32)  
        self.pool1 = nn.MaxPool2d(2)  
        self.incept2a = InceptionBlock(176, 96, 48, 96, 12, 24, 32)  
        self.incept2b = InceptionBlock(248, 128, 64, 112, 16, 32, 32) 
        self.pool2 = nn.MaxPool2d(2)  
        self.incept3a = InceptionBlock(304, 160, 64, 128, 16, 32, 32) 
        self.incept3b = InceptionBlock(352, 192, 80, 160, 16, 32, 32) 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(416, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incept1a(x); x = self.incept1b(x); x = self.pool1(x)
        x = self.incept2a(x); x = self.incept2b(x); x = self.pool2(x)
        x = self.incept3a(x); x = self.incept3b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x