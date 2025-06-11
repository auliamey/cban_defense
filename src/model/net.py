import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)  # output: 16x16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, 2)  # output: 8x8

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, return_logits_only=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32x32 -> 16x16
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))              # 8x8
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        logits = self.fc2(x)
        
        if return_logits_only:
            return logits
        return F.log_softmax(logits, dim=1)
