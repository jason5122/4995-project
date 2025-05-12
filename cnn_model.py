import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()

        self.feature_size = input_size // (2**3)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Linear(self.feature_size * self.feature_size * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3a = nn.Linear(128, 4)
        self.fc3b = nn.Linear(128, 88)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        super_out = self.fc3a(x)
        sub_out = self.fc3b(x)
        return super_out, sub_out
