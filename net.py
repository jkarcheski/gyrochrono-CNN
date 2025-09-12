import torch.nn as nn

class StellarClusterCNN(nn.Module):
    def __init__(self):  # <-- corrected here
        super(StellarClusterCNN, self).__init__()  # <-- and here

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x