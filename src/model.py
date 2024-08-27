import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv3d(4, 16, 8, 3, padding=[3, 0, 0], padding_mode='replicate')
        self.conv2 = nn.Conv3d(16, 32, 4, 2, padding=[2, 0, 0], padding_mode='replicate')
        self.conv3 = nn.Conv3d(32, 64, 3, 1, padding=[1, 0, 0], padding_mode='replicate')
        self.conv4 = nn.Conv3d(64, 128, 3, 1, padding=[1, 1, 0], padding_mode='replicate')

        hiddenSize = 128
        self.fc1 = nn.Linear(1 * 2 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, hiddenSize)

        self.dueling_value = nn.Linear(hiddenSize, 1)
        self.dueling_action = nn.Linear(hiddenSize, 3 + 3 + 4 + 6 + 6 + 8)

    def forward(self, x: torch.FloatTensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))

        action = self.dueling_action(x)
        x = action - action.mean(dim=1, keepdim=True) + self.dueling_value(x)

        return x