import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv3d(4, 16, 8, 3, padding=[3, 0, 0], padding_mode='replicate')
        self.conv2 = nn.Conv3d(16, 32, 4, 2, padding=[2, 0, 0], padding_mode='replicate')
        self.conv3 = nn.Conv3d(32, 64, 3, 1, padding=[1, 0, 0], padding_mode='replicate')

        hiddenSize = 128
        self.fc1 = nn.Linear(2 * 6 * 64, hiddenSize)

        self.dueling_value = nn.Linear(hiddenSize, 1)

        self.dueling_action0 = nn.Linear(hiddenSize, 3)
        self.dueling_action1 = nn.Linear(hiddenSize, 3)
        self.dueling_action2 = nn.Linear(hiddenSize, 4)
        self.dueling_action3 = nn.Linear(hiddenSize, 4)
        self.dueling_action4 = nn.Linear(hiddenSize, 4)

        self.dueling_actions = [
            self.dueling_action0, self.dueling_action1, self.dueling_action2, self.dueling_action3, self.dueling_action4
        ]

    def forward(self, x: torch.FloatTensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        v = self.dueling_value(x)
        qValues = []
        for dueling_action in self.dueling_actions:
            action = dueling_action(x)
            qValues.append(action - action.mean(dim=1, keepdim=True) + v)

        return qValues