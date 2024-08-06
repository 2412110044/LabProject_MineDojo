import numpy as np
import numpy.typing as npt
import torch

from model import NeuralNetwork

class Agent:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainNet = NeuralNetwork().to(self.device)

    def select_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        action = np.zeros((8,))

        obsTensor = torch.FloatTensor([obs]).to(self.device)
        qValues = self.trainNet(obsTensor)

        action[0] = qValues[0].max(1)[1]
        action[1] = qValues[1].max(1)[1]
        action[2] = qValues[2].max(1)[1]
        action[3] = qValues[3].max(1)[1] * 6
        action[4] = qValues[4].max(1)[1] * 6

        return action