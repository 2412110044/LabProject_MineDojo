import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from typing import Any

from model import NeuralNetwork

class Agent:
    def __init__(self, param_path='') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.trainNet = NeuralNetwork().to(self.device)
        if os.path.isfile(param_path):
            self.trainNet.load_state_dict(torch.load(param_path, map_location=self.device.type, weights_only=True))

        self.targetNet = NeuralNetwork().to(self.device)
        self.targetNet.load_state_dict(self.trainNet.state_dict())

        self.optimizer = optim.Adam(self.trainNet.parameters(), lr=0.01)

        self.timeSteps = 0

    def selectAction(self, obs: npt.NDArray[np.int32], masks: Any) -> npt.NDArray[np.int32]:
        obsTensor = torch.tensor(np.array([obs])).to(self.device, torch.float32)

        epsilon = 0.6*(1 - self.timeSteps/10000) + 0.01*self.timeSteps/10000
        if self.timeSteps % 100 == 0: print('ε: ', epsilon)
        action = self.generateAction(obsTensor, [masks], np.random.random() < epsilon)[0]
        self.timeSteps += 1

        return action.to(self.cpu).numpy()

    
    def train(self, states: list[npt.NDArray[np.int32]], nextStates: list[npt.NDArray[np.int32]], actions: list[npt.NDArray[np.int32]], rewards: list[float], masks: list[Any]):
        discount = 0.99

        states = torch.tensor(states).to(self.device, dtype=torch.float32)
        actions = torch.tensor(actions).to(self.device)
        nextStates = torch.tensor(nextStates).to(self.device, dtype=torch.float32)

        rewards = torch.tensor(rewards).to(self.device, dtype=torch.float32)
        rewards = self.batchNorm(rewards)
        
        trainQ = self.gatherQ_value(self.trainNet(states), actions)
        with torch.no_grad():
            nextActions = self.generateAction(nextStates, masks)
            targetNetQ = rewards + discount * self.gatherQ_value(self.targetNet(nextStates), nextActions)

        loss = F.mse_loss(trainQ, targetNetQ)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.trainNet.parameters():
            if param.grad != None: param.grad.data.clamp(-1, 1)
            else: raise
        self.optimizer.step()

        self.updateTargetNetwork_soft()
        return loss.detach().cpu().numpy()
    
    def batchNorm(self, batch: torch.Tensor) -> torch.Tensor:
        mean = batch.mean(None)
        dispersion = ((batch - mean)**2).mean(None)
        return (batch - mean)/torch.sqrt(dispersion + 1e-5)
    
    def updateTargetNetwork_soft(self):
        tau = 0.001
        for targetVar, var in zip(self.targetNet.parameters(), self.trainNet.parameters()):
            targetVar.data.copy_((1 - tau) * targetVar.data + tau * var.data)

    def generateAction(self, obs: torch.FloatTensor, masks: list[Any], isRandom=False):
        qValue = self.trainNet(obs)
        if isRandom: qValue = torch.rand(qValue.size()).to(self.device)

        masks_action_type = []
        for mask in masks:
            masks_action_type.append(mask["action_type"])
        masks_action_type = torch.tensor(np.array(masks_action_type)).to(self.device)

        actions = torch.zeros(len(obs), 8).to(self.device, dtype=torch.int64)
        actions[:, 0] = qValue[:, 0:3].argmax(1, keepdim=True).transpose(1, 0)
        actions[:, 1] = qValue[:, 3:6].argmax(1, keepdim=True).transpose(1, 0)
        actions[:, 2] = qValue[:, 6:10].argmax(1, keepdim=True).transpose(1, 0)
        actions[:, 3] = (6 + qValue[:, 10:16].argmax(1, keepdim=True) * 2).transpose(1, 0)
        actions[:, 4] = (6 + qValue[:, 16:22].argmax(1, keepdim=True) * 2).transpose(1, 0)
        actions[:, 5] = (qValue[:, 22:30] * masks_action_type).argmax(1, keepdim=True).transpose(1, 0)

        return actions

    def gatherQ_value(self, qMaps: torch.FloatTensor, actions: list[int]) -> list[int]:
        indexes = []
        for action in actions:
            indexes.append(self.actionToIndex(action))
        indexes = torch.tensor(indexes).to(self.device, dtype=torch.int64)

        return qMaps.gather(1, indexes).sum(1)

    def actionToIndex(self, action: list[int]):
        index = [0] * 5

        index[1] = action[1] + 3
        index[2] = action[2] + 6
        index[3] = (action[3] / 3 - 6) + 10
        index[4] = (action[4] / 3 - 6) + 14

        return index