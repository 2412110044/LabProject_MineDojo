import minedojo
import torch
from time import time
import numpy as np

from wrapper import EnvWrapper
from replayBuffer import ReplayBuffer
from agent import Agent

env = EnvWrapper(minedojo.make(task_id="harvest_milk", image_size=(160, 256)))
buffer = ReplayBuffer()
agent = Agent('checkPoints_1724222014/trainNet.pth')

env.reset()
obs, reward, masks = env.step(np.zeros(8))

for i in range(10000):
    action = agent.selectAction(obs, masks)
    next_obs, reward, masks = env.step(action)

    print("reward:", reward)

    buffer.add(obs, next_obs, action, reward, masks)
    obs = next_obs

    if i > 256:
        loss = agent.train(*buffer.sample(256))

        if i%10 == 0: print("loss:", loss)

    if i % 250 == 249:
        torch.save(agent.trainNet.state_dict(), 'checkPoints/trainNet_' + str(int(time())) + '.pth')

torch.save(agent.trainNet.state_dict(), 'checkPoints/trainNet_' + str(int(time())) + '.pth')