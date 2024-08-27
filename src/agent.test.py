import numpy as np

from agent import Agent

def testSelectAction():
    agent = Agent()

    obs = np.random.randint(0, 256, [4, 3, 40, 64])
    for _ in range(10): print(agent.selectAction(obs, {"action_type": [True, True, True, True, False, False, False, False]}))


def testTrain():
    agent = Agent()

    batchSize = 16
    train_obs = np.random.randint(0, 256, [batchSize, 4, 3, 40, 64])
    train_nextObs = np.random.randint(0, 256, [batchSize, 4, 3, 40, 64])

    train_actions = []
    for _ in range(batchSize):
        train_actions.append([
            np.random.randint(0, 3),
            np.random.randint(0, 3),
            np.random.randint(0, 4),
            np.random.randint(0, 4),
            np.random.randint(0, 4),
            0,
            0,
            0
        ])
    train_actions = np.array(train_actions)

    train_rewards = np.random.rand(batchSize)

    train_masks = []
    for _ in range(batchSize):
        train_masks.append({"action_type": [True, True, True, True, False, False, False, False]})

    for _ in range(10):
        print(agent.train(train_obs, train_nextObs, train_actions, train_rewards, train_masks))
        for _ in range(0): agent.train(train_obs, train_nextObs, train_actions, train_rewards, train_masks)

testSelectAction()
testTrain()