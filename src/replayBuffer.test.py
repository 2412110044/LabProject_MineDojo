import numpy as np

from replayBuffer import ReplayBuffer

buffer = ReplayBuffer()
buffer.add(np.array([0]), np.array([1]), np.array([0, 0, 0, 0, 0, 0, 0, 0]), 0.2, {"action_type": [True, True, True, True, False, False, False, False]})
print(buffer.sample(4))

buffer.add(np.array([[0,1], [2, 3]]), np.array([[1,2], [3,4]]), np.array([1, 2, 0, 0, 0, 0, 0, 0]), 0.8, {"action_type": [True, True, True, True, False, False, False, True]})
print(buffer.sample(8))