import numpy as np
import numpy.typing as npt
from typing import Any

class ReplayBuffer:
    def __init__(self) -> None:
        self.storage = []

    def add(self, state: npt.NDArray[np.int32], nextState: npt.NDArray[np.int32], action: npt.NDArray[np.int32], reward: float, mask: Any):
        self.storage.append((state, nextState, action, reward, mask))

    def sample(self, batchSize: int) -> tuple[list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]], list[npt.NDArray[np.int32]], list[float]]:
        indexes = np.random.randint(0, len(self.storage), size=batchSize)

        states, nextStates, actions, rewards, masks = [], [], [], [], []

        for i in indexes:
            state, nextState, action, reward, mask = self.storage[i]

            states.append(state)
            nextStates.append(nextState)
            actions.append(action)
            rewards.append(reward)
            masks.append(mask)

        return np.array(states), np.array(nextStates), np.array(actions), np.array(rewards), masks