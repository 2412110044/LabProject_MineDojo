import numpy as np
import numpy.typing as npt

class EnvWrapper:
    def __init__(self, env) -> None:
        self.env = env

        self.frames = 4
        self.buffer = []


    def reset(self):
        obs = self.env.reset()
        return self.getObs(obs)

    def step(self, action: list[int]):
        obs, reward, done, info = self.env.step(action)

        return (self.getObs(obs), reward, done, info)

    def getObs(self, obs):
        shaped = self.shapeObs(obs)

        self.buffer.append(shaped)
        if len(self.buffer) > self.frames: self.buffer.pop(0)

        state = []
        for c in range(4):
            if len(self.buffer) <= c: state.append(shaped)
            else: state.append(self.buffer[c])

        return state

    def shapeObs(self, obs):
        # obs["rgb"].shape = (3, 160, 256)
        # postObs.shape = (3, 40, 64)

        rgb: npt.NDArray[np.float64] = obs["rgb"]
        postObs = rgb[:, 0::4, 0::4]

        x = np.array(postObs, copy=True)

        return x