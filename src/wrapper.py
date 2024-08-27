import numpy as np
import numpy.typing as npt
from mineclip import MineCLIP
import torch
from typing import Any

class EnvWrapper:
    def __init__(self, env) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.env = env

        self.frames = 16
        self.buffer = []

        self.clip = MineCLIP("vit_base_p16_fz.v2.t2", hidden_dim=512, image_feature_dim=512, mlp_adapter_spec="v0-2.t0", pool_type="attn.d2.nh8.glusw", resolution=[160, 256]).to(self.device)
        self.clip.load_ckpt("./checkPoints/attn.pth", strict=True)

        self.prompts = [
            # "find a tree.",
            # "walk straight to the tree.",
            # "chop the tree.",
            # "collect the tree block.",

            # "I'm looking for some trees to chop.",
            # "I have to find a tree and chop it.",
            # "I want some woods, so I'm walking around to find these.",
            # "I love oak trees very much. I want to see it more.",
            # "I'm going to the forest. There are a lot of trees.",
            # "I'm a wood chopper. My job is to chop many woods."

            # "stand still",
            # "don't move",
            # "stare at a point",
            # "not look around"

            "The player chases a sheep across the screen.",
            "The sheep is in the center as the player runs after it.",
            "The player spots the sheep and begins the chase.",
            "The sheep is drectly in the middle while the player pursues it.",
            "The player follows the sheep, which is centered on the screen.",
            "The chase is on, with the sheep in the focus.",
            "The player tracks the sheep right in the center of the frame.",
            "The sheep runs ahead while the player is in hot pursuit.",
            "The player dashes after the sheep, which is prominently in view.",
            "The player's screen is focused on the sheep as they chase it."
        ]

    def reset(self) -> npt.NDArray[np.int32]:
        obs = self.env.reset()
        return self.registerObs(obs)

    def step(self, action: list[int]) -> tuple[npt.NDArray[np.int32], float, Any]:
        obs, reward, done, info = self.env.step(action)
        reward = self.calcReward()

        return (self.registerObs(obs), reward, obs["masks"])
    
    def calcReward(self):
        with torch.no_grad():
            video = torch.tensor(np.array([self.buffer])).to(self.device)
            rewards = self.clip(video, self.prompts)[0].to(self.cpu).numpy()
            return rewards.mean(None)

    def registerObs(self, obs) -> npt.NDArray[np.int32]:
        frame = obs["rgb"]

        self.buffer.insert(0, frame)
        if len(self.buffer) > self.frames: self.buffer.pop()
        elif len(self.buffer) < self.frames:
            for _ in range(self.frames - len(self.buffer)): self.buffer.insert(0, frame)

        shapedObservations = []
        for frame in self.buffer[:4]:
            shapedObservations.append(self.shapeObs(frame))

        return np.array(shapedObservations, copy=True)

    def shapeObs(self, rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        postObs = rgb[:, 0::4, 0::4]
        return np.array(postObs, copy=False)