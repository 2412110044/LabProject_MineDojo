import minedojo
import time
import math
from random import random

env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))

obs = env.reset()

interval = 0.1
lastTime = time.time()

for _ in range(100000):
    action = env.action_space.no_op()
    action[0] = math.floor(random()*3)
    action[1] = math.floor(random()*3)
    action[2] = math.floor(random()*4)
    action[4] = math.floor(random()*3) + 10

    next_obs, reward, done, info = env.step(action)

    print(reward, done)

    waitTime = max(0, interval - (time.time() - lastTime))
    lastTime = time.time()
    time.sleep(waitTime)