import minedojo
import time
import matplotlib.pyplot as plt

env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))

obs = env.reset()

interval = 1
lastTime = time.time()

img = None
for i in range(1):
    action = env.action_space.no_op()
    next_obs, reward, done, info = env.step(action)

    print(reward, done)

    if i == 0: img = next_obs["rgb"]

    waitTime = max(0, interval - (time.time() - lastTime))
    time.sleep(waitTime)

plt.imshow(img[:, ::4, ::4].transpose(1, 2, 0))
plt.show()