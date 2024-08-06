import minedojo

from wrapper import EnvWrapper
from agent import Agent

env = EnvWrapper(minedojo.make(task_id="harvest_milk", image_size=(160, 256)))
agent = Agent()

obs = env.reset()

for i in range(10000):
    action = agent.select_action(obs)
    next_obs, reward, done, info = env.step(action)
