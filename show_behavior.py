import gym
import rlsolvers
import sys

path_agent = sys.argv[1]
print("Loading "+path_agent)

env     = gym.make('FrozenLake8x8-v1')
trainer = rlsolvers.gymTrainer(env, sys.argv[1])

trainer.render_episode(dt=0.3)

