import gym
import rlsolvers
import sys

if len(sys.argv) < 2:
    raise Exception(f"A path to a pickled agent must be specified. Use as: python {sys.argv[0]} PATH_TO_AN_AGENT.")
    
path_agent = sys.argv[1]
print("Loading "+path_agent)

env     = gym.make('FrozenLake8x8-v1')
trainer = rlsolvers.gymTrainer(env, sys.argv[1])

trainer.render_episode(dt=0.3)

