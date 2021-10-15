## rl-gym-solvers

In this project I explore reinforcement learning algorithms applied to the Gym Python library environments. At this moment, only a simple tabular Q-learning agent is available.

## API

Currently, two main classes are provided in the `rlsolvers` package:   
`gymTrainer` and `discreteQlearningAgent`. It is straightforward to train a new model:

```python
import gym
import rlsolvers

env = gym.make('FrozenLake8x8-v1')

# Instantiate an agent
agent = rlsolvers.discreteQlearningAgent()
# Instantiate a gymTrainer
trainer = rlsolvers.gymTrainer(env, agent)
# Train the agent
trainer.train(Nepisodes=100000, save_name="agent.pickle")
```

To test the trained agent call `trainer.test()`. To render a single episode using a loaded agent call `trainer.render_episode()` (see `show_behavior.py` for an example).


It is also possible to specify the path with a pickle containing the agent, e.g.: `trainer = gymTrainer(env, "agent.pickle")` and `trainer.test(agent="agent.pickle")`.

For more examples check out the notebook `exploreFrozenLake.ipynb`. 

## Requirements

I tested this code in two linux environments:
```
python=3.7.12
numpy=1.19.5
matplotlib=3.2.2
gym=0.17.2
pickle=4.0
```
and
```
python=3.8.5
numpy=1.19.2
matplotlib=3.3.2
gym=0.21.0
pickle=4.0
```
