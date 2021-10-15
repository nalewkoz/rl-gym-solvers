"""This is the agents module. 

It contains agents (classes) solving reinforcement learning tasks (environments). 
Currently, designed to be used with discrete spaces of the Gym package.

"""

import numpy as np

# ------------RL agent (tabular Q-learning)-----------------
class discreteQlearningAgent:
    """A class representing an agent solving the task defined in the OpenAI Gym FrozenLake8x8-v0 (or -v1) environment. 

    Attributes
    ----------
    gamma: the reward discount factor.
    alpha0, alpha_tau: the learning rate is scheduled as: alpha = alpha0/(1 + t/alpha_tau), 
        where t is the number of episodes observed so far.
    beta0, beta_tau: the inverse temperature of the softmax policy is scheduled as: beta = beta0 + t/beta_tau, 
        where t is the number of episodes observed so far.
    random_beta: if True, beta_tau is ignored and beta is generated randomly from the exponential 
        distribution with <beta>=beta_0.
    Q: Q-function lookup table.
    s: current state of the agent.
    a: last action taken by the agent.
    
    Methods
    -------
    new_episode:
        Resets attributes representing state (s) and action (a). Should be 
        called at the beginning of each episode.
    select_action:
        Selects action.
    update:
        Update the state and Q-function (observation + learning).
    set_alpha:
        Sets a new value of the learning rate (alpha).
    set_beta:
        Sets a new value of the inverse temperature (beta).
    
    Comments
    --------
    Currently, only tabular Q-learning is implemented. This is fine if the map is fixed, but 
    will not generalize to new maps. A more powerful representation with a functional 
    approximation (ConvNet?) is needed to solve a more general (and much more interesting!) 
    problem of an environment in which the map is randomly generated in each episode.

    This implementation is not very general. In particular, the number of states and actions 
    is not inferred from the environment and as such this class by default only deals with 
    the specific environment it was designed for (FrozenLake8x8-v0). 
    However, the user can set Nstates and Nactions during the creation of an object, 
    which should allow this class to handle any Discrete action and state Space from the Gym (not tested).

    It is also possible to write a more generic code using Spaces: env.observation_space and env.action_space,
    see: https://gym.openai.com/docs/#spaces.

    A more elegant way would be to introduce an abstract class of an agent ("baseAgent") (e.g., using abc).
    That would allow us to check if an object 'agent' is an instance of the class baseAgent. 
    Due to limited time I won't take this approach here.
    """

    def __init__(self, gamma=0.99, alpha0=0.1, alpha_tau = 2000, beta0=3, beta_tau = 1e9, random_beta = True, state=0, Qinit=1, Nstates = 64, Nactions = 4):
        """Initialize the agent. """
        
        # Initialize tabular Q function.
        # Shape should be (# of states, # of actions) = (64, 4). 
        self.Q  = np.zeros( (Nstates, Nactions) ) + Qinit # Start with an optimistic agent (to encourage exploration). 

        # Initialize learning parameters.
        self.gamma     = gamma
        self.alpha     = alpha0
        self.alpha0    = alpha0
        self.alpha_tau = alpha_tau
        
        # Initialize action selection parameters.
        self.beta0       = beta0
        self.beta_tau    = beta_tau
        self.random_beta = random_beta

        # Set variables that should be reset at the beginning of each episode.
        self.new_episode(episode = 0) 

    def new_episode(self, episode=None, state=0):
        # Initialize the state.
        self.s  = state
        # Initialize the action variable (None, because no action was taken so far)
        self.a  = None
        # Either add 1 to the episode number, or set it externally
        if episode is None:
            self.episode += 1
            episode = self.episode
        else:
            self.episode = episode
        # Set a new value of alpha.
        self.set_alpha(step = episode)
        # Set a new value of beta.
        self.set_beta(step = episode, random=self.random_beta)

    def select_action(self, greedy=False):
        """Select and return an action.

        Available policies:
        softmax and greedy. 

        Parameters:
        s -- current state
        Q -- tabular Q function (Q[state,action])
        beta -- inverse temperature of the softmax function
        greedy (default False) -- if True, a deterministic (greedy with respect to Q) selection is used (effectively beta=infinity) 
        
        Returns: an integer (action)
        """ 
        if greedy:
            self.a = np.argmax(self.Q[self.s,:]) 
            return self.a
        
        probs = np.exp(self.beta*self.Q[self.s,:])
        probs = probs/np.sum(probs)

        self.a = np.random.choice(len(probs), p=probs) 
        return self.a
    
    def update(self, obs, r, done, learn = True):
        """Update the state of the agent as well as the Q-function estimator (if learn==True).
        
        The agent must have taken an action before.
        """
        if learn:
            if self.a is None:
                raise Exception("Cannot update because the agent did not take any action so far. Call select_action() first.")
            # Update the Q function using the TD error.
            old_prediction = self.Q[self.s, self.a] 
            if done:
                # There will not be any next step here, so we should not 
                # bootstrap (add our prediction of future rewards).
                new_prediction = r 
            else:
                new_prediction = r + self.gamma*np.max(self.Q[obs,:])

            # Update the action-value function
            self.Q[self.s, self.a] += self.alpha*(new_prediction - old_prediction)

        # Update the state.
        self.s = obs

    def set_alpha(self, alpha=0, step = 0, scheduler=True):
        """Set the learning rate. 

        If scheduler is False alpha is set manually.
        Otherwise the argument step should be provided which 
        allows this function to determine the value of alpha according 
        to the scheduler.
        """ 
        if not scheduler:
            self.alpha = alpha
        else:
            # Currently only ~1/t scheduler implemented. 
            self.alpha = self.alpha0/(1 + step/self.alpha_tau)
        return self.alpha

    def set_beta(self, beta=0, step = 0, scheduler=True, random=False):
        """Set an inverse temperature for the softmax action selection. 

        If scheduler=False beta is set manually.
        Otherwise parameter step should be provided which allows 
        this function to determine the value of beta according 
        to the scheduler.
        Two schedulers are available:
        (1) non-random: beta increases linearly in time.
        (2) random: beta is generated randomly from an exponential distribution.
        """ 
        if not scheduler:
            self.beta = beta
        else:
            # Currently only ~1/t scheduler implemented.
            if random:
                self.beta = -self.beta0*np.log(np.random.rand())
            else: 
                self.beta = self.beta0 + step/self.beta_tau
        return self.beta

# ------------Handcrafted agent-----------------
# In FrozenLake environment actions are encoded as follows:
# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP    
class testAgent(discreteQlearningAgent):
    """A simple handcrafted agent solving the task defined in the OpenAI Gym FrozenLake8x8-v0 environment. 

    Strategy: 
        (1) go up if in the top row. This will lead to one of three outcomes: stay, left, or right.
        (2) go right if the the first column from the right. This will lead to one of three outcomes: stay, up, or down.
    Given this strategy, the agent should never move outside of of top-right strips. Thus, the agent will be "surprised" 
    to find itself in such a situation and will throw an exception.

    """
    def select_action(self, greedy=False):
        if self.s < 7:      # Top row excluding the top right corner
            self.a = 3      # go UP
        elif self.s%8 == 7: # The first column from the right 
            self.a = 2      # go RIGHT
        else:
            raise Exception(f"I did not expect this state: {self.s}")
        
        return self.a    

