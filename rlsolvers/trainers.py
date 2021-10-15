"""This is trainer module. 

It wraps training and test loops into convenient methods. 
Currently, it can only be used with Gym environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os

# By default the FrozenLake environment has an internal maximum of 200 steps (done is set to True after 200 steps).
# Because of that, we do not really need the constant below. However, it is possible to remove this restriction in the environment. 
# In that case we can restrict the number of steps taken in the environment at the level of this module using the constant below.
MAX_LEN_EPISODE = 201 

# ------------Trainer (gym wrapper) class--------------------
class gymTrainer:
    """A trainer class for OpenAI Gym. 

    Tested only in a single environment (FrozenLake8x8-v0), 
    but it should also work fine in other environments. 
    
    Methods:
    -------
    train: trains an agent. The agent can be specified during class instantation. 
    It can also  be provided in the train method; in this case the environment will store 
    that agent for future trainings and tests.
    
    test: tests an agent. A greedy policy of the agent will be used (agent's select_action method must provide a flag greedy).
    
    save_agent: save an agent to a file.

    load_agent: load an agent from a file or set the 'student' agent to an agent loaded externally.

    """
    def __init__(self, env, agent=None):
        """Initialize an instance of the gymTrainer class.

        Args:
            env (object): gym environment, must be loaded externally.
            agent (object or str): either an instance of discreteQlearningAgent class, or a filename of a corresponding pickle. 
        """
        self.env   = env
        self.load_agent(agent)
        self.reset_logs()


    def train(self, agent=None, Nepisodes=1000, visualize=True, save_name=None):
        """Train an agent.

        Args:
            agent (object or str): an instance of a class that provides methods new_episode, select_action, and update, see discreteQlearningAgent class.
                                   if None, use the agent that is stored in self.agent. 
                                   if str, load the agent from a file.
            Nepisodes (int): Number of episodes to test on.
            visualize (bool): whether to visualize the training progress (using matplotlib).
            save_name (str): filename of a file to save the trained model (pickle). If None, the model is not saved.
        
        Returns:
            self.log_noisy (list): logs of episodes using stochastic policy. For each episode, store a tuple (t, r, L), i.e., episode #, reward, length.
            self.log_greedy (list): logs of episodes using greedy policy (with respect to Q). For each episode, store a tuple (t, r, L), i.e., episode #, reward, length.
            
        Note that this method will replace self.agent with a new agent, if provided. 
        In this case it will also reset training logs.
        """
        if agent is not None:
            # Load a new agent.
            self.load_agent(agent)
            # Reset logs.
            self.reset_logs()
        else:
            if self.agent is None:
                raise Exception("No agent to train (self.agent is None)")

        for i in range(Nepisodes):
            # We could read the state here, for environments not starting from s=0:
            obs = self.env.reset()
            
            try:
                self.agent.new_episode(state=obs)
            except AttributeError:
                raise Exception("agent: method new_episode not implemented correctly.")
            # Alternate between greedy and stochastic policy. 
            # This should allow the agent to explore more, but also to 
            # make sure that it can obtain a positive reward more often. 
            # Additionally, this allows us to better visualize the learning progress. 
            # The observed performance of the greedy policy is a proxy to 
            # how well the Q-function has been learned so far.
            if i%2:
                greedy = True
            else:
                greedy = False
            for j in range(MAX_LEN_EPISODE):
                # Select an action.
                try:
                    action = self.agent.select_action(greedy=greedy)
                except AttributeError:
                    raise Exception("agent: method select_action not implemented correctly.")
                # Take a step in the environment using our selected action and collect observations.
                obs, r, done, info = self.env.step(action)
                # Update agent's state and Q-function.
                try:
                    self.agent.update(obs, r, done)
                except AttributeError:
                    raise Exception("agent: method update not implemented correctly.")
                # If the episode is done, update the history.
                if done:
                    if greedy:
                        self.log_greedy.append( (i, r, j+1) )
                    else:             
                        self.log_noisy.append( (i, r, j+1) )
                    break
                if j+1==MAX_LEN_EPISODE:
                    raise Exception(f"The length of the episode has hit the maximum number: {MAX_LEN_EPISODE}")
        if visualize:
            self.__visualize_training_progress(self.log_noisy, title="Stochastic policy")
            self.__visualize_training_progress(self.log_greedy,title="Greedy policy")
        
        if not save_name is None:
            # Save the agent in a (pickle) file.
            # We could also save the logs, but frankly I would rather 
            # use an external logger/tracking tool like Tensorboard or Neptune.
            self.save_agent(save_name)

        return self.log_noisy, self.log_greedy

    def test(self, agent=None, Nepisodes=1000, summary=True):
        """Test an agent.

        Args:
            agent (object or str): an instance of a class that provides methods new_episode, select_action, and update, see discreteQlearningAgent class.
                                   if None, use the agent that is stored in self.agent. 
                                   if str, load the agent from a file.
            Nepisodes (int): Number of episodes to test on.
            summary (bool): whether to print a summary.
            
        In contrast to the train method, this method will not change self.agent, so it can be used to test 
        different agents while keeping a single agent for training.
        
        Returns:
            rhist (list): observed rewards in each episode.
            lhist (list): observed length of each episode.
        """
        rhist = []
        lhist = []
        if agent is None:
            agent = self.agent
        else:
            agent = self.load_agent(agent, remember_agent=False)

        for i in range(Nepisodes):

            obs = self.env.reset()
            
            try:
                agent.new_episode(state=obs)
            except AttributeError:
                raise Exception("agent: method new_episode not implemented correctly.")

            for j in range(MAX_LEN_EPISODE):
                # Select an action according to the greedy policy.
                try:
                    action = agent.select_action(greedy=True)
                except AttributeError:
                    raise Exception("agent: method select_action not implemented correctly.")
                # Take a step in the environment using our selected action and collect observations.
                obs, r, done, info = self.env.step(action)
                # Update agent's state (but do not train!)
                try:
                    agent.update(obs, r, done, learn=False)
                except AttributeError:
                    raise Exception("agent: method update not implemented correctly.")

                if done:
                    rhist.append(r)
                    lhist.append(j+1)
                    break
                if j+1==MAX_LEN_EPISODE:
                    raise Exception(f"The length of the episode has hit the maximum number: {MAX_LEN_EPISODE}")
        
        if summary:
            rhist_np = np.asarray( rhist )
            r_mean = np.mean( rhist_np )
            #r_std  = np.std( rhist_np ) # We can infer this from r_mean for binary (Bernoulli) random variable
            
            lhist_np = np.asarray( lhist )
            l_mean = np.mean( lhist_np )
            l_std  = np.std( lhist_np )
        
            print(f"Win rate: {100*r_mean:.1f} \u00B1 {100*3*np.sqrt(r_mean*(1-r_mean))/np.sqrt(len(rhist_np)):.1f} %")
            print(f"Average number of steps: {l_mean:.1f} \u00B1 {3*l_std/np.sqrt(len(lhist_np)):.1f}")
        return rhist, lhist
    
    def render_episode(self, agent=None, dt=1):
        """Render a single episode.
        
        Args:
            agent (object or str): an agent from the module agents or a path to the corresponding pickle.
            dt (float): delay between frames (in seconds)
        """

        if agent is None:
            agent = self.agent
        else:
            agent = self.load_agent(agent, remember_agent=False)
    
        obs = self.env.reset() 
        
        try:
            agent.new_episode(state=obs)
        except AttributeError:
            raise Exception("agent: method new_episode not implemented correctly.")
        
        os.system('cls' if os.name=='nt' else 'clear')
        self.env.render()
        
        for j in range(MAX_LEN_EPISODE):
            time.sleep(dt)
            # Select an action according to the greedy policy.
            try:
                action = agent.select_action(greedy=True)
            except AttributeError:
                raise Exception("agent: method select_action not implemented correctly.")
            # Take a step in the environment using our selected action and collect observations.
            obs, r, done, info = self.env.step(action)
            os.system('cls' if os.name=='nt' else 'clear')
            self.env.render()
            # Update agent's state (but do not train!)
            try:
                agent.update(obs, r, done, learn=False)
            except AttributeError:
                raise Exception("agent: method update not implemented correctly.")

            if done:
                break
            if j+1==MAX_LEN_EPISODE:
                raise Exception(f"The length of the episode has hit the maximum number: {MAX_LEN_EPISODE}")
         
    def save_agent(self, fname):
        """Save an object (self.agent) to a file (fname).
        
        Args:
            fname (str): filename used to save the agent.
            
        Note that this function could save other objects too, 
        since we are not checking here if the objects is an 
        instance of a class that is consistent with 
        the agent class. 
        """
        pickle.dump(self.agent, file=open(fname, "wb"))
    
    def load_agent(self, agent, remember_agent=True):
        """Load and return an object (agent). 
        
        Args:
            agent (object or str): an agent from the module agents or a path to the corresponding pickle.
            remember_agent (bool): whether to store the loaded agent as self.agent.
        Returns:
            agent_loaded (object): the loaded agent or None.
            
        If agent is a string, load the agent from a pickle file.
        If loading the file fails, set agent as None.
 
        If agent is not a string this function assumes 
        that it is an instance of the class discreteQlearningAgent 
        and loads it as such (setting self.agent to agent).

        The flag remember_agent is used to determine if self.agent should be replaced. 
        This is useful to switch between two alternative behaviors: 
        loading an agent temporarily (for tests), or loading a new agent for training. 

        Note that this function can load other objects too, 
        since we are not checking here if the objects is an 
        instance of a class that is consistent with 
        the agent class. 
        
        TO DO: check if the loaded object is an instance of the class.
        """

        if isinstance(agent, str):
            try:
                agent_loaded = pickle.load(file=open(agent, "rb"))
            except FileNotFoundError:
                print(f"File {agent} does not exist.") 
                agent_loaded = None
        else:
            agent_loaded = agent
        
        if remember_agent:
            self.agent = agent_loaded

        return agent_loaded
    
    def reset_logs(self):
        """Empty the lists containing logs."""
        self.log_greedy = []
        self.log_noisy  = []

    def __visualize_training_progress(self, log_list, title="", tau1=100, tau2=1000):
        """ Visualize training progress. 
        
        Args:
            log_list (list): list of tuples (each of length 3).
            title (str): title of the figure.
            tau1 (int): time constant of the first low-pass filter.
            tau2 (int): time constant of the second low-pass filter.
        """
        
        # List of tuples --> ndarray.
        # log[:,0] -- 'time steps' (episode #)
        # log[:,1] -- rewards
        # log[:,2] -- length of the episode
        log = np.array( log_list )

        fig, ax = plt.subplots(1,2, figsize=(10,4))
        # Plot (low-pass filtered) rewards
        ax[0].plot(log[:,0], self.__low_pass(log[:,1], tau=tau1))
        ax[0].plot(log[:,0], self.__low_pass(log[:,1], tau=tau2))
        ax[0].set_xlabel('# of episodes')
        ax[0].set_ylabel('average (low-pass filtered) reward')
        # Plot (low-pass filtered) number of steps 
        ax[1].plot(log[:,0], self.__low_pass(log[:,2], tau=tau1))
        ax[1].plot(log[:,0], self.__low_pass(log[:,2], tau=tau2))
        ax[1].set_xlabel('# of episodes')
        ax[1].set_ylabel('average # of steps per episode')

        fig.suptitle(title, fontsize=14)

    def __low_pass(self, x, tau=1):
        """Calculate and return a first-order linear low-pass filter of a 1D time series. 

        Args:
            x (ndarray): 1D numpy array with the time series to be filtered.
            tau (int): time constant of the filter.
        
        Returns: 
            y (ndarray): 1D numpy array with the filtered signal (same length as x).
            
        """
        y = np.empty( (len(x)) )
        y[0] = np.mean(x[:tau])     # Initialize the first element with an average.
        a = 1/tau

        for i in range(len(x)-1):
            y[i+1] = (1-a)*y[i] + a*x[i]

        return y

