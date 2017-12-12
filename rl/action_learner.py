from . import block_movement_env as bme
import plotting
import numpy as np
import collections
import itertools

class ActionLearner(object):
	"""
	This class combine the learning logics of all other class
	It has the following components:

	- It stores a general config 
	- It stores a data project (project.Project) 
		that can be load from a project file path
	- It stores a progress learner (progress_learner.EventProgressEstimator) 
		that have a model be load from a project file path
	- It stores a policy_estimator and a value_estimator
	- It is a REINFORCE RL learner

	We don't initialize ActionLearner but we set different components for it
	so that we can expose the logics of saving/loading models to the outside of the learner
	otherwise we have to 
	"""

	def __init__(self, config, project, progress_estimator, 
			policy_estimator, value_estimator):
		"""
		Parameters:
		-----------

		"""
		self.config = config

		# All of these components should be 
		# This should belong to class Project
		# We assume that the data put in the project here has been preprocessed
		self.project = project

		# This should be a kind of class EventProgressEstimator
		# We assume that the progress_estimator put in the project has been learned
		self.progress_estimator = progress_estimator

		# This should belong to class PolicyEstimator
		self.policy_estimator = policy_estimator

		# This should belong to class ValueEstimator
		self.value_estimator = value_estimator

		# This can be created on fly because it is just a simple environment
		self.env = bme.BlockMovementEnv(config, project.speed, project.name, 
				progress_estimator = self.progress_estimator)


	def reinforce( self ):
	    """
	    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
	    function approximator using policy gradient.
	    
	    Returns:
	        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	    """

	    num_episodes = self.config.num_episodes
	    discount_factor = self.config.discount_factor

	    # Keeps track of useful statistics
	    stats = plotting.EpisodeStats(
	        episode_lengths=np.zeros(num_episodes),
	        episode_rewards=np.zeros(num_episodes))    
	    
	    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
	    
	    for i_episode in range(num_episodes):
	        # Reset the self.environment and pick the fisrst action
	        state = self.env.reset()
	        
	        episode = []
	        
	        # One step in the self.environment
	        for t in itertools.count():
	            
	            # Take a step
	            action_probs = self.policy_estimator.predict(state)
	            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
	            next_state, reward, done, _ = self.env.step(action)
	            
	            # Keep track of the transition
	            episode.append(Transition(
	              state=state, action=action, reward=reward, next_state=next_state, done=done))
	            
	            # Update statistics
	            stats.episode_rewards[i_episode] += reward
	            stats.episode_lengths[i_episode] = t
	            
	            # Print out which step we're on, useful for debugging.
	            print("\rStep {} @ Episode {}/{} ({})".format(
	                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
	            sys.stdout.flush()

	            if done:
	                break
	                
	            state = next_state
	        
	        accumulate_reward = 0
	        
	        # Go from backward
	        for t in range(len(episode)-1, -1, -1):
	            state, action, reward, _, _ = episode[t]
	            # G_t
	            accumulate_reward = accumulate_reward * discount_factor + reward
	            
	            """
	            IMPORTANT:
	            
	            The order between these two next commands are very important
	            Predict before update:
	            and the average stuck at -100 which means that
	            the algorithm never find the target
	            
	            Update before predict:
	            It would converge correctly
	            
	            ===== Possible explanation ======
	            If I update before predict
	            
	            the baseline is much closer to correct value
	            advantage therefore is much smaller
	            
	            In the book, the algorithm is as following:
	            do predict before update
	            add a scaling factor correspond to the advantage
	            
	            """
	            estimator_value.update(state, accumulate_reward)
	            
	            predicted_reward = estimator_value.predict(state)
	            
	            
	            # advantage
	            advantage = accumulate_reward - predicted_reward
	            
	            self.policy_estimator.update(state, discount_factor ** t * advantage, action)
	            
	    return stats