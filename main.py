
class ActionLearner(object):
	"""
	This class combine the learning logics of all other class
	It has the following components:

	- It stores a general config 
	- It stores a data project (project.Project) 
		that can be load from a project file path
	- It stores a progress learner (progress_learner.EventProgressEstimator) 
		that have a model be load from a project file path
	- It stores a REINFORCE RL learner ()
	"""

	def __init__(self, config, project, progress_estimator, policy_estimator):
		self.config = config

		# This should belong to class Project
		self.project = project

		# This should belong to class EventProgressEstimator
		self.progress_estimator = progress_estimator

		# This should belong to class PolicyEstimator
		self.policy_estimator = policy_estimator

		# This should belong to class ValueEstimator
		self.value_estimator = value_estimator

	def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
	    """
	    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
	    function approximator using policy gradient.
	    
	    Args:
	        env: OpenAI environment.
	        estimator_policy: Policy Function to be optimized 
	        estimator_value: Value function approximator, used as a baseline
	        num_episodes: Number of episodes to run for
	        discount_factor: Time-discount factor
	    
	    Returns:
	        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	    """
	    # Keeps track of useful statistics
	    stats = plotting.EpisodeStats(
	        episode_lengths=np.zeros(num_episodes),
	        episode_rewards=np.zeros(num_episodes))    
	    
	    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
	    
	    for i_episode in range(num_episodes):
	        # Reset the environment and pick the fisrst action
	        state = env.reset()
	        
	        episode = []
	        
	        # One step in the environment
	        for t in itertools.count():
	            
	            # Take a step
	            action_probs = estimator_policy.predict(state)
	            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
	            next_state, reward, done, _ = env.step(action)
	            
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
	            
	            estimator_policy.update(state, discount_factor ** t * advantage, action)
	            
	    return stats