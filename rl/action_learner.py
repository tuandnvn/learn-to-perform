from __future__ import print_function
import sys
    
import tensorflow as tf
import numpy as np
import collections
import itertools

from . import block_movement_env as bme
import plotting
import traceback

from gym.wrappers import TimeLimit

def random_action(state, policy_estimator, no_of_actions = 1, verbose = False, session = None):
    action_means, action_stds = policy_estimator.predict(state, sess = session)
                    
    actions = np.random.multivariate_normal(action_means,np.diag(action_stds), size = no_of_actions)

    if verbose:
        print ('action_means = ' + str(action_means) + ' ; action_stds = ' + str(action_stds))
    return actions

# def best_n_random_action(n):
#     def best_random_action (state, policy_estimator, verbose = False):
#         action_means, action_stds = self.policy_estimator.predict(state)
                        
#         action = np.random.normal(action_means,action_stds)

#         if verbose:
#             print ('action_means = ' + str(action_means) + ' ; action_stds = ' + str(action_stds))
#         return action

#     return best_random_action

REINFORCE = 'REINFORCE'
ACTOR_CRITIC = 'ACTOR_CRITIC'

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
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    

    def __init__(self, config, project, progress_estimator, 
            policy_estimator, value_estimator, limit_step = 50, session = None):
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

        self.limit_step = limit_step

        self.session = session

    def policy_learn( self , action_policy, depth = 1, breadth = 1, verbose = False, choice = REINFORCE):
        """
        REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
        function approximator using policy gradient.
        Actor-ciritc algorithm. Similar to REINFORCE but with TD-target

        Params:
        =========
        Two choices: 'REINFORCE', 'ACTOR-CRITIC'
        
        Returns:
        =========
            An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        num_episodes = self.config.num_episodes
        discount_factor = self.config.discount_factor

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))
        
        # select object index is 0
        select_object = 0

        past_envs = []

        for i_episode in range(num_episodes):
            if verbose:
                print ('========================================')

            self.env = TimeLimit(bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session), max_episode_steps=self.limit_step)
            policy_rate = self.config.policy_learning_rate * self.config.policy_decay ** i_episode
            self.policy_estimator.assign_lr( policy_rate, sess= self.session )

            value_rate = self.config.value_learning_rate * self.config.value_decay ** i_episode
            self.value_estimator.assign_lr( value_rate, sess= self.session )

            try:
                # Reset the self.environment and pick the first action
                state = self.env.reset()
                
                episode = []
                
                # One step in the self.environment
                for t in itertools.count():
                    best_action = None
                    best_reward = -1

                    actions = random_action(state, self.policy_estimator, 
                        verbose = verbose, no_of_actions = breadth, session = self.session)

                    for breadth_step in range(breadth):
                        action = actions[breadth_step]
                        next_state, reward, done, _ = self.env.step((select_object,action))

                        self.env.env.back()

                        if done:
                            best_action = action
                            break
                        else:
                            if reward > best_reward:
                                best_reward = reward
                                best_action = action

                    if best_reward < 0:
                        # This action is not worth taking
                        break

                    # At this point, best_action corresponds to the best reward
                    # really do the action
                    next_state, reward, done, _ = self.env.step((select_object,best_action))
                    
                    if verbose:
                        print ((action, reward, done))

                    if choice == REINFORCE:
                        transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
                        # Keep track of the transition
                        episode.append(transition)
                    
                    # Update statistics
                    stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t
                    
                    # Print out which step we're on, useful for debugging.
                    if verbose:
                        print("Step {} @ Episode {}/{} ({})".format(
                                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode]))
                    else:
                        print("\rStep {} @ Episode {}/{} ({})".format(
                                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
                    #sys.stdout.flush()

                    if choice == ACTOR_CRITIC:
                        """
                        We handle update right here
                        """
                        predicted_next_state_value = self.value_estimator.predict(next_state, sess= self.session)
                        td_target = reward + discount_factor * predicted_next_state_value
                        self.value_estimator.update(state, td_target)
                        
                        predicted_target = self.value_estimator.predict(state)
                        
                        """
                        Implement update right away
                        """
                        # advantage
                        advantage = td_target - predicted_target
                        
                        # To be correct this would be discount_factor ** # of steps * advantage
                        self.policy_estimator.update(state, advantage, action)

                    if done:
                        break
                        
                    state = next_state
                
                

                past_envs.append(self.env)

                if choice == REINFORCE:
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

                        self.value_estimator.update(state, accumulate_reward, sess= self.session)

                        predicted_reward = self.value_estimator.predict(state, sess= self.session)

                        # advantage
                        advantage = accumulate_reward - predicted_reward

                        if verbose:
                            print ("accumulate_reward = %.2f; predicted_reward = %.2f; advantage = %.2f" %\
                             (accumulate_reward, predicted_reward, advantage) )
                        

                        _, regularizer_loss = self.policy_estimator.update(state, discount_factor ** t * advantage, action, sess= self.session)
                        #print ('regularizer_loss = %.2f' % regularizer_loss)
            except Exception as e:
                print ('Exception in episode %d ' % i_episode)
                traceback.print_exc()
        return (past_envs, stats)

