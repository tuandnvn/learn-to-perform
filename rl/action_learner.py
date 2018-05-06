from __future__ import print_function
import sys
    
import tensorflow as tf
import numpy as np
import collections
import itertools
import scipy

from . import uniform_env_space
from . import block_movement_env as bme
import plotting
import traceback

from gym.wrappers import TimeLimit
from gym.utils import seeding

def random_action(state, policy_estimator, no_of_actions = 1, verbose = False, session = None):
    action_means, action_stds = policy_estimator.predict(state, sess = session)

    variances = action_stds ** 2

    actions = np.random.multivariate_normal(action_means,np.diag(variances), size = no_of_actions)

    return action_means, action_stds, actions

"""
The problem with the current learning algorithm is that gaussian distribution focus a lot on just one point, 
where as we might want to do some exploratory move as well
"""
def epsilon_greedy_action( state, policy_estimator, uniform_space, no_of_actions = 1, verbose = False, session = None, epsilon = 0.3):
    """
    In epsilon-amount of time, just use uniform_space instead of policy_estimator
    """
    """When action_choice == 0, do the greedy action; when action choice == 1, random an action"""
    action_means, action_stds = policy_estimator.predict(state, sess = session)

    variances = action_stds ** 2

    action_choices = np.random.choice(2, no_of_actions, p = [1 - epsilon, epsilon])

    no_greedy = len(action_choices[action_choices == 0])
    no_random = len(action_choices[action_choices == 1])

    if verbose:
        print ((action_means, variances))

    greedy_actions = np.random.multivariate_normal(action_means,np.diag(variances), size = no_greedy) 

    if no_random == 0:
        return action_means, action_stds, greedy_actions

    random_actions = []
    for i in range(no_random):
        random_actions.append(uniform_space.sample())

    if no_greedy == 0:
        return action_means, action_stds, np.array(random_actions)

    actions = np.concatenate([greedy_actions, random_actions], axis = 0)

    return action_means, action_stds, actions

def _get_relative_transform ( moving_object, static_object ):
    static_object_pos = static_object[:2] 
    moving_object_pos = moving_object[:2]
    
    theta = static_object[2]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    rel_pos = np.reshape( moving_object_pos - static_object_pos, [2])
    rel_pos = R.transpose().dot( rel_pos ).flatten()

    relative_transform = np.array([rel_pos[0], rel_pos[1]])

    return relative_transform

def normalize_state ( state ):
    moving_object, static_object = state[3:6], state[9:12]

    current_pos = _get_relative_transform ( moving_object, static_object ) 

    # Location of moving object in previous frame
    prev_moving_object = state[:3]

    prev_pos = _get_relative_transform ( prev_moving_object, static_object )

    if  np.all(np.absolute(current_pos - prev_pos) < 0.001):
        """
        No movement yet
        """
        return np.concatenate((current_pos, np.zeros(2)))
    else:
        normalized_action = (current_pos - prev_pos) / np.linalg.norm(current_pos - prev_pos)
        return np.concatenate((current_pos, normalized_action))

def realize_action ( env, select_object, normalized_action ):
    moving_object = env.objects[select_object].transform.get_feat()
    static_object = env.objects[1 - select_object].transform.get_feat()

    static_object_pos = static_object[:2] 
    moving_object_pos = moving_object[:2]

    theta = static_object[2]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    location = R.dot(normalized_action[:2]).flatten() + static_object_pos

    return np.concatenate( [location, [moving_object[2]]] )

def get_pdf (policy_estimator, normalized_state, action, session):
    action_means, action_stds = policy_estimator.predict(normalized_state, sess = session)
    variances = action_stds ** 2

    return action_means, scipy.stats.multivariate_normal(action_means, np.diag(variances)).pdf(action)

REINFORCE = 'REINFORCE'
ACTOR_CRITIC = 'ACTOR_CRITIC'
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

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
    - It is a REINFORCE or ACTOR_CRITIC RL learner

    We don't initialize ActionLearner but we set different components for it
    so that we can expose the logics of saving/loading models to the outside of the learner
    otherwise we have to 
    """

    def __init__(self, config, project, progress_estimator, 
            policy_estimator, value_estimator, limit_step = 20, session = None):
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

        self.np_random, _ = seeding.np_random(None)

        # This is to generate the original positions of the objects a little bit closer to the center
        playground_x = [self.config.block_size-1,self.config.block_size-1, 0]
        playground_dim = [2-2*self.config.block_size, 2-2*self.config.block_size, np.pi/2]
        self.uniform_space = uniform_env_space.Uniform(p = playground_x, 
                                         dimension = playground_dim, 
                                         randomizer = self.np_random)

    def policy_learn( self , action_policy, depth = 1, breadth = 1, 
            verbose = False, choice = REINFORCE, default = False):
        """
        REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
        function approximator using policy gradient.
        Actor-ciritc algorithm. Similar to REINFORCE but with TD-target

        Params:
        =========
        action_policy: A function that takes (state, policy_estimator, no_of_actions) 
                        and return no_of_actions of actions 
        choice: Two choices: 'REINFORCE', 'ACTOR-CRITIC'. By default is REINFORCE
        default: Whether we should always use a default setup for learning. = True is 
            a debug mode. By default is False
        
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
            sigma = self.config.start_sigma * ( self.config.end_sigma / self.config.start_sigma ) ** (float(i_episode) / num_episodes)

            if verbose:
                print ('========================================')

            self.env = TimeLimit(bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session), max_episode_steps=self.limit_step)
            policy_rate = self.config.policy_learning_rate * self.config.policy_decay ** (i_episode // self.config.policy_decay_every )
            self.policy_estimator.assign_lr( policy_rate, sess= self.session )
            self.policy_estimator.assign_sigma( sigma, sess= self.session )

            value_rate = self.config.value_learning_rate * self.config.value_decay ** (i_episode // self.config.value_decay_every )
            self.value_estimator.assign_lr( value_rate, sess= self.session )

            try:
                # Reset the self.environment and pick the first action
                if default:
                    self.env.reset()
                    state = self.env.env.default()
                else:
                    state = self.env.reset()
                
                episode = []
                
                # One step in the self.environment
                for t in itertools.count():
                    best_action = None
                    best_reward = -1

                    normalized_state = normalize_state(state)

                    action_means, action_stds, actions = action_policy(normalized_state, self.policy_estimator,
                        verbose = verbose, no_of_actions = breadth, session = self.session)

                    if verbose:
                        print ((action_means, action_stds))

                    for breadth_step in range(breadth):
                        action = actions[breadth_step]
                        realized_action = realize_action(self.env.env.e, select_object, action)
                        realized_means = realize_action(self.env.env.e, select_object, action_means)
                        next_state, reward, done, _ = self.env.step((select_object, realized_action, realized_means, action_stds))

                        if verbose:
                            print ('action = %s' % str((action, reward)))
                        self.env.env.back()

                        if done:
                            best_reward = reward
                            best_action = action
                            break
                        else:
                            if reward > best_reward:
                                best_reward = reward
                                best_action = action

                    # if best_reward < 0:
                    #     # This action is not worth taking
                    #     break

                    # At this point, best_action corresponds to the best reward
                    # really do the action
                    best_realized_action = realize_action(self.env.env.e, select_object, best_action)
                    next_state, reward, done, _ = self.env.step((select_object, best_realized_action, realized_means, action_stds))
                    normalized_next_state = normalize_state(next_state)

                    # if abs(reward - best_reward) > 0.01:
                    #     print ('Damn wrong: reward = %.4f; best_reward = %.4f' % (reward, best_reward))
                    
                    if verbose:
                        print ('best_action = %s; best_realized_action = %s, best_reward = %.3f ' % 
                            (str(best_action), str(best_realized_action), reward))

                    if choice == REINFORCE:
                        transition = Transition(state=normalized_state, action=action, reward=reward, next_state=normalized_next_state, done=done)
                        # Keep track of the transition
                        episode.append(transition)
                    
                    # Update statistics
                    stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t
                    
                    # Print out which step we're on, useful for debugging.
                    if verbose:
                        print("Step {} @ Episode {}/{} ({}), (sigma = {})".format(
                                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode], sigma))
                    else:
                        print("\rStep {} @ Episode {}/{} ({}), (sigma = {})".format(
                                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1], sigma), end="")
                    #sys.stdout.flush()

                    if choice == ACTOR_CRITIC:
                        """
                        We handle update right here
                        """
                        predicted_next_state_value = self.value_estimator.predict(normalized_state, sess= self.session)
                        td_target = reward + discount_factor * predicted_next_state_value
                        self.value_estimator.update(normalized_state, td_target, sess= self.session)
                        
                        predicted_target = self.value_estimator.predict(normalized_state, sess= self.session)
                        
                        """
                        Implement update right away
                        """
                        # advantage
                        advantage = td_target - predicted_target

                        if verbose:
                            print ('td_target = %.2f, predicted_target = %.2f, advantage = %.2f' 
                                % (td_target, predicted_target, advantage) )
                        
                        before_mean, before_pdf = get_pdf(self.policy_estimator, normalized_state, best_action, self.session)

                        # To be correct this would be discount_factor ** # of steps * advantage
                        loss = self.policy_estimator.update(normalized_state, advantage, best_action, sess= self.session)

                        after_mean, after_pdf = get_pdf(self.policy_estimator, normalized_state, best_action, self.session)

                        if verbose:
                            print ('Before mean = %s, pdf = %.3f, After mean = %s, pdf = %.3f' % (before_mean, before_pdf, after_mean, after_pdf) )
                        #print ('loss = %.2f' % loss)
                    if done:
                        break
                        
                    state = next_state
                
                

                past_envs.append(self.env)

                if choice == REINFORCE:
                    accumulate_reward = 0

                    # We just cut 
                    # already_cut_negative_reward = False
                    # Go from backward
                    for t in range(len(episode)-1, -1, -1):
                        normalized_state, action, reward, _, _ = episode[t]
                         
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

                        self.value_estimator.update(normalized_state, accumulate_reward, sess= self.session)

                        predicted_reward = self.value_estimator.predict(normalized_state, sess= self.session)

                        # advantage
                        advantage = accumulate_reward - predicted_reward

                        if verbose:
                            print ("accumulate_reward = %.2f; predicted_reward = %.2f; advantage = %.2f" %\
                             (accumulate_reward, predicted_reward, advantage) )
                        
                        before_mean, before_pdf = get_pdf(self.policy_estimator, normalized_state, best_action, self.session)

                        loss = self.policy_estimator.update(normalized_state, discount_factor ** t * advantage, action, sess= self.session)

                        after_mean, after_pdf = get_pdf(self.policy_estimator, normalized_state, best_action, self.session)

                        if verbose:
                            print ('Before mean = %s, pdf = %.3f, After mean = %s, pdf = %.3f' % (before_mean, before_pdf, after_mean, after_pdf) )
                        #print ('loss = %.2f' % loss)
            except Exception as e:
                print ('Exception in episode %d ' % i_episode)
                traceback.print_exc()
        return (past_envs, stats)

