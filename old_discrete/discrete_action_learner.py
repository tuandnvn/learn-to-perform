from __future__ import print_function
import sys
    
import tensorflow as tf
import numpy as np
import collections
import itertools
import math

from . import uniform_env_space
from . import block_movement_env as bme
import plotting
import traceback

from gym.wrappers import TimeLimit
from gym.utils import seeding
from gym.spaces import MultiDiscrete

from .action_learner import random_action


class MultiDiscreteNoZero ( MultiDiscrete ) :
    def __init__(self, array_of_param_array, no_zero_range):
        """
        Range of indices that don't allow 0
        """
        MultiDiscrete.__init__(self, array_of_param_array)
        self.no_zero_range = no_zero_range

    def sample(self):
        while True:
            s = MultiDiscrete.sample(self)
            b = np.array(s[self.no_zero_range[0] : self.no_zero_range[1]])
            if len(b[b == 0]) == 0:
                return s


def epsilon_greedy_action_2( state, policy_estimator, uniform_space, no_of_actions = 1, verbose = False, session = None, epsilon_1 = 0.5, epsilon_2 = 0.3):
    """
    In epsilon_1 -amount of time, use the mode value only, and ignoring the Gaussian distribution
    In epsilon_2 -
    In (1- epsilon_1 - epsilon_1)-amount of time, just use uniform_space instead of policy_estimator

    Notice that these actions values would still need to be processed into discretized values
    """
    """When action_choice == 0, do the greedy action; when action choice == 1, random an action"""
    action_means, action_stds = policy_estimator.predict(state, sess = session)

    variances = action_stds ** 2

    action_choices = np.random.choice(3, no_of_actions, p = [epsilon_1, epsilon_2, 1 - epsilon_1 - epsilon_2])

    no_greedy = len(action_choices[action_choices == 0])
    no_gaussian = len(action_choices[action_choices == 1])
    no_random = len(action_choices[action_choices == 2])

    if verbose:
        print ((action_means, variances))

    if no_greedy == 0:
        greedy_actions = np.zeros((0, len(action_means)))
    else:
        greedy_actions = [action_means for i in range (no_greedy)]

    if no_gaussian == 0:
        gaussian_actions = np.zeros((0, len(action_means)))
    else:
        gaussian_actions = np.random.multivariate_normal(action_means,np.diag(variances), size = no_gaussian) 

    if no_random == 0:
        random_actions = np.zeros((0, len(action_means)))
    else:
        random_actions = []
        for i in range(no_random):
            random_actions.append(uniform_space.sample())

    actions = np.concatenate([greedy_actions, gaussian_actions, random_actions], axis = 0)

    if verbose:
        print (actions)

    return action_means, action_stds, actions

def realize_action( env, select_object, action, discretized_space = [0.18, 0.36, 0.72], discretized_rotation = np.pi/8):
    """
    Translate an action, could be of continuous, or discretized form (but relative to a static object), to a real tranform for action
    For example: if the static object is (0.5, 0.5, 0.5)
    Notice the rotation is modulo 90 degree

    action is (1, 1, 1) -> real action postion = ( 0.5 + 1 * 0.18, 0.5 + 1 * .18, ( 0.5 + 1 * np.pi/8 ) % (np.pi/2) )
    
    action is (0.8, 0.8, 0.8) -> quantize to (1, 1, 1)  -> same real action position

    However, we don't quantize to (0,0,0) as that would violate overlaping constraint
    We also only quantize down to values <= len(discretized_space)

    action is (0.1, 0.1, 0.1) -> quantize to (1, 1, 0)  

    action is (5, 5, 1) -> quantize to (3, 3, 1)  

    Return a real position of action
    """
    if select_object == 0:
        static_object = 1
    else:
        static_object = 0

    static_object_transform = env.e.objects[static_object].transform
    pos = static_object_transform.position
    theta = static_object_transform.rotation

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Clone
    discrete_action = np.array(action)

    for i in range(3):
        discrete_action[i] = round(discrete_action[i])

    for i in range(2):
        if discrete_action[i] == 0:
            if action[i] > 0:
                discrete_action[i] = 1
            elif action[i] < 0:
                discrete_action[i] = -1
            else:
                discrete_action[i] = [1, -1] [np.random.choice(2)]


        if math.fabs(discrete_action[i]) > len(discretized_space):
            discrete_action[i] = math.copysign( len(discretized_space),  discrete_action[i])

    # print (discrete_action)

    # Just position without rotation
    values = [ math.copysign( discretized_space[int(math.fabs(discrete_action[i])) - 1], discrete_action[i]) for i in range(2) ]

    # print (values)
    # Rotated without translation
    location = R.dot( np.array( [[values[0]], [values[1]]]) )
    # print (location)
    # Real loation
    location = location.flatten() + np.reshape( pos, [2])

    rotation = ( theta + discretized_rotation * discrete_action[2] ) % (np.pi/2)

    return np.concatenate( [location, [rotation] ] )

def quantize_position ( env, select_object, discretized_space = [0.18, 0.36, 0.72], discretized_rotation = np.pi/8 ):
    """
    Translate position of the select_object into a quantized form in relative to the static object

    This is a reverse of realize_action
    """
    if select_object == 0:
        static_object = 1
    else:
        static_object = 0

    static_object_transform = env.e.objects[static_object].transform
    moving_object_transform = env.e.objects[select_object].transform
    return quantize_feat ( moving_object_transform.get_feat(), static_object_transform.get_feat(), discretized_space, discretized_rotation)

def quantize_feat ( moving_object, static_object, discretized_space = [0.18, 0.36, 0.72], discretized_rotation = np.pi/8, zero_tolerance = 0 ):
    """
    moving_object : array of 3
    static_object : array of 3
    """
    static_object_pos = static_object[:2] 
    moving_object_pos = moving_object[:2]
    
    theta = static_object[2]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    distance = np.reshape( moving_object_pos - static_object_pos, [2])

    distance = R.transpose().dot( distance )

    prev_margin = zero_tolerance
    abs_distance = np.abs(distance)

    values = np.zeros(3)

    for d_i, d in enumerate(discretized_space):
        q = abs_distance / d

        for i, v in enumerate(q):
            if prev_margin <= abs_distance[i]:
                if d_i < len(discretized_space) - 1:
                    if v < 1.5:
                        values[i] = math.copysign( d_i + 1, distance[i] )
                else:
                    values[i] = math.copysign( d_i + 1, distance[i] )

        prev_margin = d * 1.5

    rotation_diff = moving_object[2] - static_object[2]
    if rotation_diff < 0:
        rotation_diff += np.pi / 2

    values[2] = rotation_diff // discretized_rotation
    return values


def quantize_state ( state ):
    """
    State from the block environment is features of moving and static objects
    Discretize for the moving object
    """

    # Get relative
    pos = quantize_feat ( state[3:6], state[9:12] )
    velocity = quantize_feat ( state[3:6], state[:3], zero_tolerance = 0.05 )

    return np.concatenate([pos, velocity])

REINFORCE = 'REINFORCE'
ACTOR_CRITIC = 'ACTOR_CRITIC'
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class DiscreteActionLearner(object):
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
    """

    def __init__(self, config, project, progress_estimator, 
            policy_estimator, value_estimator, discretized_space = [0.18, 0.36, 0.72], discretized_rotation = np.pi/8, limit_step = 20, session = None):
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

        playground_x = [self.config.block_size-1,self.config.block_size-1, 0]
        playground_dim = [2-2*self.config.block_size, 2-2*self.config.block_size, np.pi/2]

        

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
                if default:
                    self.env.reset()
                    state = self.env.env.default()
                else:
                    state = self.env.reset()

                quantized_state = quantize_state(state)
                
                episode = []
                
                # One step in the self.environment
                for t in itertools.count():
                    best_action = None
                    best_reward = -1

                    action_means, action_stds, actions = action_policy(quantized_state, self.policy_estimator,
                        verbose = verbose, no_of_actions = breadth, session = self.session)

                    actions = [realize_action( self.env.env, select_object, action ) for action in actions]

                    if verbose:

                        print ((action_means, action_stds))

                    for breadth_step in range(breadth):
                        action = actions[breadth_step]

                        _, reward, done, _ = self.env.step((select_object,action, action_means, action_stds))

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

                    if verbose:
                        print ('best reward = %.2f' % best_reward)

                    translated_action_means = action_means + state[9:12]
                    # At this point, best_action corresponds to the best reward
                    # really do the action
                    next_state, reward, done, _ = self.env.step((select_object,best_action, translated_action_means, action_stds))

                    quantized_next_state = quantize_state(next_state)

                    # if abs(reward - best_reward) > 0.01:
                    #     print ('Damn wrong: reward = %.4f; best_reward = %.4f' % (reward, best_reward))
                    
                    if verbose:
                        print ('best_action = ' + str((best_action, reward, done)))

                    if choice == REINFORCE:
                        transition = Transition(state=quantized_state, action=action, reward=reward, next_state=quantized_next_state, done=done)
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
                        predicted_next_state_value = self.value_estimator.predict(quantized_next_state, sess= self.session)
                        td_target = reward + discount_factor * predicted_next_state_value
                        self.value_estimator.update(quantized_state, td_target, sess= self.session)
                        
                        predicted_target = self.value_estimator.predict(quantized_state, sess= self.session)
                        
                        """
                        Implement update right away
                        """
                        # advantage
                        advantage = td_target - predicted_target

                        if verbose:
                            print ('td_target = %.2f, predicted_target = %.2f, advantage = %.2f' 
                                % (td_target, predicted_target, advantage) )
                        
                        # To be correct this would be discount_factor ** # of steps * advantage
                        loss = self.policy_estimator.update(quantized_state, advantage, action, sess= self.session)
                        #print ('loss = %.2f' % loss)
                    if done:
                        break
                        
                    quantized_state = quantized_next_state
                

                past_envs.append(self.env)

                if choice == REINFORCE:
                    accumulate_reward = 0

                    # We just cut 
                    # already_cut_negative_reward = False
                    # Go from backward
                    for t in range(len(episode)-1, -1, -1):
                        state, action, reward, _, _ = episode[t]

                        # if not already_cut_negative_reward:
                        #     if reward < 0:
                        #         continue
                        #     else:
                        #         already_cut_negative_reward = True
                         
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
                        

                        loss = self.policy_estimator.update(state, discount_factor ** t * advantage, action, sess= self.session)
                        #print ('loss = %.2f' % loss)
            except Exception as e:
                print ('Exception in episode %d ' % i_episode)
                traceback.print_exc()
        return (past_envs, stats)

