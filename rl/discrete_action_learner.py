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

def random_action(state, policy_estimator, no_of_actions = 1, verbose = False, session = None):
    action_probs = policy_estimator.predict(state, sess = session)

    actions = np.random.choice(len(action_probs), size = no_of_actions, p = action_probs)

    return None, None, actions

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

def calculate_angle ( vector, discretize_angle = 8 ):
    """
    discretize_angle is an even number of regions that we will split our 2*pi angle
    
    We will name the regions from 0 to discretize_angle - 1
    """
    angle = np.arctan2(vector[1], vector[0])
    if angle < 0:
        angle = 2 * np.pi + angle
    angle += np.pi / discretize_angle

    # Discretize
    region = angle // (2 * np.pi / discretize_angle)

    region = region % discretize_angle

    return region

def quantize_feat ( moving_object, static_object, discretized_space = [0.25, 0.5, 1], zero_tolerance = 0.05 ):
    """
    Quantize from the positions of two objects moving_object and static_object into the relative position 
    of moving_object w.r.t static_object

    We treat the relative positions of moving object the same if the angle make with the static object
    is 0, 90, 180, 270 degree (because the rectangle shape is 4 way symmetric)

    We will quantize the block on the same row as 1, 2, 3
     ____     ____
    |    |   |    |
    |____|   |____|
     
    We will quantize the block diagonal to each other as 4, 5, 6
    
     ____ 
    |    | 
    |____|
           ____ 
          |    | 
          |____|

    So in total we have 6 different positions for a relative block 

    Parameters:
    =============
    moving_object : array of 3
    static_object : array of 3

    Return:
    =============
    value: 1 to len(discretized_space)
    region: 1 to 8
    """
    static_object_pos = static_object[:2] 
    moving_object_pos = moving_object[:2]
    
    theta = static_object[2]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    distance = np.reshape( moving_object_pos - static_object_pos, [2])

    distance = R.transpose().dot( distance )

    prev_margin = zero_tolerance
    abs_distance = np.linalg.norm(distance)

    for d_i, d in enumerate(discretized_space):
        q = abs_distance / d

        if prev_margin <= abs_distance:
            if d_i < len(discretized_space) - 1:
                if q < 1.5:
                    value = d_i + 1
                    break
            else:
                value = d_i + 1
                break

        prev_margin = d * 1.5

    region = calculate_angle (distance) 

    return value, region

def get_quantized_state ( moving_object, static_object, discretized_space = [0.25, 0.5, 1], zero_tolerance = 0.05 ):
    value, region = quantize_feat (distance)

    if region % 2 == 1:
        value += len(discretized_space)

    return value

def _test_quantize_feat ():
    print (get_quantized_state(np.array([0,0.25,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([0.25,0,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([0.25,0.25,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([.5,.5,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([1,1,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([1,0,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([0,1,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([-1,-1,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([-1,1,1]), np.array([0,0,0])))
    print (get_quantized_state(np.array([1,-1,1]), np.array([0,0,0])))

def quantize_movement ( prev, cur, static ):
    """
    Produce the previous action that moves the moving object from prev to cur
    """
    prev_pos = quantize_feat ( prev, static )
    cur_pos = quantize_feat ( cur, static )

    return get_action_from_quantized_states(prev_pos, cur_pos)

def get_action_from_quantized_states ( prev, cur ):
    """
    Parameter
    ============
    prev, cur: quantized (value, region) of relative positions of moving object 
            value: 1 to 3
            region: 0 to 7

    Return
    ============
    action: A value from 0 to 4 (0 means no movement, 1 to 4 means UP, RIGHT, DOWN, LEFT)
    """
    if prev == cur:
        return 0

    if prev[0] % 3 == cur[0] % 3:
        if prev[1] == (cur[1] + 1) % 8:
            return 2
        if (prev[1] + 1) % 8 == cur[1]:
            return 4

    if prev[1] == cur[1]:
        if prev[0] == cur[0] - 1:
            return 1
        if prev[0] == cur[0] + 1:
            return 3

    raise ValueError('The action from %s to %s is illegal' % (prev, cur))

def _test_get_action_from_quantized_states():
    for j in range(3):
        for i in range(1, 4):
            try:
                print ('The action %d from %s to %s' % (get_action_from_quantized_states((2,1), (i,j)), (2,1), (i,j)))
            except ValueError as e:
                print (e)

def get_next_from_action ( prev, action ):
    """
    Parameter
    ============
    prev: quantized (value, region) of relative positions of moving object 
            value: 1 to 3
            region: 0 to 7
    action: A value from 0 to 4

    Return
    ============
    cur: quantized (value, region) of relative positions of moving object 
            value: 1 to 3
            region: 0 to 7
    if action is illegal, we stop immediately for return of 0
    """
    if action == 0:
        return prev

    if action == 1:
        return min(3, prev[0] + 1), prev[1]

    if action == 3:
        return max(1, prev[0] - 1), prev[1]

    if action == 2:
        return prev[0], (prev[1] + 7) % 8

    if action == 4:
        return prev[0], (prev[1] + 1) % 8

def realize_action( env, select_object, action, discretized_space = [0.25, 0.5, 1]):
    moving_object = env.objects[select_object].transform.get_feat()
    static_object = env.objects[1 - select_object].transform.get_feat()
    return _realize_action ( moving_object, static_object, action, discretized_space )

def _realize_action( moving_object, static_object, action, discretized_space = [0.25, 0.5, 1]):
    """
    Translate an action, could be of continuous, or discretized form (but relative to a static object), to a real tranform for action
    For example: if the static object is (0.5, 0.5, 0.5)
    Notice the rotation is modulo 90 degree

    action is from 0 to 4

    Return a real position of action
    """

    static_object_pos = static_object[:2] 
    moving_object_pos = moving_object[:2]
    
    theta = static_object[2]

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    prev = quantize_feat ( moving_object, static_object )

    print ('prev', prev)

    cur = get_next_from_action ( prev, action )

    print ('cur', cur)

    # copy value from action
    alpha = cur[1] * np.pi / 4
    Q = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    vals = np.expand_dims([discretized_space[int(cur[0]) - 1], 0], axis =1 )  

    # print (values)
    # Rotated without translation
    location = R.dot( Q.dot( vals ))
    # print (location)
    # Real loation
    location = location.flatten() + np.reshape( static_object_pos, [2])

    return np.concatenate( [location, [theta] ] )

def _test_realize_action():
    moving_object = np.array([0.7, 0.7, 0.5])
    static_object = np.array([0.3, 0.4, 0])
    print ('quantized', quantize_feat(moving_object, static_object))

    for i in range(5):
        action = _realize_action ( moving_object, static_object, i)
        print ('action', action)

        print ('requantized', quantize_feat(action, static_object))

def quantize_state ( state, progress ):
    """
    State from the block environment is features of moving and static objects
    Discretize for the moving object
    """
    # Get relative position between two objects
    pos = get_quantized_state ( state[3:6], state[9:12] )
    action = quantize_movement ( state[:3], state[3:6], state[9:12], zero_tolerance = 0.05 )

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

