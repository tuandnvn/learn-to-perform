# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
    
import tensorflow as tf
import numpy as np
import collections
import itertools

from . import uniform_env_space
from . import block_movement_env as bme
import plotting
import traceback

from gym.wrappers import TimeLimit
from gym.utils import seeding
# from importlib import reload
# reload(bme)

def random_action_constraint(state = None, policy_estimator = None, action_means = None, action_stds = None, no_of_actions = 1, verbose = False, 
       session = None, constraint_function = lambda a : True):
    """
    Random no_of_actions actions that satisfy constraints specified by constraint_function

    Parameters:
    ==================
    state: (optional) input to policy_estimator, required if policy_estimator is not None
    policy_estimator: (optional) value_estimator.PolicyEstimator instance
    action_means: (optional) provide if policy_estimator is None (np.array of size 3) 
    action_stds: (optional) provide if policy_estimator is None (np.array of size 3)
    no_of_actions: 
    verbose: 
    session: tf Session
    constraint_function: function that check the action (np.array of size 3), return true if action satisfies.

    Returns:
    ==================
    """
    if not policy_estimator is None:
        action_means, action_stds = policy_estimator.predict(state, sess = session)

    variances = action_stds ** 2

    actions = []

    while True:
        tempo = np.random.multivariate_normal(action_means,np.diag(variances), size = no_of_actions)

        actions += [act for act in tempo if constraint_function(act)]

        if len(actions) > no_of_actions:
            break

    return action_means, action_stds, actions[:no_of_actions]

def action_policy(config):
    """
    Given a config that has defined a playground
    """
    def boundary_constraint(action):
        # Ignore rotation
        for i in range(2):
            if action[i] < config.playground_x[i]:
                return False
            if action[i] > config.playground_x[i] + config.playground_dim[i]:
                return False
        
        return True
    
    def q(state, policy_estimator = None, action_means = None, action_stds = None, no_of_actions = 1, verbose = False, 
       session = None):
        return random_action_constraint(state = state, policy_estimator = policy_estimator, action_means = action_means, action_stds = action_stds,
                    no_of_actions = no_of_actions, verbose = verbose, session = session, constraint_function = boundary_constraint)
    
    return q

class ActionLearner_Search(object):
    """
    This search method starts is similar to REINFORCE,
    but with a twist.
    - Firstly, it starts with a large standard deviation (say 2) so that all points in the play ground have similar distribution
    - For each start configuration, it runs a thorough search, with a breath. For example, breadth = 10 -> it randomizes at least
    10 legal actions. For each next move, it randomizes another 10 actions, for 100 configurations, than it selects the best 10 for 
    next search.
    - In general, the algorithm would stop increase action-steps when there are a trajectory that satisfy progress function > threshold.
    - It keeps searching over the remaining space for possibly more trajectory. 
    - All "good" trajectories are kept as update samples for the policy.

    - The target is for the agent to quickly find a good policy 
    - When the number of actions need to search doesn't improve at a step
    
    """
    def __init__(self, config, project, progress_estimator, limit_step = 10, session = None, env = None):
        self.config = config

        # All of these components should be 
        # This should belong to class Project
        # We assume that the data put in the project here has been preprocessed
        self.project = project

        # This should be a kind of class EventProgressEstimator
        # We assume that the progress_estimator put in the project has been learned
        self.progress_estimator = progress_estimator

        # This should belong to class PolicyEstimator
        # with tf.variable_scope("search", reuse = True) as scope:
        #     self.policy_estimator = PolicyEstimator(self.config)

        self.limit_step = limit_step

        self.session = session

        self.np_random, _ = seeding.np_random(None)

        if env == None:
            env = bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session)
            env.reset()
        
        self.env = env

        self.action_policy = action_policy(self.config)

    def learn_one_setup( self, select_object = 0, verbose = False):
        # sigma = self.config.start_sigma
        # self.policy_estimator.assign_sigma( sigma, sess= self.session )

        # Every action_level, we would search for keep_branching * branching new positions
        # keep_branching is the number of explorations keep from the previous step
        # For the first action_level, keep_branching = 1
        #  branching is the number of new action explored for each exploration
        # For the first action_level, keep_branching = keep_branching * branching
        keep_branching = self.config.keep_branching
        branching = self.config.branching
        # shorten
        env = self.env

        explorations = [env.clone()]

        # Each accumulated reward for each exploration
        rewards = [0]

        found_completed_act = False
        # We do one action at a time for all exploration
        for action_level in range(6):
            if verbose:
                print ('action_level = %d' % action_level)
        
            # This would store a tuple of (exploration_index, accumulated_reward, action, action_means, action_stds)
            # branching ** 2
            tempo_rewards = []
            
            for exploration_index, exploration in enumerate(explorations):
                if verbose:
                    print ('exploration_index = %d' % exploration_index)

                if action_level == 0:
                    no_of_search = keep_branching * branching
                    state = exploration.get_observation_start()
                else:
                    no_of_search = branching
                    # State interpolated by WHOLE mode
                    state, _ = exploration.get_observation_and_progress()
                #print ('state = ' + str(state))

                # Simply use the static object as means
                means = env.e.objects[1 - select_object].transform.get_feat()

                action_means, action_stds, actions = self.action_policy(state = state, action_means = means, action_stds = self.config.start_sigma,
                    verbose = verbose, no_of_actions = no_of_search, session = self.session)

                #print (actions)

                for action_index, action in enumerate(actions):
                    _, reward, done, _ = exploration.step((select_object,action, action_means, action_stds))
                    #print ((action, reward))
                    exploration.back()

                    tempo_rewards.append( (exploration_index, rewards[exploration_index] + reward,
                        action, action_means, action_stds) )

                    if done:
                        print ("=== found_completed_act ===")
                        found_completed_act = True

                # tuple_actions = [(select_object, action, action_means, action_stds) for action in actions]
                # legal_action_indices, all_progress = exploration.try_step_multi(tuple_actions)

                # for index, progress in zip (legal_action_indices, all_progress):
                #     tempo_rewards.append( (exploration_index, progress,
                #         actions[index], action_means, action_stds) )

                #     if progress > self.config.progress_threshold:
                #         print ("=== found_completed_act ===")
                #         found_completed_act = True

            tempo_rewards = sorted(tempo_rewards, key = lambda t: t[1], reverse = True)
            test = [(t[0], t[1]) for t in tempo_rewards]

            if verbose:
                print (test[:keep_branching])

            new_explorations = []
            rewards = []
            for exploration_index, acc_reward, action, action_means, action_stds in tempo_rewards[:keep_branching]:
                env = explorations[exploration_index].clone()
                env.step((select_object,action, action_means, action_stds))
                new_explorations.append(env)
                rewards.append(acc_reward)
            
            explorations = new_explorations

            if found_completed_act:
                # Stop increase action_level
                break

        return explorations
