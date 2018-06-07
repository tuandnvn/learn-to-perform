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

    for _ in range(20):
        tempo = np.random.multivariate_normal(action_means,np.diag(variances), size = 20 * no_of_actions)

        actions += [act for act in tempo if constraint_function(act)]

        if len(actions) > no_of_actions:
            break

    return action_means, action_stds, actions[:no_of_actions]

def boundary_constraint(config, action):
    # Ignore rotation
    for i in range(2):
        if action[i] < config.playground_x[i]:
            return False
        if action[i] > config.playground_x[i] + config.playground_dim[i]:
            return False
    
    return True

def random_action_policy(config):
    """
    Given a config that has defined a playground
    """
    def q(state = None, policy_estimator = None, action_means = None, action_stds = None, no_of_actions = 1, verbose = False, 
       session = None):
        return random_action_constraint(state = state, policy_estimator = policy_estimator, action_means = action_means, action_stds = action_stds,
                    no_of_actions = no_of_actions, verbose = verbose, session = session, constraint_function = lambda action: boundary_constraint(config, action) )
    
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
    def __init__(self, config, project, progress_estimator, session = None, env = None, action_policy = random_action_policy):
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

        self.session = session

        self.np_random, _ = seeding.np_random(None)

        if env == None:
            env = bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session)
            env.reset()
        
        self.env = env

        self.action_policy = random_action_policy(self.config)

    def _get_actions(self, select_object, exploration, no_of_search, verbose) :
        # Simply use the static object as means
        means = exploration.e.objects[1 - select_object].transform.get_feat()

        action_means, action_stds, actions = self.action_policy(action_means = means, action_stds = np.array([2.0, 2.0, 0.5]),
            verbose = verbose, no_of_actions = no_of_search, session = self.session)

        return action_means, action_stds, actions

    def _get_no_of_search(self, exploration, action_level):
        if action_level == 0:
            no_of_search = self.config.keep_branching * self.config.branching
            state = exploration.get_observation_start()
        else:
            no_of_search = self.config.branching
            state, _ = exploration.get_observation_and_progress()

        return no_of_search, state

    def greedy ( self, select_object = 0, verbose = False ):
        no_of_search = self.config.branching
        env = self.env

        action_level = 0
        progress = 0

        found_completed_act = False

        while True:
            if verbose:
                print ('action_level = %d' % action_level)

            action_means, action_stds, actions = self._get_actions(select_object, env, no_of_search, verbose)

            best_reward = -1
            best_action = None

            for action_index, action in enumerate(actions):
                _, reward, done, _ = env.step((select_object, action, action_means, action_stds))
                env.back()

                if reward > best_reward:
                    best_reward = reward
                    best_action = action

                if done:
                    if verbose:
                        print ("=== found_completed_act ===")
                    found_completed_act = True
        

            if best_reward > 0:
                if verbose:
                    print ("==========best action========== ", best_action)
                env.step((select_object, best_action, action_means, action_stds))
                action_level += 1

                progress += best_reward

                if verbose:
                    print ("==========progress========== ", progress)

                if found_completed_act :
                     break
            else:
                break

        return action_level, progress, env

    def back_up ( self, select_object = 0, verbose = False ):
        explorations = self.learn_one_setup(select_object, verbose)
        best = explorations[0]

        _, progress = best.get_observation_and_progress()
        action_level = len(best.action_storage)

        return action_level, progress, best

    def learn_one_setup( self, select_object = 0, verbose = False):
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
        # We keep the best exploration progress at best
        best = 0

        # We do one action at a time for all exploration
        # Loop index
        action_level = 0
        while True:
            if verbose:
                print ('action_level = %d' % action_level)
        
            # This would store a tuple of (exploration_index, accumulated_reward, action, action_means, action_stds)
            # branching ** 2
            tempo_rewards = []
            
            for exploration_index, exploration in enumerate(explorations):
                if verbose:
                    print ('exploration_index = %d' % exploration_index)

                no_of_search, state = self._get_no_of_search( exploration, action_level )

                #print ('state = ' + str(state))

                action_means, action_stds, actions = self._get_actions(select_object, exploration, no_of_search, verbose)

                #print (actions)

                for action_index, action in enumerate(actions):
                    _, reward, done, _ = exploration.step((select_object, action, action_means, action_stds))
                    #print ((action, reward))
                    exploration.back()

                    tempo_rewards.append( (exploration_index, rewards[exploration_index] + reward,
                        action, action_means, action_stds) )

                    if done:
                        if verbose:
                            print ("=== found_completed_act ===")
                        found_completed_act = True

            tempo_rewards = sorted(tempo_rewards, key = lambda t: t[1], reverse = True)
            test = [(t[0], t[1]) for t in tempo_rewards]

            if verbose:
                print ('=== Best explorations ===')
                print (test[:keep_branching])

            if test[0][1] == best:
                if verbose:
                    print ("--- No more progress ---")
                    print ('Best progress value = %.3f' % best)
                break

            best = test[0][1]

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

            action_level += 1

        return explorations
