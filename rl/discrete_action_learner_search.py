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
from importlib import reload
import math

def grid_random_action(c, no_of_actions = 1, verbose = False):
    """
    Random actions in the slot around the grid point

    c: config


    return: List of (action_mean, action_std, action)
    """
    sqrt_action = math.sqrt(no_of_actions)

	x_coordinate = np.linspace( c.playground_x[0] + c.playground_dim[0] / (2 * sqrt_action) , c.playground_x[0] + c.playground_dim[0] - c.playground_dim[0] / (2 * sqrt_action), sqrt_action )
	y_coordinate = np.linspace( c.playground_x[1] + c.playground_dim[1] / (2 * sqrt_action) , c.playground_x[1] + c.playground_dim[1] - c.playground_dim[1] / (2 * sqrt_action), sqrt_action )

	action_std = np.array([c.playground_dim[0] / (2 * sqrt_action), c.playground_dim[1] / (2 * sqrt_action)])
    variances = action_std ** 2

    actions = []
    action_means = []
    action_stds = action_std * no_of_actions

    while True:
        tempo = np.random.multivariate_normal(np.zeros(2), np.diag(variances), size = sqrt_action ** 2)

        actions += list(tempo + np.array(list(zip(*(x.flat for x in np.meshgrid(x, y))))))
        action_means += list(np.array(list(zip(*(x.flat for x in np.meshgrid(x, y))))))

        if len(actions) > no_of_actions:
            break

    return action_means[:no_of_actions], action_stds, actions[:no_of_actions]

class Discrete_ActionLearner_Search(object):
    """

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

        self.limit_step = limit_step

        self.session = session

        self.np_random, _ = seeding.np_random(None)

        if env == None:
            env = bme.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                progress_estimator = self.progress_estimator, session = self.session)
            env.reset()
        
        self.env = env

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
        for action_level in range(1):
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

                actions_with_means_and_stds = grid_random_action(self.config, no_of_actions = no_of_search)

                tuple_actions = [(select_object, action, action_mean, action_std) for action_mean, action_std, action in actions_with_means_and_stds]
                legal_action_indices, all_progress = exploration.try_step_multi(tuple_actions)

                for index, progress in zip (legal_action_indices, all_progress):
                    tempo_rewards.append( (exploration_index, progress,
                        actions[index], action_means, action_stds) )

                    if progress > self.config.progress_threshold:
                        print ("=== found_completed_act ===")
                        found_completed_act = True

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