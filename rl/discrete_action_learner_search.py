from __future__ import print_function
import sys
    
import tensorflow as tf
import numpy as np
import collections
import itertools

from . import uniform_env_space
from . import block_movement_env as bme
from . import action_learner_search
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


def quantized_random_action(c, env, select_object, discretized_space = [0.18, 0.36, 0.72], no_of_actions = 1, verbose = False, constraint_function = lambda a : True):
    """
    Random actions in the slot around the grid point
    with the grid point being the ones that takes into 
    consideration the angle made with the static square

    c: config
    env: to get the positions of the objects in the current environment

    return: List of (action_mean, action_std, action)
    """
    if select_object == 0:
        static_object = 1
    else:
        static_object = 0

    static_object_transform = env.e.objects[static_object].transform
    theta = static_object_transform.rotation

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])


    actions = []

    while True:
        for ds in discretized_space:
            # Starting from straight north and going clockwise
            for values in [ (0, ds), (ds, ds), (ds, 0), (ds, -ds), (0, -ds), (-ds, -ds), (-ds, 0), (-ds, ds) ]:
                location = R.dot( np.array([[values[0]], [values[1]]]) )
                location = location.flatten() + np.reshape( static_object_transform.position, [2])
                action = np.concatenate( [location, np.array([theta])] )

                if constraint_function(action):
                    actions.append(action)
                    if len(actions) >= no_of_actions:
                        return None, None, actions

class Discrete_ActionLearner_Search(action_learner_search.ActionLearner_Search):
    """

    """
    def __init__(self, config, project, progress_estimator, session = None, env = None):
        super().__init__(config, project, progress_estimator, session = session, env = env)

    def _get_actions(self, select_object, exploration, no_of_search, verbose) :
        return quantized_random_action(self.config, exploration, select_object, no_of_actions = no_of_search)

    def _get_no_of_search(self, exploration, action_level):
        if action_level == 0:
            no_of_search = self.config.branching
            state = exploration.get_observation_start()
        else:
            no_of_search = self.config.branching
            state, _ = exploration.get_observation_and_progress()

        return no_of_search, state