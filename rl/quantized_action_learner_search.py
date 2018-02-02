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

def quantized_random_action(c, no_of_actions = 1, verbose = False):
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