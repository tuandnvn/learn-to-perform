import tensorflow as tf
import os
import sys
import collections

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
a = os.path.join(module_path, "strands_qsr_lib\qsr_lib\src3")
sys.path.append(a)

import ensemble_learner
from rl import action_learner, action_learner_search, value_estimator
import progress_learner
import config
import project
# Need to add this import to load class
from project import Project
from importlib import reload
from rl import block_movement_env
import matplotlib
from matplotlib import pyplot as plt
import plotting

def action_policy(config):
    """
    Given a config that has defined a playground
    """
    def boundary_constraint(action):
        for i in range(3):
            if action[i] < config.playground_x[i]:
                return False
            if action[i] > config.playground_x[i] + config.playground_dim[i]:
                return False
        
        return True
    
    def q(state, policy_estimator, no_of_actions = 1, verbose = False, 
       session = None):
        return action_learner_search.random_action_constraint(state, policy_estimator,
                    no_of_actions, verbose, session, boundary_constraint)
    
    return q

def create_ensemble_learner():
    c = config.Config()
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_est = action_learner_search.PolicyEstimator(c)
    sess =  tf.Session()
    sess.run(tf.global_variables_initializer())

    projects = {}
    progress_estimators = {}

    action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]

    for project_name in action_types:
        print ('========================================================')
        print ('Load for action type = ' + project_name)
        p_name = project_name.lower() + "_project.proj"

        projects[project_name] = project.Project.load('../' + p_name)

        with tf.variable_scope("model") as scope:
            print('-------- Load progress model ---------')
            progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=False, name = projects[project_name].name, config = c)  

    # Print out all variables that would be restored
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
        print (variable.name)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))

    for project_name in action_types:
        saver.restore(sess, '../progress_' + project_name + '.mod')

    ensemble_learner = ensemble_learner.Ensemble_Learner(config, action_types, projects, progress_estimators, 
            policy_estimator, action_policy, limit_step = 4, session = sess)

    return ensemble_learner

# if __name__ == '__main__':
#     ensemble_learner = create_ensemble_learner()

#     # (0,0.7,0.5), Extends (0.9, 0, 0.7)
#     table = [(0,0.7,0.5), (0.9, 0, 0.7)]
#     block1 = [-0.4, 0.8, 0.2]
#     block2 = [0.1, 0.8, 0.6]
#     ensemble_learner.receive_state( (table, block1, block2) )

#     # Slide block1 Past block2
#     exploration, command = ensemble_learner.produce_action( "SlidePast" , 0 )

