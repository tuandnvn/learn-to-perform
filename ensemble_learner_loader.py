import tensorflow as tf
import os
import sys
import collections

sys.path.append( os.path.join( "strands_qsr_lib", "qsr_lib", "src3") )

from importlib import reload
import ensemble_learner

from rl import action_learner, action_learner_search, value_estimator
import progress_learner
import config
import project
# Need to add this import to load class
from project import Project

from rl import block_movement_env
import matplotlib
from matplotlib import pyplot as plt
import plotting

reload(ensemble_learner)
reload(config)

def create_ensemble_learner():
    c = config.Config()
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # policy_est = action_learner_search.PolicyEstimator(c)
    sess =  tf.Session()
    sess.run(tf.global_variables_initializer())

    projects = {}
    progress_estimators = {}

    action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]
    # action_types = ["SlideToward", "SlideAway"]

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

    for project_name in action_types:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

        saver.restore(sess, '../progress_' + project_name + '.mod')

    learner = ensemble_learner.Ensemble_Learner(c, action_types, projects, progress_estimators, limit_step = 4, session = sess)

    return learner

# if __name__ == '__main__':
#     ensemble_learner = create_ensemble_learner()

#     # (0,0.7,0.5), Extends (0.9, 0, 0.7)
#     table = [(0,0.7,0.5), (0.9, 0, 0.7)]
#     block1 = [-0.4, 0.8, 0.2]
#     block2 = [0.1, 0.8, 0.6]
#     ensemble_learner.receive_state( (table, block1, block2) )

#     # Slide block1 Past block2
#     exploration, command = ensemble_learner.produce_action( "SlidePast" , 0 )

