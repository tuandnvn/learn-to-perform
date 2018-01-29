import tensorflow as tf
import os
import sys
import collections

from rl import action_learner, action_learner_search, value_estimator
import progress_learner
import config
import project
# Need to add this import to load class
from project import Project
# from importlib import reload
from rl import block_movement_env
import matplotlib
from matplotlib import pyplot as plt
import plotting

c = config.Config()
c.no_of_loops = 1
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)

sess =  tf.Session()

with tf.variable_scope("search") as scope:
    policy_estimator = action_learner_search.PolicyEstimator(c)

sess.run(tf.global_variables_initializer())

projects = {}
progress_estimators = {}

action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]

for project_name in action_types:
    print ('========================================================')
    print ('Load for action type = ' + project_name)
    p_name = project_name.lower() + "_project.proj"

    projects[project_name] = project.Project.load(p_name)

    with tf.variable_scope("model") as scope:
        print('-------- Load progress model ---------')
        progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=False, name = projects[project_name].name, config = c)  

# Print out all variables that would be restored
for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
    print (variable.name)

for project_name in action_types:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    saver.restore(sess, 'progress_' + project_name + '.mod')

c.progress_threshold=0.92
action_lns = {}
for project_name in action_types:
    action_lns[project_name] = action_learner_search.ActionLearner_Search(c, projects[project_name], progress_estimators[project_name], session = sess)

for project_name in action_types:
    prefix = os.path.join( "experiments", "human_evaluation_2d" , project_name)
    print ("============")
    print (prefix)
    al = action_lns[project_name]
    for n in range(30):
        al.env.reset()
        explorations = al.learn_one_setup(verbose = True)
        explorations[0].save(os.path.join( prefix, str(n) + ".dat" ))
        explorations[0].save_visualization_to_file(os.path.join( prefix, str(n) + ".mp4" ))