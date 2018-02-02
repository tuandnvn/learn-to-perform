import tensorflow as tf
import os
import sys
import collections
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
a = os.path.join(module_path, "strands_qsr_lib\qsr_lib\src3")

sys.path.append(a)
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
import pickle

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

# action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]
action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast"]

for project_name in action_types:
    print ('========================================================')
    print ('Load for action type = ' + project_name)
    p_name = project_name.lower() + "_project.proj"

    projects[project_name] = project.Project.load(p_name)

    with tf.variable_scope("model") as scope:
        print('-------- Load progress model ---------')
        progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=True, name = projects[project_name].name, config = c) 

feedbacks = {}
for project_name in action_types:
	feedback = os.path.join('experiments', 'human_evaluation_2d', 'feedback', project_name.lower() + '.txt')
	feedbacks[project_name] = feedback

	with open(feedback, 'r') as fh:
	    wrong_demonstration_indices = []
	    correct_demonstration_indices = []
	    for line in fh:
	        values = [int(v) for v in line.split()]
	        index = values[0]
	        average_score = np.mean(values[1:])
	        if average_score <= 2.5:
	            wrong_demonstration_indices.append(index)
	        if average_score >= 6:
	            correct_demonstration_indices.append(index)

	print (len(wrong_demonstration_indices))
	print (wrong_demonstration_indices)
	print (len(correct_demonstration_indices))
	print (correct_demonstration_indices)

	

