"""
This file is created on 5/6/2018

Based on an older research notebook at notebooks/incorporate_feedback.ipynb

The idea is incorporating cold feedback and generating updated models
"""

import tensorflow as tf
import os
import sys
import collections
import numpy as np
import pickle

sys.path.append("strands_qsr_lib\qsr_lib\src3")

import progress_learner
import config
import project
# Need to add this import to load class
from project import Project
from rl import block_movement_env
import matplotlib
from matplotlib import pyplot as plt
import plotting
import test_all_searcher


# tf.reset_default_graph()

# global_step = tf.Variable(0, name="global_step", trainable=False)

# sess =  tf.Session()

action_types = ["SlideCloser", "SlideAway", "SlideNext", "SlidePast"]

# configs, projects, progress_estimators = test_all_searcher.get_default_models( action_types, sess )

feedbacks = {}
for project_name in action_types:
	print ('=== For ' + project_name)
	feedback = os.path.join('experiments', 'human_evaluation_2d', project_name.lower() + '.txt')
	feedbacks[project_name] = feedback

	with open(feedback, 'r') as fh:
	    wrong_demonstration_indices = []
	    correct_demonstration_indices = []
	    for line in fh:
	        values = [int(v) for v in line.split()]
	        index = values[0]
	        average_score = np.mean(values[1:])
	        if average_score <= 2.5:
	            wrong_demonstration_indices.append((index, average_score))
	        if average_score >= 6:
	            correct_demonstration_indices.append((index, average_score))

	print ('There are %d bad demonstrations which are: %s' % (len(wrong_demonstration_indices), str(wrong_demonstration_indices)))
	print ('There are %d good demonstrations which are: %s' % (len(correct_demonstration_indices), str(correct_demonstration_indices)))