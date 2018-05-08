"""
This file is created on 5/6/2018

Based on an older research notebook at notebooks/incorporate_feedback.ipynb

The idea is incorporating cold feedback and generating updated models


Improved learned progress function
by updating with human-evaluted scores of demonstrated actions

- Score of demonstration: 0 to 10
- Quality of demonstration: a binary value indicating whether a demonstration is good or bad. 

Using the grades that we get from annotators (which ranges from 0 to 10), we can pick out the very good and very bad demonstrations, by put a upper and lower threshold on the grades. 
We could to choose some consistent thresholds for all action types, or choose threshold depending on how many good or bad samples we received. 
A good landmark threshold for good and bad could be 6 and 3. 
We, therefore, can use any demonstrations that got higher than 6 to be additional positive samples, and ones that have grades lower than 3 as negative samples. 
The progress learner would be updated with those samples, and new demonstrations would be generated with the same initial configurations as all old demonstrations.
    
The reason we do not use the demonstration that got some medium score is that they are not obviously bad or good. 
Many of them start as a good demonstration, but make some mistake at the end. 
Technically speaking, they are still ``bad demonstrations'', but how to use them as negative samples with cold feedback method is not obvious. 
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
# import project
# # Need to add this import to load class
# from project import Project
from rl import block_movement_env
# import matplotlib
# from matplotlib import pyplot as plt
# import plotting
from test_all_searcher import get_model, get_default_models


lower_threshold = 3.5
higher_threshold = 7

def create_batch_size ( samples, batch_size ):
    if len(samples) < batch_size:
        repeat = batch_size // len(samples)
        remain = batch_size % len(samples)
        
        q = samples * repeat + samples[:remain]
        return np.stack(q)

def retrain_action ( project_name , lower_threshold, higher_threshold, sess, feedback_file = None,
                project_path = None, progress_path = None, output_file = None, episode = 10 ):
    print ('=== For ' + project_name)
    print ('%d & %d' % (lower_threshold, higher_threshold))

    if feedback_file is None:
        feedback_file = os.path.join('experiments', 'human_evaluation_2d', project_name.lower() + '.txt')

    with open(feedback_file, 'r') as fh:
        bad_demonstration_indices = []
        good_demonstration_indices = []
        for line in fh:
            values = [int(v) for v in line.split()]
            index = values[0]
            average_score = np.mean(values[1:])
            if average_score <= lower_threshold:
                bad_demonstration_indices.append(index)
            if average_score >= higher_threshold:
                good_demonstration_indices.append(index)

    print ('There are %d bad demonstrations which are: %s' % (len(bad_demonstration_indices), str(bad_demonstration_indices)))
    print ('There are %d good demonstrations which are: %s' % (len(good_demonstration_indices), str(good_demonstration_indices)))

    prefix = os.path.join("experiments", "human_evaluation_2d" , project_name)
    prev_demonstrations = {}

    c, p, pe = get_model ( project_name , sess)

    for index in range(30):
        stored_config_file = os.path.join(prefix, str(index) + ".dat")
        try:
            with open(stored_config_file, 'rb') as fh:
                # need this encoding 
                stored_config = pickle.load(fh, encoding='latin-1')
                
                e = block_movement_env.BlockMovementEnv(c, p.speed, p.name, 
                        progress_estimator = pe, session = sess)
                e.reset_env_to_state(stored_config['start_config'], stored_config['action_storage'])
                
                prev_demonstrations[index] = e
        except FileNotFoundError as e:
            print (e)

    # Retrain some new models here

    negative_samples = []
    negative_samples += [prev_demonstrations[index].get_feature_only() for index in bad_demonstration_indices if index in prev_demonstrations]

    negative_data = create_batch_size(negative_samples, c.batch_size)

    positive_samples = []
    positive_samples += [prev_demonstrations[index].get_feature_only() for index in good_demonstration_indices if index in prev_demonstrations]

    positive_data = create_batch_size(positive_samples, c.batch_size)


    for i in range(episode):
        print ('-------------------------------')
        lr_decay = c.lr_decay ** i
        new_learning_rate = c.learning_rate * 0.05 * lr_decay
        print ('Rate = %.5f' % (new_learning_rate))
        pe.assign_lr(new_learning_rate, sess = sess)
        
        # Update with negative samples
        pe.update(negative_data, np.zeros(c.batch_size), sess = sess)
        
        # Update with positive samples
        pe.update(positive_data, np.ones(c.batch_size), sess = sess)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    if output_file is None:
        output_file = os.path.join('learned_models', 'progress_' + project_name + '.mod.updated')
    saver.save(sess, output_file)

    print ('Model is saved at %s' % output_file)

if __name__ == '__main__':
    tf.reset_default_graph()
    sess =  tf.Session()

    action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]
    action_types = ["SlideAround"]

    for project_name in action_types:
        if project_name == 'SlideAway':
            lower_threshold = 4.5

        if project_name == 'SlideAround':
            lower_threshold = 1.5
            higher_threshold = 5.5

        retrain_action ( project_name ,lower_threshold, higher_threshold, sess )