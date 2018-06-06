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
from test_all_searcher import get_model, get_default_models, TEST_FUNCION

def create_batch_size ( samples, batch_size ):
    """
    A list pack samples into batch of $batch_size items
    Fill it up by copying same elements to other locations
    """
    if len(samples) <= batch_size:
        repeat = batch_size // len(samples)
        remain = batch_size % len(samples)
        
        q = samples * repeat + samples[:remain]
        return [np.stack(q)]
    else:
        # Round up
        num = (len(samples) - 1) // batch_size + 1

        q = samples + samples [: batch_size - (len(samples) - 1) % batch_size - 1]

        return [np.stack ( q[i * batch_size : (i + 1) * batch_size]) for i in range( num )] 

def get_prev_demonstration ( p, pe, stored_config_prefix = None ) :
    """
    Open all files of name (number +_ '.dat') in stored_config_prefix directory
    and load demonstrations from them

    Parameters:
    ==============

    Returns:
    ==============
    """
    if stored_config_prefix is None:
        stored_config_prefix = os.path.join("experiments", "human_evaluation_2d" , project_name)

    prev_demonstrations = {}

    for index in range(30):
        stored_config_file = os.path.join(stored_config_prefix, str(index) + ".dat")
        try:
            with open(stored_config_file, 'rb') as fh:
                # need this encoding 
                stored_config = pickle.load(fh, encoding='latin-1')
                
                e = block_movement_env.BlockMovementEnv(pe.config, p.speed, p.name, 
                        progress_estimator = pe, session = sess)
                e.reset_env_to_state(stored_config['start_config'], stored_config['action_storage'])
                
                prev_demonstrations[index] = e
        except FileNotFoundError as e:
            print (e)

    return prev_demonstrations

def demonstration_evaluator ( demonstration_name, demonstration ):
    # A default demonstration evaluator
    # that use either a demonstration_name or demonstration itself to evaluate
    return 1

def human_feedback_evaluator ( lower_threshold, higher_threshold, feedback_file = None ):
    """
    Return a function similar to demonstration_evaluator

    Based on score given from a feedback_file
    """
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

    def feedback_evaluator (demonstration_name, demonstration) :
        """
        Using demonstration_name as demonstration_index
        """
        if demonstration_name in bad_demonstration_indices:
            return 0

        if demonstration_name in good_demonstration_indices:
            return 1

        return None

    return feedback_evaluator

def automatic_feedback_evaluator ( action_type ):
    test_function = TEST_FUNCION[action_type]

    def feedback_evaluator (demonstration_name, demonstration) :
        if test_function ( demonstration ) == 1:
            return 1

        if test_function ( demonstration ) == 0:
            return 0

        return None

    return feedback_evaluator


def retrain_action ( project_name, sess, stored_config_prefix = None, demonstration_evaluator = None,
                project_path = None, progress_path = None, output_file = None, episode = 10 ):
    """
    Parameters:
    ==============

    Returns:
    ==============
    """
    print ('=== For ' + project_name)

    p, pe = get_model ( project_name , sess)

    prev_demonstrations = get_prev_demonstration ( p, pe, stored_config_prefix )

    negative_samples = []
    positive_samples = []

    for index, demonstration in prev_demonstrations.items():
        val = demonstration_evaluator ( index, demonstration )
        print (index, val)

        if val == 0:
            negative_samples += [demonstration.get_feature_only()]
        elif val == 1:
            positive_samples += [demonstration.get_feature_only()]

    negative_data = create_batch_size ( negative_samples, pe.config.batch_size )
    positive_data = create_batch_size ( positive_samples, pe.config.batch_size )

    # Retrain a new model here
    update_model( pe, project_name, episode, sess, positive_data, negative_data, output_file )

def update_model ( progress_estimator, project_name, episode, sess, positive_data, negative_data, output_file = None ):
    """
    Update a model saved in progress_estimator

    Parameters:
    ==============

    Returns:
    ==============
    """
    c = progress_estimator.config
    for i in range(episode):
        print ('-------------------------------')
        lr_decay = c.lr_decay ** i
        new_learning_rate = c.learning_rate * 0.05 * lr_decay
        print ('Rate = %.5f' % (new_learning_rate))
        progress_estimator.assign_lr(new_learning_rate, sess = sess)
        
        # Update with negative samples
        progress_estimator.update(negative_data, np.zeros(c.batch_size), sess = sess)
        
        # Update with positive samples
        progress_estimator.update(positive_data, np.ones(c.batch_size), sess = sess)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    if output_file is None:
        output_file = os.path.join('learned_models', 'progress_' + project_name + '.mod.updated.2')
    saver.save(sess, output_file)

    print ('Model is saved at %s' % output_file)

if __name__ == '__main__':
    import argparse

    AUTOMATIC = 'AUTOMATIC'
    HUMAN = 'HUMAN'

    tf.reset_default_graph()
    sess =  tf.Session()

    parser = argparse.ArgumentParser(description='This file load a progress learner and update it with demonstrations from a directory.\
                Note that these demonstrations need to be generated with this progress learner.')

    parser.add_argument('-a', '--action', action='store', metavar = ('ACTION'), 
                                help = "Action type. Choose from 'SlideToward', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideAround'" )

    parser.add_argument('-m', '--mode', action='store', metavar = ('MODE'), 
                                help = "Choose between HUMAN and AUTOMATIC. HUMAN mode run experiments with real human scores. AUTOMATIC uses automatic oracles.")

    parser.add_argument('-m', '--mode', action='store', metavar = ('MODE'), 
                                help = "Choose between HUMAN and AUTOMATIC. HUMAN mode run experiments with real human scores. AUTOMATIC uses automatic oracles.")

    args = parser.parse_args()
    project_name = args.action
    mode = args.mode

    if mode == HUMAN:
        lower_threshold = 3.5
        higher_threshold = 7

        if project_name == 'SlideAway':
            lower_threshold = 4.5

        if project_name == 'SlideAround':
            lower_threshold = 1.5
            higher_threshold = 5.5

        demonstration_evaluator = human_feedback_evaluator ( project_name, lower_threshold, higher_threshold )

    if mode == AUTOMATIC:
        demonstration_evaluator = automatic_feedback_evaluator ( project_name ) 

    retrain_action ( project_name, sess, demonstration_evaluator = demonstration_evaluator, 
        stored_config_prefix = os.path.join("experiments", "human_evaluation_2d" , 'SlideAroundDiscrete') )