import os
import sys
import collections
import tensorflow as tf
from importlib import reload
import time
import numpy as np
import argparse

sys.path.append("strands_qsr_lib\qsr_lib\src3")

## PLOTTING 
import matplotlib
from matplotlib import pyplot as plt
import plotting


### IMPORT FROM CURRENT PROJECT
import progress_learner
import config
import project
from project import Project

### RL module
from rl import action_learner, action_learner_search, value_estimator
from rl import block_movement_env

def print_action_prob(policy_est, progress_state = True):
    if progress_state:
        for progress in range(5):
            for pos in range(6):
                for prev_action in range(5):
                    if (pos == 0 or pos == 3) and prev_action == 1:
                        # Illegal
                        continue

                    if (pos == 2 or pos == 5) and prev_action == 3:
                        # Illegal
                        continue

                    state = np.zeros(150)
                    index = int(pos * 25 + prev_action * 5 + progress)

                    state[index] = 1

                    probs = policy_est.predict(state, sess = sess)

                    best_action = np.argmax(probs)
                    print ('%d & %d & %d & %s & %d' % (pos, prev_action, progress, ','.join(('%.3f' % p) for p in probs), best_action) )
    else:
        for pos in range(6):
            for prev_action in range(5):
                if (pos == 0 or pos == 3) and prev_action == 1:
                    # Illegal
                    continue

                if (pos == 2 or pos == 5) and prev_action == 3:
                    # Illegal
                    continue

                state = np.zeros(30)
                index = int(pos * 5 + prev_action)

                state[index] = 1

                probs = policy_est.predict(state, sess = sess)

                best_action = np.argmax(probs)
                print ('%d & %d & %s & %d' % (pos, prev_action, ','.join(('%.3f' % p) for p in probs), best_action) )

def print_value_est(value_est, progress_state = True):
    if progress_state:
        for progress in range(5):
            for pos in range(6):
                for prev_action in range(5):
                    if (pos == 0 or pos == 3) and prev_action == 1:
                        # Illegal
                        continue

                    if (pos == 2 or pos == 5) and prev_action == 3:
                        # Illegal
                        continue
                    
                    state = np.zeros(150)
                    index = int(pos * 25 + prev_action * 5 + progress)

                    state[index] = 1

                    val = value_est.predict(state, sess = sess)
                    print ('pos = %d, prev_action = %d, progress = %d, val = %.2f' % (pos, prev_action, progress, val) )
    else:
        for pos in range(6):
            for prev_action in range(5):
                if (pos == 0 or pos == 3) and prev_action == 1:
                    # Illegal
                    continue

                if (pos == 2 or pos == 5) and prev_action == 3:
                    # Illegal
                    continue
                
                state = np.zeros(30)
                index = int(pos * 5 + prev_action)

                state[index] = 1

                val = value_est.predict(state, sess = sess)
                print ('pos = %d, prev_action = %d, val = %.2f' % (pos, prev_action, val) )
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test continuous action learner.')

    parser.add_argument('-e', '--episode', action='store', metavar = ('EPISODE'),
                                help = "Number of episodes." )

    parser.add_argument('-m', '--model', action='store', metavar = ('MODEL'), default = 'ACTOR_CRITIC',
                                help = "Model type. Choose ACTOR_CRITIC or REINFORCE. Default is ACTOR_CRITIC" )

    parser.add_argument('-b', '--breadth', action='store', metavar = ('BREADTH'), default = 1,
                                help = "Breadth. Number of actions generated at each step. Default is 1" )

    parser.add_argument('-p', '--progress', action='store', metavar = ('BREADTH'), default = 1,
                                help = "Breadth. Number of actions generated at each step. Default is 1" )

    args = parser.parse_args()

    breadth = int(args.breadth)
    num_episodes = int(args.episode)

    model_type = args.model

    if model_type not in ['ACTOR_CRITIC', 'REINFORCE']:
        model_type = 'ACTOR_CRITIC'


    ### MAIN CODE
    tf.reset_default_graph()

    c = config.Config()

    global_step = tf.Variable(0, name="global_step", trainable=False)


    policy_est = value_estimator.PolicyEstimator(c)
    value_est = value_estimator.ValueEstimator(c)


    sess =  tf.Session()

    sess.run(tf.global_variables_initializer())

    projects = {}
    progress_estimators = {}

    action_types = ["SlideAround"]

    for project_name in action_types:
        print ('========================================================')
        print ('Load for action type = ' + project_name)
        p_name = project_name.lower() + "_project.proj"

        projects[project_name] = project.Project.load(os.path.join('learned_models', p_name))

        with tf.variable_scope("model") as scope:
            print('-------- Load progress model ---------')
            progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training = False,
                                                                                        is_dropout = False, 
                                                                                        name = projects[project_name].name, 
                                                                                        config = c)  

    # Print out all variables that would be restored
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
        print (variable.name)

    for project_name in action_types:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

        saver.restore(sess, os.path.join('learned_models', 'progress_' + project_name + '.mod.1'))


    start_time = time.time()

    c.num_episodes = num_episodes

    action_ln = action_learner.ActionLearner(c, projects['SlideAround'], progress_estimators['SlideAround'], 
                               policy_est, value_est, session = sess, limit_step = 12)

    past_envs, stats, es_stats = action_ln.policy_learn(action_learner_search.action_policy(c), breadth = breadth, verbose = False,
                                              choice = model_type, default = False)


    print ('Run finish after %d' % (time.time() -start_time))

    print_action_prob(policy_est)

    print_value_est(value_est)

    plotting.plot_episode_stats(stats, smoothing_window=50)