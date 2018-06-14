import os
import sys
import collections
import tensorflow as tf
from importlib import reload
import time
import numpy as np
import argparse

sys.path.append( os.path.join( "strands_qsr_lib", "qsr_lib", "src3") )

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
from rl import discrete_value_estimator as  dve
from rl import discrete_action_learner as dal
from test_all_searcher import get_model

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
    parser = argparse.ArgumentParser(description='Test discrete action learner.')

    parser.add_argument('-e', '--episode', action='store', metavar = ('EPISODE'),
                                help = "Number of episodes." )

    parser.add_argument('-m', '--model', action='store', metavar = ('MODEL'), default = 'ACTOR_CRITIC',
                                help = "Model type. Choose ACTOR_CRITIC or REINFORCE. Default is ACTOR_CRITIC" )

    parser.add_argument('-b', '--breadth', action='store', metavar = ('BREADTH'), default = 1,
                                help = "Breadth. Number of actions generated at each step. Default is 1" )

    parser.add_argument('-p', '--progress_state', action='store', default = 'True',
                                help = "Whether to keep progress as state component. Default is True." )

    parser.add_argument('-f', '--progress', action='store', metavar = ('PROGRESS'),
                                help = "Path of progress file. Default is 'learned_models/progress_SlideAround.mod'" )

    parser.add_argument('-r', '--random', action='store', metavar = ('RANDOM'), default = 'POLICY',
                                help = "Action generation method. Default is 'POLICY', i.e., use Policy to generate action probablities. \n \
                                    Other choice is 'EPSILON', i.e., in epsilon x time, use uniform distribution instead." )

    args = parser.parse_args()

    breadth = int(args.breadth)
    num_episodes = int(args.episode)

    # ==========
    if args.progress_state == 'True':
        progress_state = True
    else:
        progress_state = False

    print ('progress_state = %r' % progress_state)

    # ==========
    progress_path = args.progress

    print ('progress_path = %s' % progress_path)

    # ==========
    model_type = args.model

    if model_type not in ['ACTOR_CRITIC', 'REINFORCE']:
        model_type = 'ACTOR_CRITIC'

    print ('model_type = %s' % model_type)

    # ==========
    POLICY = 'POLICY'
    EPSILON = 'EPSILON'

    random_policy = POLICY # Default case
    if args.random == POLICY:
        random_policy = POLICY
    elif args.random == EPSILON:
        random_policy = EPSILON

    print ('random_policy = %s' % random_policy)


    ### MAIN CODE
    tf.reset_default_graph()

    c = config.Qual_Plan_Config()

    
    if progress_state:
        c.state_dimension = 150
    else:
        c.state_dimension = 30

    global_step = tf.Variable(0, name="global_step", trainable=False)


    policy_est = dve.DiscretePolicyEstimator(c)
    value_est = dve.ValueEstimator(c)

    sess =  tf.Session()

    sess.run(tf.global_variables_initializer())

    project_name = "SlideAround"

    if progress_path is None:
        progress_path = os.path.join('learned_models', 'progress_' + project_name + '.mod')
    
    p, pe = get_model ( project_name, sess, project_path = os.path.join('learned_models', project_name.lower() + "_project.proj"), 
        progress_path = progress_path)


    start_time = time.time()

    c.num_episodes = num_episodes

    action_ln = dal.DiscreteActionLearner(c, p, pe, 
                                   policy_est, value_est, session = sess, limit_step = 12, progress_state = progress_state)

    if random_policy == POLICY:
        random_action_function = dal.random_action
    elif random_policy == EPSILON:
        random_action_function = dal.e_greedy_random_action

    past_envs, stats, es_stats = action_ln.policy_learn(dal.random_action, breadth = breadth, verbose = False,
                                              choice = model_type, default = False)


    print ('Run finish after %d' % (time.time() -start_time))

    print_action_prob(policy_est, progress_state = progress_state)

    print_value_est(value_est, progress_state = progress_state)

    plotting.plot_episode_stats(stats, smoothing_window=50)