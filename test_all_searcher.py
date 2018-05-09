import os
import sys
import numpy as np
import pickle

sys.path.append(os.path.join("strands_qsr_lib", "qsr_lib", "src3"))

import project
# Need to add this import to load class
from project import Project
import config
import progress_learner
from rl import block_movement_env
from rl import action_learner_search as als
from rl import discrete_action_learner_search as dals
import math
from importlib import reload
import automatic_evaluator
import pickle
import time
import argparse

import tensorflow as tf

action_levels = []
progresses = []
scores = []
times = []

def reset():
    global action_levels, progresses, scores, times
    action_levels = []
    progresses = []
    scores = []
    times = []

def get_default_models( action_types, sess ):
    """
    Get all action progress learners with defaults loading 
    """
    import tensorflow as tf
    import project
    # Need to add this import to load class
    from project import Project
    import config
    import progress_learner

    projects = {}
    progress_estimators = {}
    configs = {}

    for project_name in action_types:
        progress_estimator.config = config.Config()
        if project_name == 'SlideNext':
            progress_estimator.config.n_input = 8
            
        print ('========================================================')
        print ('Load for action type = ' + project_name)
        p_name = project_name.lower() + "_project.proj"

        p = project.Project.load(os.path.join('learned_models', p_name))
        
        with tf.variable_scope("model") as scope:
            print('-------- Load progress model ---------')
            progress_estimator = progress_learner.EventProgressEstimator(is_training=True, 
                                                                                        is_dropout = False, 
                                                                                        name = p.name, 
                                                                                        config = progress_estimator.config)  
            
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

        saver.restore(sess, os.path.join('learned_models', 'progress_' + project_name + '.mod'))

    return configs, projects, progress_estimators

def get_model ( project_name, sess, project_path = None, progress_path = None):
    import tensorflow as tf
    import project
    # Need to add this import to load class
    from project import Project
    import config
    import progress_learner

    c = config.Config()
    if project_name == 'SlideNext':
        c.n_input = 8
        
    print ('========================================================')
    print ('Load for action type = ' + project_name)

    if project_path is None:
        project_path = os.path.join('learned_models', project_name.lower() + "_project.proj")

    if progress_path is None:
        progress_path = os.path.join('learned_models', 'progress_' + project_name + '.mod')

    p = project.Project.load(project_path)
    
    with tf.variable_scope("model") as scope:
        print('-------- Load progress model ---------')
        pe = progress_learner.EventProgressEstimator(is_training=True, 
                                                    is_dropout = False, 
                                                    name = p.name, 
                                                    config = c)  
    
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
        print (variable.name)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))
    saver.restore(sess, progress_path)

    print ('Load progress learner from ' + progress_path)

    return p, pe

ALL = 'ALL'
GREEDY = 'GREEDY'
BACKUP = 'BACKUP'
CONTINUOUS = 'CONTINUOUS'
DISCRETE = 'DISCRETE'

TEST_FUNCION = {
    "SlideToward" : lambda exploration: automatic_evaluator.test_slide_close(exploration, threshold = 3.5 * config.Config().block_size),
    "SlideAway" : lambda exploration: automatic_evaluator.test_slide_away(exploration, ratio_threshold = 2.3),
    "SlideNext" : lambda exploration: automatic_evaluator.test_slide_nextto(exploration, angle_diff = 0.05 * np.pi, threshold = 1.7),
    "SlidePast" : lambda exploration: automatic_evaluator.test_slide_past(exploration, side_ratio = 1.1, angle_threshold = np.pi / 2), 
    "SlideAround" : lambda exploration: automatic_evaluator.test_slide_around(exploration, alpha_1 = 1.1 * np.pi, alpha_2 = 1.7 * np.pi)
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test searcher.')

    parser.add_argument('-a', '--action', action='store', metavar = ('ACTION'),
                                help = "Action type. Choose from 'SlideToward', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideAround'" )
    parser.add_argument('-n', '--size', action='store', metavar = ('SIZE'),
                                help = "Number of configurations " )
    parser.add_argument('-l', '--algorithm', action='store', metavar = ('ALGORITHM'),
                                help = "Choose one of the followings: ALL, GREEDY, BACKUP, CONTINUOUS, DISCRETE. Default is ALL" )
    parser.add_argument('-p', '--progress', action='store', metavar = ('PROGRESS'),
                                help = "Path of progress file. Default is 'learned_models/progress_' + project_name + '.mod'" )

    args = parser.parse_args()
    size = int(args.size)
    project_name = args.action
    algorithm = args.algorithm
    progress_path = args.progress

    ### MAIN CODE
    tf.reset_default_graph()
    sess =  tf.Session()

    p, progress_estimator = get_model (project_name, sess, progress_path = progress_path)


    # Save it down so we can load it later
    STORE_ENVS = "stored_envs.dat"
    if os.path.isfile(STORE_ENVS):
        print ('Load ' + STORE_ENVS)
        stored_envs = pickle.load( open( STORE_ENVS, "rb" ) )
    else:
        print ('Create ' + STORE_ENVS)
        stored_envs = []

        for i in range(50):
            # Just generate some environment to store 
            e = block_movement_env.BlockMovementEnv(config.Config(), 1, session = sess)
            
            stored_envs.append(e.start_config)
        pickle.dump( stored_envs, open( STORE_ENVS, "wb" ) )

    verbose = False

    averaged_action_levels = []
    averaged_progress = []
    averaged_scores = []
    averaged_times = []
    
    test_function = TEST_FUNCION[project_name]
    print ('=====================')
    print (project_name)

    def add_stat (action_level, progress, exploration):
        action_levels.append(action_level)
        progresses.append(progress)
        scores.append( test_function ( exploration ) )

    def summary_state ( ):
        print ('Average action level = %.2f' % np.average(action_levels) )
        print ('Average progress = %.2f' % np.average(progresses) )
        print ('Average score = %.2f' % np.average(scores) )
        print ('Average time = %.2f' % np.average(times) )

        averaged_action_levels.append(np.average(action_levels))
        averaged_progress.append(np.average(progresses))
        averaged_scores.append(np.average(scores))
        averaged_times.append(np.average(times))

    e = block_movement_env.BlockMovementEnv(progress_estimator.config, speed = p.speed, 
                                                progress_estimator = progress_estimator,
                                                session = sess)

    # if algorithm in [ALL, GREEDY, CONTINUOUS]:
    #     reset()
    #     for i in range(size):
    #         print (i)
    #         start_time = time.time()
    #         ## GREEDY
    #         # ==================
    #         e.reset_env_to_state(stored_envs[i], [])
    #         searcher = als.ActionLearner_Search(progress_estimator.config, p, 
    #                                             progress_estimator, session = sess, env = e)
    #         action_level, progress, exploration = searcher.greedy(verbose = verbose)
    #         add_stat (action_level, progress, exploration)
    #         times.append(time.time() - start_time)

    #     print ('GREEDY CONTINUOUS')
    #     summary_state()
    
    # if algorithm in [ALL, GREEDY, DISCRETE]:
    #     reset()
    #     for i in range(size):
    #         print (i)
    #         start_time = time.time()
    #         ## GREEDY    
    #         # ==================
    #         e.reset_env_to_state(stored_envs[i], [])
    #         searcher = dals.Discrete_ActionLearner_Search(progress_estimator.config, p, 
    #                                             progress_estimator, session = sess, env = e)
    #         action_level, progress, exploration = searcher.greedy(verbose = verbose)
    #         add_stat (action_level, progress, exploration)
    #         times.append(time.time() - start_time)

    #     print ('GREEDY DISCRETE')
    #     summary_state()
    
    # if algorithm in [ALL, BACKUP, CONTINUOUS]:
    #     reset()
    #     for i in range(size):
    #         print (i)
    #         start_time = time.time()
    #         ## BACK UP SEARCH
    #         # ==================
    #         e.reset_env_to_state(stored_envs[i], [])
    #         searcher = als.ActionLearner_Search(progress_estimator.config, p, 
    #                                             progress_estimator, session = sess, env = e)
    #         action_level, progress, exploration = searcher.back_up(verbose = verbose)
    #         add_stat (action_level, progress, exploration)
    #         times.append(time.time() - start_time)

    #     print ('BACKUP CONTINUOUS')
    #     summary_state()
    
    if algorithm in [ALL, BACKUP, DISCRETE]:
        reset()
        for i in range(size):
            print (i)
            start_time = time.time()
            ## BACK UP SEARCH    
            # ==================
            e.reset_env_to_state(stored_envs[i], [])
            searcher = dals.Discrete_ActionLearner_Search(progress_estimator.config, p, 
                                                progress_estimator, session = sess, env = e)
            action_level, progress, exploration = searcher.back_up(verbose = verbose)
            add_stat (action_level, progress, exploration)
            times.append(time.time() - start_time)

        print ('BACKUP DISCRETE')
        summary_state()