import tensorflow as tf
import os
import sys
import collections

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
a = os.path.join(module_path, "strands_qsr_lib\qsr_lib\src3")

sys.path.append(a)
from rl import action_learner, action_learner_search, value_estimator, discrete_action_learner_search
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
c.keep_branching = 24
c.branching = 24
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)

sess =  tf.Session()

with tf.variable_scope("search") as scope:
    policy_estimator = action_learner_search.PolicyEstimator(c)

sess.run(tf.global_variables_initializer())

projects = {}
progress_estimators = {}

# action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]
action_types = ["SlideAround"]

for project_name in action_types:
    print ('========================================================')
    print ('Load for action type = ' + project_name)
    p_name = project_name.lower() + "_project.proj"

    projects[project_name] = project.Project.load(p_name)

    with tf.variable_scope("model") as scope:
        print('-------- Load progress model ---------')
        progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=True, name = projects[project_name].name, config = c)  

# Print out all variables that would be restored
for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
    print (variable.name)

action_lns = {}
for project_name in action_types:
    action_lns[project_name] = discrete_action_learner_search.Discrete_ActionLearner_Search(c, projects[project_name], progress_estimators[project_name], session = sess)

prefix = os.path.join( "experiments", "human_evaluation_2d" , "SlideAround")
p = progress_estimators["SlideAround"]
al = action_lns["SlideAround"]
new_demonstrations = []
for index in range(30):
    stored_config_file = os.path.join(prefix, str(index) + ".dat")
    with open(stored_config_file, 'rb') as fh:
        # need this encoding 
        if sys.version_info >= (3,0):
            stored_config = pickle.load(fh, encoding='latin-1')
        else:
            stored_config = pickle.load(fh)

        e = block_movement_env.BlockMovementEnv(al.config, al.project.speed, al.project.name, 
                progress_estimator = p, session = al.session)
        e.reset_env_to_state(stored_config['start_config'], [])
        
        new_demonstrations.append(e)

new_prefix = os.path.join( "experiments", "human_evaluation_2d" , "SlideAroundDiscrete")

for project_name in action_types:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    saver.restore(sess, 'progress_' + project_name + '.mod.1')


for index in range(15, 30):
    print ('Re-demonstrate for ' + str(index))
    a = discrete_action_learner_search.Discrete_ActionLearner_Search(c, projects['SlideAround'], p, session = sess, env = new_demonstrations[index])
    explorations = a.learn_one_setup(verbose = True)
    explorations[0].save(os.path.join( new_prefix, str(index) + ".dat" ))
    explorations[0].save_visualization_to_file(os.path.join( new_prefix, str(index) + ".mp4" ))

for project_name in action_types:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    saver.restore(sess, 'progress_' + project_name + '.mod.2')


for index in range(0, 15):
    print ('Re-demonstrate for ' + str(index))
    a = discrete_action_learner_search.Discrete_ActionLearner_Search(c, projects['SlideAround'], p, session = sess, env = new_demonstrations[index])
    explorations = a.learn_one_setup(verbose = True)
    explorations[0].save(os.path.join( new_prefix, str(index) + ".dat" ))
    explorations[0].save_visualization_to_file(os.path.join( new_prefix, str(index) + ".mp4" ))