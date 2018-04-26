import os
import sys
import collections
import tensorflow as tf
from importlib import reload

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
a = os.path.join(module_path, "strands_qsr_lib\qsr_lib\src3")

sys.path.append(a)

## PLOTTING 
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
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


### RELOAD modules here
# reload(config)
# reload(value_estimator)
# reload(config)
# reload(block_movement_env)
# reload(action_learner_search)
# reload(progress_learner)
# reload(dal)
# reload(dve)


### AUXILIARY CODE
discretized_space = [0.18, 0.36, 0.72]
discretized_rotation = np.pi/8 

uniform_space = dal.MultiDiscreteNoZero( [(-len(discretized_space), len(discretized_space)), 
            (-len(discretized_space), len(discretized_space)), 
            (0, int((np.pi/2) // discretized_rotation)) ], (0,2) )

def action_policy ( state, policy_estimator, no_of_actions, verbose, session ):
    return dal.epsilon_greedy_action_2( state, policy_estimator, uniform_space, no_of_actions = no_of_actions, verbose= verbose,
                                       session = session, epsilon_1 = 0.6, epsilon_2 = 0 )


### MAIN CODE
tf.reset_default_graph()

c = config.Config()

global_step = tf.Variable(0, name="global_step", trainable=False)


policy_est = dve.DiscretePolicyEstimator(c)
value_est = value_estimator.ValueEstimator(c)

sess =  tf.Session()

sess.run(tf.global_variables_initializer())

projects = {}
progress_estimators = {}

# action_types = ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]
action_types = ["SlideAround"]

for project_name in action_types:
    print ('========================================================')
    print ('Load for action type = ' + project_name)
    p_name = project_name.lower() + "_project.proj"

    projects[project_name] = project.Project.load('../' + p_name)

    with tf.variable_scope("model") as scope:
        print('-------- Load progress model ---------')
        progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=True, name = projects[project_name].name, config = c)  

# Print out all variables that would be restored
for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
    print (variable.name)

for project_name in action_types:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

    saver.restore(sess, '../progress_' + project_name + '.mod.1')

c.num_episodes = 1000
action_ln = dal.DiscreteActionLearner(c, projects['SlideAround'], progress_estimators['SlideAround'], 
                               policy_est, value_est, session = sess, limit_step = 25)

_, stats = action_ln.policy_learn(action_policy, breadth = 5, verbose = True,
                                          choice = 'ACTOR_CRITIC', default = True)

with open('session_data_actor_critic_disrete', 'wb') as f:
    pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

print('----Done saving stats data ---')