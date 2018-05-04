import os
import sys
import collections
import tensorflow as tf
from importlib import reload
import pickle

a = os.path.join("strands_qsr_lib", "qsr_lib ", "src3")

sys.path.append(a)

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
from rl import action_learner_search as als 
from rl import discrete_action_learner_search as dals
from rl import block_movement_env

class InteractiveLearner ( object ):
    def __init__(self, c = None, action_type = "SlideAround"):
        tf.reset_default_graph()

        if c is None:
            c = config.Config()

        self.config = c

        global_step = tf.Variable(0, name="global_step", trainable=False)

        self.sess = sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        projects = {}
        progress_estimators = {}

        if action_type is None:
            action_type = "SlideAround"

        
        print ('========================================================')
        print ('Load for action type = ' + action_type)
        p_name = action_type.lower() + "_project.proj"

        self.project = p = project.Project.load(os.path.join('learned_models', p_name))

        with tf.variable_scope("model") as scope:
            print('-------- Load progress model ---------')
            pe = progress_learner.EventProgressEstimator(is_training = False,
                                                        is_dropout = False, 
                                                        name = p_name, 
                                                        config = c)  

        # Print out all variables that would be restored
        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
            print (variable.name)

        for action_type in action_types:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + action_type))

            saver.restore(sess, os.path.join('learned_models', 'progress_' + action_type + '.mod.1'))

    def load_demo( self, demo_file ):
        """
        Set self.e to the initial state in demo_file if demo_file store a demonstration

        Parameters:
        -----
        demo_file: a .dat file that has been saved as picked file

        Returns:
        --------
        """
        with open(demo_file, 'rb') as fh:
            # need this encoding 
            if sys.version_info >= (3,0):
                stored_config = pickle.load(fh, encoding='latin-1')
            else:
                stored_config = pickle.load(fh)

            self.e = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = p, session = self.session)
            self.e.reset_env_to_state(stored_config['start_config'], [])

    def new_demo ( self ):
        """
        Create a new environment in self.e
        """
        self.e = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = p, session = self.session)
        self.e.reset()

    def visualize ( self ):
        """
        """
        class Callback(object):
            ind = 0

            def next(self, event):
                self.ind += 1
                plt.draw()

        callback = Callback()
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)