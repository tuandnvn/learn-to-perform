import os
import sys
import collections
import tensorflow as tf
from importlib import reload
import pickle

sys.path.append("strands_qsr_lib\qsr_lib\src3")

## PLOTTING 
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
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
    def __init__(self, c = None, action_type = "SlideAround", discrete = True):
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
                                                        name = action_type, 
                                                        config = c)  

        # Print out all variables that would be restored
        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
            print (variable.name)
        
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + action_type))

        saver.restore(sess, os.path.join('learned_models', 'progress_' + action_type + '.mod.1'))

        if discrete:
            self.searcher = dals.Discrete_ActionLearner_Search(c, p, pe, self.sess )
        else:
            self.searcher = als.ActionLearner_Search(c, p, pe, self.sess )

        self.select_object = 0
        self.no_of_search = 24
        self.progress = 0

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

            # Reset environment in self.searcher
            self.searcher.env = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = p, session = self.session)
            self.searcher.env.reset_env_to_state(stored_config['start_config'], [])

            self.progress = 0

    def new_demo ( self ):
        """
        Create a new environment in self.e
        """
        self.searcher.env = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = p, session = self.session)
        self.searcher.env.reset()

        self.progress = 0

    def search_next( self ):
        """
        Given the current state, search for the next action
        """
        exploration = self.searcher.env
        tempo_rewards = []
        action_means, action_stds, actions = self.searcher._get_actions(self.select_object, exploration, self.no_of_search, verbose = False)

        best_reward = -1
        best_action = None

        for action_index, action in enumerate(actions):

            _, reward, done, _ = exploration.step((self.select_object, action, action_means, action_stds))
            print (action, reward)
            exploration.back()

            if reward > best_reward:
                best_reward = reward
                best_action = action

            if done:
                print ("=== found_completed_act ===")
                found_completed_act = True

        print ("==========best action========== ", best_action)
        exploration.step((self.select_object, best_action, action_means, action_stds))

        self.progress += best_reward
        print ("==========progress========== ", self.progress)

    def visualize ( self ):
        """
        """
        class Callback(object):
            def __init__(self, outer, fig, ax ):
                self.outer = outer
                self.fig = fig
                self.ax = ax

            def next(self, event):
                self.outer.search_next()
                self.outer.searcher.env._render(fig = self.fig, ax = self.ax, show = False)

        fig = plt.figure()  # a new figure window
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.subplots_adjust(bottom=0.2)

        callback = Callback( self, fig, ax )
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)

        self.searcher.env._render(fig = fig, ax = ax, show = True)

if __name__ == '__main__':
    il = InteractiveLearner()
    il.visualize()