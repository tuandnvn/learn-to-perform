import os
import sys
import collections
import tensorflow as tf
from importlib import reload
import pickle
import numpy as np
import argparse

sys.path.append(os.path.join("strands_qsr_lib", "qsr_lib", "src3"))

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

from test_all_searcher import get_model

class InteractiveLearner ( object ):
    def __init__(self, c = None, action_type = "SlideAround", discrete = True, 
                    online = True, project_path = None, progress_model_path = None):
        """
        c: config.Config
        action_type: Default is "SlideAround". Note that all models tensors have scope='model/' + action_type
        project_path: path to project.Project pickle file
        progress_model_path: path to Tensorflow stored file 
        discrete: = True use dals, otherwise use als
        online: = True means every time you click Next, it would look for next best action
                  False means it searches all the path from the beginning, only shows the step
        """
        tf.reset_default_graph()

        self.online = online

        if c is None:
            c = config.Qual_Config()

        self.config = c

        global_step = tf.Variable(0, name="global_step", trainable=False)

        self.sess = sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        if action_type is None:
            action_type = "SlideAround"

        self.action_type = action_type

        if project_path is None:
            project_path = os.path.join('learned_models', action_type.lower() + "_project.proj")
        
        p, pe = get_model ( action_type, sess, project_path = project_path, 
                    progress_path = progress_model_path, c = self.config)
        self.project = p
        self.pe = pe

        if discrete:
            self.searcher = dals.Discrete_ActionLearner_Search(c, p, pe, self.sess )
        else:
            self.searcher = als.ActionLearner_Search(c, p, pe, self.sess )

        self.select_object = 0
        self.no_of_search = 24
        self.progress = [0]

    def load_demo( self, demo_file, online = False ):
        """
        Set self.e to the initial state in demo_file if demo_file store a demonstration

        Parameters:
        -----
        demo_file: a .dat file that has been saved as picked file
        online: = False: just watch the old demonstration
                = True: Re planning

        Returns:
        --------
        """
        self.online = online

        with open(demo_file, 'rb') as fh:
            # need this encoding 
            if sys.version_info >= (3,0):
                stored_config = pickle.load(fh, encoding='latin-1')
            else:
                stored_config = pickle.load(fh)

            # Reset environment in self.searcher
            self.searcher.env = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = self.pe, session = self.sess)
            self.searcher.env.reset_env_to_state(stored_config['start_config'], [])

            self.action_storage =  [(object_index, next_transform, action_means, action_stds) 
                                for object_index, _, next_transform, _, _, success, action_means, action_stds in stored_config['action_storage'] if success]

            self.next_action = 0

            self.progress = [0]

    def new_demo ( self ):
        """
        Create a new environment in self.e
        """
        self.searcher.env = block_movement_env.BlockMovementEnv(self.config, self.project.speed, self.project.name, 
                    progress_estimator = self.pe, session = self.sess)
        self.searcher.env.reset()

        self.progress = [0]

    def search_next( self ):
        """
        Given the current state, search for the next action

        Online mode
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
        

        if best_reward > 0:
            print ("==========best action========== ", best_action)
            exploration.step((self.select_object, best_action, action_means, action_stds))

            self.progress.append(self.progress[-1] + best_reward)
            print ("==========progress========== ", self.progress[-1])

            return best_action
        else:
            print ("==========No action========== ")
            return None

    def move_next( self ):
        """
        Offline mode
        """
        if self.next_action < len(self.action_storage):
            object_index, action, action_means, action_stds = self.action_storage[self.next_action]
            action = action.get_feat()
            _, reward, _, _ = self.searcher.env.step((object_index, action, action_means, action_stds))

            self.progress.append(self.progress[-1] + reward)
            self.next_action += 1
        else:
            action = None

        return action

    class Callback(object):
        def __init__(self, outer, fig, ax ):
            self.outer = outer
            self.fig = fig
            self.ax = ax
            self.index = 0
            self.set_title()

            self.env = self.outer.searcher.env

            fig.set_size_inches(self.env.graph_size, self.env.graph_size)
            x_range = np.arange(self.env.playground_x[0], 
                                    self.env.playground_x[0] + self.env.playground_dim[0], 0.1)
            y_range = np.arange(self.env.playground_x[1], 
                                    self.env.playground_x[1] + self.env.playground_dim[1], 0.1)
            ax.set_xticks(x_range)
            ax.set_yticks(y_range)
            ax.set_xlim(self.env.playground_x[0], 
                                    self.env.playground_x[0] + self.env.playground_dim[0])
            ax.set_ylim(self.env.playground_x[1], 
                                    self.env.playground_x[1] + self.env.playground_dim[1])

            # Storing block objects
            self.lcs = []
            # Storing text objects (action values written into block)
            self.tes = []

            self._draw_collection_from_env()


        def set_title(self, additional = ''):
            if self.index == 0:
                self.ax.set_title('Start', fontsize=18)
            else:
                self.ax.set_title('At step %d, progress = %.3f %s' % (self.index, self.outer.progress[-1], additional), fontsize=18 )

        def next(self, event):
            if self.outer.online:
                action = self.outer.search_next()
            else:
                action = self.outer.move_next()

            if action is not None:
                self.index += 1
                te = self.ax.text(action[0]-0.02, action[1]-0.02, '%d' % self.index, fontsize=18)
                self.tes.append(te)
            self.set_title()

            self._draw_collection_from_env()

        def _draw_collection_from_env(self):
            lc = self.outer.searcher.env.get_collection()
            self.ax.add_collection(lc)
            self.lcs.append(lc)
            self.fig.canvas.draw()

        def prev(self, event):
            if self.index > 0:
                self.outer.searcher.env.back()
                self.index -= 1
                self.outer.next_action -= 1
                self.set_title()

                self.lcs[-1].remove()
                del self.lcs[-1]
                del self.outer.progress[-1]
                self.tes[-1].remove()
                del self.tes[-1]

                self.fig.canvas.draw()

        def reset(self, event):
            self.outer.new_demo()
            self.ax.clear()
            self.__init__(self.outer, self.fig, self.ax)


    def visualize ( self ):
        """
        """
        fig = plt.figure()  # a new figure window 
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        plt.subplots_adjust(bottom=0.2)

        self.callback = callback = InteractiveLearner.Callback(self, fig, ax )

        axreset = plt.axes([0.59, 0.05, 0.1, 0.075])
        breset = Button(axreset, 'Reset')
        breset.on_clicked(callback.reset)

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test greedy interactive API.')

    parser.add_argument('-a', '--action', action='store', metavar = ('ACTION'),
                                help = "Action type. Choose from 'SlideToward', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideAround'" )
    parser.add_argument('-p', '--progress', action='store', metavar = ('PROGRESS'),
                                help = "Path of progress file. Default is 'learned_models/progress_' + action + '.mod.updated'" )
    parser.add_argument('-d', '--discrete', action='store', metavar = ('DISCRETE'),
                                help = "Whether it is discrete greedy search. Default is False" )

    args = parser.parse_args()

    progress_path = args.progress

    project_name = args.action

    if project_name is None:
        project_name = 'SlideAround'

    if progress_path is None:
        progress_path = os.path.join('learned_models', 'progress_' + project_name + '.mod.updated')

    if args.discrete == 'True':
        discrete = True
    else:
        discrete = False

    il = InteractiveLearner(discrete = discrete, action_type = project_name, online = True, progress_model_path = progress_path)
    # il.load_demo(os.path.join('experiments', 'human_evaluation_2d', 'SlideAroundDiscrete', '9.dat'))

    # il = InteractiveLearner(discrete = False, online = True, progress_model_path = os.path.join('learned_models', 'progress_SlideAround.mod'))
    # il.load_demo(os.path.join('experiments', 'human_evaluation_2d', 'SlideAround', '0.dat'))
    il.visualize()