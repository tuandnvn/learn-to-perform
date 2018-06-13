import os
from collections import defaultdict
import argparse

import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from action_learner_interactive import InteractiveLearner
from feedback_improver import create_batch_size

class HistoryStorage ( object ) :
    """
    At each step, when a user has generated a new demonstration of the action
    we can correct the progress on each of the search position. 

    We want to clamp down on ones that has progress values higher than the selected one.
    """
    def __init__ (self) :
        # Indexing with self.step
        self.samples = defaultdict(list)

        # Indexing with self.step 
        self.corrected_samples = {}

    def add_sample (self, step, X, y):
        self.samples[step].append((X, y))

    def add_corrected_sample ( self, step, X, y):
        """
        Notice that y is the progress as predicted by the current progress function

        We want to clamp down on all other samples that has higher value of y
        to value y * ( 1 - alpha ) .
        """
        self.corrected_samples[step] = (X, y)


class InteractiveLearnerHot ( InteractiveLearner ):
    """
    An interactive learner that support fixing an action in a demonstration by choosing another action in the searching space.
    
    This interactive learner should only use Greedy search on Continuous space, because the new actions are taken on Continuous space.
    """
    def __init__(self, c = None, action_type = "SlideAround", online = True, project_path = None, 
                        progress_model_path = None, new_progress_model_path = None):
        super().__init__(c = c, action_type = action_type, discrete = False, online = online, 
            project_path = project_path, progress_model_path = progress_model_path)

        self.new_progress_model_path = new_progress_model_path
        """
        We store history of searching 
        so that we can update the reward at a later phase

        This history stores features of all search actions
        and corresponding progress. 
        """
        self.history_storage =  HistoryStorage()

    def new_demo ( self ):
        """
        """
        super().new_demo()
        self.history_storage =  HistoryStorage()

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

        self.collected_sample = []

        for action_index, action in enumerate(actions):
            _, reward, done, info = exploration.step((self.select_object, action, action_means, action_stds))

            if info ['action_accepted']:
                X = exploration.get_feature_only()
                y = self.progress[-1] + reward

                # We collect samples for later adding into self.history_storage
                self.collected_sample.append((X, y))

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

            self.prev_action = best_action
            return best_action
        else:
            print ("==========No action========== ")
            self.prev_action = None
            return None

    def get_onclick(self):
        def onclick(event):
            if event.inaxes == self.callback.ax:
                p = event.xdata, event.ydata
                print (p)

                exploration = self.searcher.env

                if self.prev_action is not None:
                    # Back to previous step means not increasing the index
                    exploration.back()
                    del self.progress[-1]
                else:
                    self.callback.index += 1

                for X, y in self.collected_sample:
                    self.history_storage.add_sample(self.callback.index, X, y)

                action = (p[0], p[1], 0)
                new_action = (self.select_object, action)
                _, reward, done, _ = exploration.step(new_action)

                X = exploration.get_feature_only()
                y = self.progress[-1] + reward
                self.progress.append(y)
                self.history_storage.add_corrected_sample(self.callback.index, X, y)

                te = self.callback.ax.text(action[0]-0.02, action[1]-0.02, '%d' % self.callback.index, fontsize=18)
                self.callback.tes.append(te)
                self.callback.set_title(additional = ', at corrected location.')

                self.callback._draw_collection_from_env()

        return onclick

    def save ( self, event ):
        """
        Save the progress model to a file
        """
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + self.action_type))
        saver.save(self.sess, self.new_progress_model_path)

        print ('Model is saved at %s' % self.new_progress_model_path)

    def update_with_data ( self, X, y, episode = 10 ):
        print (X.shape)
        print (y.shape)
        c = self.pe.config
        for i in range(episode):
            print ('-------------------------------')
            lr_decay = c.lr_decay ** i
            new_learning_rate = c.learning_rate * 0.05 * lr_decay
            print ('Rate = %.5f' % (new_learning_rate))
            self.pe.assign_lr(new_learning_rate, sess = self.sess)
            
            # Update with negative samples
            self.pe.update(X, y, sess = self.sess)

    def update ( self, event ):
        print ('================')
        print ('Samples')
        for step in self.history_storage.samples:
            vals = [('%.3f' % y) for X, y in self.history_storage.samples[step]]
            print ('step = %d, vals = %s' % (step, ','.join(vals)))
        print ('Corrected samples')
        for step in self.history_storage.corrected_samples:
            X, y = self.history_storage.corrected_samples[step]
            print ('step = %d, correct val = %.3f' % (step, y))
        print ('================')
        print ('Update')
        alpha = self.config.interactive_alpha
        """
        Collect samples from self.history_storage
        and update right away
        """
        # First type of update
        # Update all other positions so that progress values = correct position * ( 1 - alpha )
        X_samples_1 = []
        y_samples_original = []
        y_samples_1 = []

        for step in self.history_storage.corrected_samples:
            X_corrected, y_corrected = self.history_storage.corrected_samples[step]

            for X, y in self.history_storage.samples[step]:
                if y > y_corrected * ( 1 - alpha ):
                    X_samples_1.append(X)
                    y_samples_original.append(y)
                    y_samples_1.append( y_corrected * ( 1 - alpha ) )

        print (y_samples_original)

        batch_size = self.config.batch_size

        if len(X_samples_1) != 0 and len(y_samples_1) != 0:
            X_1_s = create_batch_size ( X_samples_1, batch_size )
            y_1_s = create_batch_size ( y_samples_1, batch_size )
            for X, y in zip(X_1_s, y_1_s):
                self.update_with_data ( X, y )


        # Update all correct positions so that progress values = step / total_step 
        # if  step / total_step > current_progress
        X_samples_2 = []
        y_samples_2 = []
        for step in self.history_storage.corrected_samples:
            X_corrected, y_corrected = self.history_storage.corrected_samples[step]

            if y_corrected < step / self.callback.index:
                X_samples_2.append(X_corrected)
                y_samples_2.append(step / self.callback.index)

        print ([self.history_storage.corrected_samples[step][1] for step in self.history_storage.corrected_samples])

        if len(X_samples_2) != 0 and len(y_samples_2) != 0:
            X_2_s = create_batch_size ( X_samples_2, batch_size )
            y_2_s = create_batch_size ( y_samples_2, batch_size )
            for X, y in zip(X_2_s, y_2_s):
                self.update_with_data ( X, y )

        # There might me one more update step to order progress values, but let's see
        print ('Recalculate')
        try:
            results = []
            for X in X_1_s:
                updated = self.pe.predict(X, sess = self.sess)
                for v in updated:
                    results.append(v)

            print (results[:len(X_samples_1)])
        except NameError:
            pass

        try:
            results = []
            for X in X_2_s:
                updated = self.pe.predict(X, sess = self.sess)
                for v in updated:
                    results.append(v)

            print (results[:len(X_samples_2)])
        except NameError:
            pass


    def visualize (self):
        # super().visualize()
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

        axupdate = plt.axes([0.48, 0.05, 0.1, 0.075])
        bupdate = Button(axupdate, 'Update')
        bupdate.on_clicked(self.update)

        axsave = plt.axes([0.37, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(self.save)

        fig.canvas.mpl_connect('button_press_event', self.get_onclick() )

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test greedy interactive API with immediate update.')

    parser.add_argument('-a', '--action', action='store', metavar = ('ACTION'),
                                help = "Action type. Choose from 'SlideToward', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideAround'" )
    parser.add_argument('-p', '--progress', action='store', metavar = ('PROGRESS'),
                                help = "Path of progress file. Default is 'learned_models/progress_' + action + '.mod.updated'" )
    parser.add_argument('-s', '--save', action='store', metavar = ('SAVE'),
                                help = "Where to save updated progress file. Default is 'learned_models/progress_' + action + '.mod.updated.updated'" )

    args = parser.parse_args()

    progress_path = args.progress
    project_name = args.action
    progress_path_save = args.save

    if project_name is None:
        project_name = 'SlideAround'

    if progress_path is None:
        progress_path = os.path.join('learned_models', 'progress_' + project_name + '.mod.updated')

    if progress_path_save is None:
        progress_path_save = os.path.join('learned_models', 'progress_' + project_name + '.mod.updated.updated')

    il = InteractiveLearnerHot(action_type = project_name, online = True, progress_model_path = progress_path,
        new_progress_model_path = progress_path_save)

    # il.load_demo(os.path.join('experiments', 'human_evaluation_2d', 'SlideAround', '0.dat'), online = True)

    il.visualize()
           
        