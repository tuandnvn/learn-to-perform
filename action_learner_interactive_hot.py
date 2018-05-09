import os
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from action_learner_interactive import InteractiveLearner

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
    def __init__(self, c = None, action_type = "SlideAround", online = True, project_path = None, progress_model_path = None):
        super().__init__(c = c, action_type = action_type, discrete = False, online = online, 
            project_path = project_path, progress_model_path = progress_model_path)
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
        print ('Samples')
        for step in self.history_storage.samples:
            vals = [('%.3f' % y) for X, y in self.history_storage.samples[step]]
            print ('step = %d, vals = %s' % (step, ','.join(vals)))
        print ('Corrected samples')
        for step in self.history_storage.corrected_samples:
            X, y = self.history_storage.corrected_samples[step]
            print ('step = %d, correct val = %.3f' % (step, y))

    def update ( self ):
        """
        Collect samples from self.history_storage
        and update right away
        """
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

        axsave = plt.axes([0.48, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(self.save)

        fig.canvas.mpl_connect('button_press_event', self.get_onclick() )

        plt.show()

if __name__ == '__main__':
    # I have used progress_SlideAround.mod for continuous search
    # and progress_SlideAround.mod.1 for discrete search
    # Couldn't remember for sure what is the difference

    il = InteractiveLearnerHot(online = True, progress_model_path = os.path.join('learned_models', 'progress_SlideAround.mod.updated'))
    # il.load_demo(os.path.join('experiments', 'human_evaluation_2d', 'SlideAroundDiscrete', '9.dat'))

    # il = InteractiveLearner(discrete = False, online = True, progress_model_path = os.path.join('learned_models', 'progress_SlideAround.mod'))
    # il.load_demo(os.path.join('experiments', 'human_evaluation_2d', 'SlideAround', '0.dat'))
    il.visualize()
           
        