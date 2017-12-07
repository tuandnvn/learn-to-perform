import os
import pickle

import session_utils
import generate_utils
import read_utils
from utils import DATA_DIR, SESSION_LEN, SESSION_OBJ_2D
from progress_learner import EventProgressEstimator
from config import Config

class Project(object):
    """
    Just a simple object to store session and other statistics
    for one action type.

    This class encapsulates the data logic
    """
    def __init__(self, name, session_names, config = Config()):
        self.session_names = session_names
        self.name = name
        self.sessions = []
        self.config = config

    def __iter__(self):
        return iter(self.sessions)

    def __getitem__(self, key):
        return self.sessions[key]

    def __setitem__(self, key, value):
        self.sessions[key] = value

    def load_data(self):
        for session_name in self.session_names:
            session = read_utils.load_one_param_file(os.path.join( DATA_DIR, self.name, session_name, 'files.param'))
            print ("Session " + session_name + " is loaded.")
            self.sessions.append(session)

    def preprocess(self):
        for session in self.sessions:
            session_utils.project_to2d(session, from_frame = 0, to_frame = session[SESSION_LEN])

    def standardize(self):
        for session in self.sessions:
            session_utils.interpolate_multi_object_data(session, object_names = session[SESSION_OBJ_2D].keys())

        self.down_sample_quotient = session_utils.get_down_sample_quotient(self)
        print('down_sample_quotient = %d' % self.down_sample_quotient)

        self.speed = session_utils.get_action_speed(self, self.down_sample_quotient)
        print (self.speed)

        self.sessions = session_utils.down_sample(self, self.down_sample_quotient)

    def generate_data(self):
        # First step is to generate data with hop_step interpolation
        # rearranged_data = (samples, num_steps, data_point_size)
        # rearranged_lbls = (samples, num_steps)
        rearranged_data, rearranged_lbls = generate_utils.turn_to_intermediate_data(project, 
            self.config.n_input, self.config.num_steps, self.config.hop_step)

        # Generate training and testing data 
        self.training_data, self.training_lbl, self.testing_data, self.testing_lbl =\
         generate_utils.generate_data(rearranged_data, rearranged_lbls, config)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print('----Done saving project---')

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f: 
            return pickle.load(f)
        print('----Done loading project---') 

    
if __name__ == "__main__":
    p = Project("SlideAround", ["Session1", "Session2"])
    print ('Load project ' + p.name)
    p.load_data()
    p.preprocess()
    p.standardize()
    p.save("slidearound.proj")
    
    