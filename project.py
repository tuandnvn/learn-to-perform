import os
import pickle

import session_utils
import generate_utils
import read_utils
import feature_utils
from utils import DATA_DIR, SESSION_LEN, SESSION_OBJ_2D
from config import Config

class ProjectData(object):
    """
    This class handle the most time consuming part of loading and cleaning the data
    """
    def __init__(self, name, session_names):
        self.session_names = session_names
        self.name = name
        self.sessions = []
        
    def load_data(self):
        for session_name in self.session_names:
            session = read_utils.load_one_param_file(os.path.join( DATA_DIR, self.name, session_name, 'files.param'))
            print ("Session " + session_name + " is loaded.")
            self.sessions.append(session)

    def preprocess(self):
        for session in self.sessions:
            session_utils.project_to2d(session, from_frame = 0, to_frame = session[SESSION_LEN])
            
        for session in self.sessions:
            session_utils.interpolate_multi_object_data(session, object_names = session[SESSION_OBJ_2D].keys())
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print('----Done saving project data---')

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f: 
            return pickle.load(f)
        print('----Done loading project data---') 
        
class Project(object):
    """
    Just a simple object to store session and other statistics
    for one action type.

    This class encapsulates the data logic
    
    project_data: Project_Data
    """
    def __init__(self, project_data, config = Config()):
        self.session_names = project_data.session_names
        self.name = project_data.name
        self.sessions = project_data.sessions
        self.config = config

    def __iter__(self):
        return iter(self.sessions)

    def __getitem__(self, key):
        return self.sessions[key]

    def __setitem__(self, key, value):
        self.sessions[key] = value

    def standardize(self, feature_extractor):
        """
        feature_extractor is a function that take in a session and add session[SESSION_FEAT] 
        """
            
        self.down_sample_quotient = session_utils.get_down_sample_quotient(self)
        print('down_sample_quotient = %d' % self.down_sample_quotient)

        self.speed = session_utils.get_action_speed(self, self.down_sample_quotient)
        print (self.speed)

        self.sessions = session_utils.down_sample(self, self.down_sample_quotient)
    
        for session in self.sessions:
            self.feature_size = feature_extractor(session,  get_location_objects = feature_utils.get_location_objects_most_active)

    def generate_data(self):
        # First step is to generate data with hop_step interpolation
        # rearranged_data = (samples, num_steps, data_point_size)
        # rearranged_lbls = (samples, num_steps)
        rearranged_data, rearranged_lbls = generate_utils.turn_to_intermediate_data(self, 
            self.feature_size, self.config.num_steps, self.config.hop_step)

        # Generate training and testing data 
        self.training_data, self.training_lbl, self.testing_data, self.testing_lbl =\
         generate_utils.generate_data(rearranged_data, rearranged_lbls, self.config)

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
    """
    This code will load the data from params files and project to 2D, which is the most time consuming part in loading
    """
    # p_data = ProjectData("SlideAround", ["Session1", "Session2"])
    # print ('Load project ' + p_data.name)
    # p_data.load_data()
    # p_data.preprocess()
    # p_data.save("slidearound_data_raw.proj")
     
    # p_data = ProjectData.load("slidearound_data_raw.proj")
    # p = Project(p_data)
    # p.standardize(feature_utils.marker_feature_extractor) 
    # p.generate_data()
    # p.save("slidearound_raw.proj")

    p = Project.load("slidearound_raw.proj")
    print (p.training_data[0][0])
    print (p.training_lbl[0][0])
    
    print (p.training_data[0][1])
    print (p.training_lbl[0][1])

    print (p.training_data.shape)
    print (p.training_lbl.shape)
    
    print (p.testing_data.shape)
    print (p.testing_lbl.shape)

    # p_data = ProjectData.load("slidearound_data_raw.proj")

    # for key in p_data.sessions[0][SESSION_OBJ_2D]:
    #     print ('-------------------------------------------------------- ' +  key )
    #     for frame in range(p_data.sessions[0][SESSION_LEN]):
    #         print (p_data.sessions[0][SESSION_OBJ_2D][key][frame])

