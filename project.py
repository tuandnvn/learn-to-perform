import os
import pickle
import sys

import session_utils
import generate_utils
import read_utils
import feature_utils
from utils import DATA_DIR, SESSION_LEN, SESSION_OBJ_2D
from config import Config

import numpy as np

class Data(object):
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print('----Done saving project data---')

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f: 
            if sys.version_info >= (3,0):
                return pickle.load(f, encoding='latin-1')
            else:
                return pickle.load(f)
        print('----Done loading project data---') 

class ProjectData(Data):
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
    
        
class Project(Data):
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
            
            # Rescale feature
            # Here we can do 
            # Selecting only features we like, or make some modification on the feature
            feature_utils.standardize_simple(session)

    def generate_data(self, linear_progress_lbl_func = 
           generate_utils.linear_progress_lbl_generator):
        # First step is to generate data with hop_step interpolation
        # rearranged_data = (samples, num_steps, data_point_size)
        # rearranged_lbls = (samples, num_steps)
        # self.rearranged_data, self.rearranged_lbls = generate_utils.turn_to_intermediate_data(self, 
        #     self.config.n_input, self.config.num_steps, self.config.hop_step, 
        #     linear_progress_lbl_func = linear_progress_lbl_func)

        self.rearranged_data, self.rearranged_lbls = generate_utils.turn_to_intermediate_data_multiscale(self, 
            self.config.n_input, self.config.num_steps, self.config.hop_step)

        # Generate training and testing data 
        self.training_data, self.training_lbl, self.validation_data, self.validation_lbl,\
         self.testing_data, self.testing_lbl =\
         generate_utils.generate_data(self.rearranged_data, self.rearranged_lbls, self.config)

class Multi_Project(Data):
    """
    An object to store multiple projects

    Support generating data from multiple projects, with data from other projects are negative samples of one project
    """
    def __init__(self, multi_project, config = Config()):
        self.multi_project = multi_project
        self.config = config

    def __iter__(self):
        return iter(self.multi_project)

    def __getitem__(self, key):
        return self.multi_project[key]

    def generate_data(self, linear_progress_lbl_func = 
           generate_utils.linear_progress_lbl_generator):

        # Mapping from project name to set of training, validation and testing data and labels
        self.data = {}

        self.tempo_data = {}

        total_sample = 0
        num_steps = self.config.num_steps
        n_input = self.config.n_input

        for project in self.multi_project:
            print ('---------- ' + project.name + ' ------------')
            rearranged_data, rearranged_lbls = generate_utils.turn_to_intermediate_data_multiscale(project, 
                self.config.n_input, self.config.num_steps, self.config.hop_step)

            # rearranged_data: (# samples, num_steps, n_input)
            # rearranged_lbls: (# samples, num_steps)

            self.tempo_data[project.name] = (rearranged_data, rearranged_lbls)

            # total_sample += # samples
            total_sample += self.tempo_data[project.name][1].shape[0]

        for project in self.multi_project:
            # Including negative samples
            total_project_data = np.zeros([total_sample, num_steps, n_input], dtype=np.float32)
            total_project_lbls = np.zeros([total_sample, num_steps], dtype=np.float32)
            # For weighting of samples
            total_project_info = np.zeros(total_sample, dtype=np.float32)

            counter = 0
            for other_project in self.multi_project:
                other_project_samples = self.tempo_data[other_project.name][1].shape[0]
                total_project_data[counter: counter + other_project_samples] = self.tempo_data[other_project.name][0]

                # We only set values of positive samples to 1
                # Otherwise just leave them as 0
                if other_project.name == project.name:
                    total_project_lbls[counter: counter + other_project_samples] = self.tempo_data[other_project.name][1]
                    total_project_info[counter: counter + other_project_samples] = 1
                else:
                    # Let give all the negative sampels the total weight the same as the positive samples
                    total_project_info[counter: counter + other_project_samples] = 1.0 / (len(self.multi_project) - 1)

                counter += other_project_samples

            # Generate training and testing data 
            self.data[project.name] = generate_utils.generate_data_info(total_project_data, total_project_lbls, total_project_info, self.config)
    
if __name__ == "__main__":
    """
    This code will load the data from params files and project to 2D, which is the most time consuming part in loading
    """
    def create_projects():
        from project import Project, ProjectData
        for project_name in ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]:
            p_data = ProjectData(project_name, ["Session1", "Session2"])
            print ('Load project ' + p_data.name)
            p_data.load_data()
            p_data.preprocess()
            p_data.save(project_name.lower() + "_data.proj")
            p = Project(p_data)
            p.standardize(feature_utils.qsr_feature_extractor)
            p.generate_data()
            p.save(project_name.lower() + "_project.proj")

    create_projects()


    # p_data = ProjectData("SlideAround", ["Session1", "Session2"])
    # print ('Load project ' + p_data.name)
    # p_data.load_data()
    # p_data.preprocess()
    # p_data.save("slidearound_p2.proj")
     
    # p_data = ProjectData.load("slidearound_p2.proj")
    # p = Project(p_data)
    # p.standardize(feature_utils.marker_feature_extractor) 
    # p.generate_data()
    # p.save("slidearound_raw.proj")

    # p = Project.load("slidearound_p2.proj")


    def load_multi_project(output_name):
        ps = []
        for project_name in ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]:
            p_data = ProjectData.load(project_name.lower() + "_p2.proj")

            p = Project(p_data)
            p.standardize(feature_utils.marker_feature_extractor)

            ps.append(p)

        multi_p = Multi_Project(ps)
        multi_p.generate_data()
        multi_p.save(output_name)

    # load_multi_project("all_actions.proj")


    # p.generate_data(linear_progress_lbl_func = 
    #        generate_utils.linear_progress_lbl_generator_retreat)

    # p.save("slidearound_hopstep_1_multiscale_quant.proj")

    # print (p.training_data[0][0])
    # print (p.training_lbl[0][0])
    
    # print (p.training_data[0][1])
    # print (p.training_lbl[0][1])

    # print (p.training_data.shape)
    # print (p.training_lbl.shape)
    
    # print (p.testing_data.shape)
    # print (p.testing_lbl.shape)

    # p_data = ProjectData.load("slidearound_data_raw.proj")

    # for key in p_data.sessions[0][SESSION_OBJ_2D]:
    #     print ('-------------------------------------------------------- ' +  key )
    #     for frame in range(p_data.sessions[0][SESSION_LEN]):
    #         print (p_data.sessions[0][SESSION_OBJ_2D][key][frame])

