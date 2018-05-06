'''
Created on Oct 1, 2017

@author: Tuan
'''
import numpy as np
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join( ROOT_DIR, 'data') 
AUXILLIARY_DIR = os.path.join( ROOT_DIR, 'auxilliary')

GLYPH_DIR =  os.path.join( AUXILLIARY_DIR, 'glyph')

FIRST_EXPERIMENT_CLASSES = ['SlideAround', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideToward']

SESSION_NAME = "session_name"
SESSION_OBJECTS = "session_objects"
SESSION_EVENTS = "session_events"
SESSION_LEN = "session_length"
SESSION_FEAT = "session_feature"
SESSION_OBJ_2D = "session_obj_2d"
SESSION_FEAT_STAND = "session_feature_standardized"

START = "start"
END = "end"
LABEL = "label"

GLYPH_BOX = "Annotator.GlyphBoxObject"
NORMAL = "Annotator.PolygonObject"

# 18 cm?
BLOCK_SIZE = 0.15
# just 10 degree
ROTATION_QUANTIZATION = np.pi / 8

TRAINING = 'TRAINING'
VALIDATING = 'VALIDATING'
TESTING = 'TESTING'

MODEL_PATH = 'learned_models'

def get_default_models( action_types ):
	import tensorflow as tf

	for project_name in action_types:
        configs[project_name] = config.Config()
        if project_name == 'SlideNext':
            configs[project_name].n_input = 8
            
        print ('========================================================')
        print ('Load for action type = ' + project_name)
        p_name = project_name.lower() + "_project.proj"

        projects[project_name] = project.Project.load(os.path.join('learned_models', p_name))
        
        with tf.variable_scope("model") as scope:
            print('-------- Load progress model ---------')
            progress_estimators[project_name] = progress_learner.EventProgressEstimator(is_training=True, 
                                                                                        is_dropout = False, 
                                                                                        name = projects[project_name].name, 
                                                                                        config = configs[project_name])  
            
    for project_name in action_types:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/' + project_name))

        saver.restore(sess, os.path.join('learned_models', 'progress_' + project_name + '.mod'))

    return configs, projects, progress_estimators
