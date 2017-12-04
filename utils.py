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

START = "start"
END = "end"
LABEL = "label"

GLYPH_BOX = "Annotator.GlyphBoxObject"
NORMAL = "Annotator.PolygonObject"

# 18 cm?
BLOCK_SIZE = 0.18
# just 10 degree
ROTATION_QUANTIZATION = np.pi / 18