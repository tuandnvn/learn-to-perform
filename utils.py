'''
Created on Oct 1, 2017

@author: Tuan
'''
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join( ROOT_DIR, 'data') 
AUXILLIARY_DIR = os.path.join( ROOT_DIR, 'auxilliary')

GLYPH_DIR =  os.path.join( AUXILLIARY_DIR, 'glyph')

FIRST_EXPERIMENT_CLASSES = ['SlideAround', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideToward']

SESSION_NAME = "session_name"
SESSION_OBJECTS = "session_objects"
SESSION_EVENTS = "session_events"

START = "start"
END = "end"
LABEL = "label"

GLYPH_BOX = "Annotator.GlyphBoxObject"
NORMAL = "Annotator.PolygonObject"