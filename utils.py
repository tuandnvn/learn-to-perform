'''
Created on Mar 4, 2017

@author: Tuan
'''
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join( ROOT_DIR, 'data') 

FIRST_EXPERIMENT_CLASSES = ['SlideAround', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideToward']