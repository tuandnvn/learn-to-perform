'''
Created on Oct 4, 2017

@author: Tuan
'''
import glob
import os
import numpy as np
from nltk.stem.porter import PorterStemmer

from utils import DATA_DIR
import xml.etree.ElementTree as ET

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace

def read_project_data():
	ps = PorterStemmer() 
    
    project_data = {}
    
    data_length = None
    
    for file_name in glob.glob(os.path.join(DATA_DIR, '*.txt')):
    	pass