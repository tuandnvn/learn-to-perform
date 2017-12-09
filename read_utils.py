# -*- coding: utf-8 -*-
'''
Created on Oct 4, 2017

@author: Tuan
'''
import glob
import os
import numpy as np
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
# from nltk.stem.porter import PorterStemmer

from utils import DATA_DIR
import xml.etree.ElementTree as ET

# from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
# from qsrlib_io.world_trace import Object_State, World_Trace

from utils import DATA_DIR, FIRST_EXPERIMENT_CLASSES,\
    SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, GLYPH_BOX, NORMAL, \
    START, END, LABEL
from marker_util import glyphs, from_1d_array_to_face_index
import xml.etree.ElementTree as ET

'''
code : String of form  "1 ,1 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,1 ,0 ,0 ,1 ,0 ,1 ,0 ,0 ,1 ,0 ,1 ,0 ,0 ,1 ,0"
'''
def code_to_array( code ):
    values = code.split(',')

    return np.array( values, dtype = np.int32 )

def values_to_array( v ):
    values = v.split(',')

    return np.array( values, dtype = np.float32 )

'''
This code is little bit different from the code in ecat_learning

Here I read directly from files.param
===========
Params: file_path: String - absolute path to the param file

Return: a dictionary session_data that has the following form:

session_data[SESSION_OBJECTS] = Dictionary q
    q[Object_name] = 
        if object is Glyph -> Dictionary m 
            m[frameNo][face_index] = np.array of size 12, type = np.float32
        if object is not -> np.array of size 3 x n, type = np.float32 where n is the number of corners

session_data[SESSION_EVENTS] = List q
    q[index] = Dictionary m
        m[START] = start frame
        m[END] = end frame
        m[LABEL] = (object, theme)

'''
def load_one_param_file( file_path ):
    # We can load rig file later
    tree = ET.parse(file_path)
    doc = tree.getroot()

    session_name = doc.attrib['name']

    session_data = {}
    session_data[SESSION_LEN] = int(doc.attrib['length'])
    session_data[SESSION_NAME] = session_name
    session_data[SESSION_OBJECTS] = {}
    session_data[SESSION_EVENTS] = []

    all_objects = doc.find('objects').findall('object')
    for obj in all_objects:
        obj_name = obj.attrib['name']
        obj_shape = obj.attrib['shape']

        markers = []
        try:
            markers = obj.find('markers3d').findall('marker')
        except Exception as e:
            print (e)
            pass

        if obj_shape == GLYPH_BOX:
            session_data[SESSION_OBJECTS][obj_name] = {}
            for marker in markers:
                frameNo = marker.attrib['frame']

                faces = marker.findall('face')
                for face in faces:
                    bounding = face.find('bounding')
                    code = face.find('code')

                    
                    face_index = from_1d_array_to_face_index( code_to_array(code.text), glyphs )

                    if face_index != None:
                        # Replace all infinity values
                        points_str = bounding.text.replace('∞', 'inf')

                        if frameNo not in session_data[SESSION_OBJECTS][obj_name]:
                            session_data[SESSION_OBJECTS][obj_name][frameNo] = {}

                        session_data[SESSION_OBJECTS][obj_name][frameNo][face_index] = values_to_array( points_str )

        if obj_shape == NORMAL and obj_name == 'table':
            session_data[SESSION_OBJECTS][obj_name] = {}
            for marker in markers:
                frameNo = marker.attrib['frame']

                points_str = marker.text.replace('∞', 'inf')

                if frameNo not in session_data[SESSION_OBJECTS][obj_name]:
                    session_data[SESSION_OBJECTS][obj_name][frameNo] = {}

                session_data[SESSION_OBJECTS][obj_name][frameNo] = values_to_array( points_str )

    all_events = doc.find('annotations').findall('annotation')
    for event in all_events:
        e = {}
        e[START] = int(event.find('duration').attrib['startFrame'])
        e[END] = int(event.find('duration').attrib['endFrame'])

        text = event.find('text').text

        found_objects_in_text = []

        # In a more rigorous approach, we need to parse the input using a syntactic parser, here we just assume that 
        # our inputs follow some pattern
        for obj_name in session_data[SESSION_OBJECTS]:
            o_location = text.find(obj_name)

            if o_location != -1:
                found_objects_in_text.append( (o_location, obj_name) )

        sorted_objects = sorted(found_objects_in_text, key = lambda l : l[0])

        e[LABEL] = [o[1] for o in sorted_objects]

        session_data[SESSION_EVENTS].append(e)

    return session_data

def read_project_data( data_dir, classes ):
    project_data = {}
    
    data_length = None
    
    for c in classes:
        project_data[c] = []

        project_path = path.join( data_dir , c)

        session_paths = glob.glob(project_path)

        for session_path in session_paths:
            session_data = load_one_param_file( path.join( session_path, 'files.param' ))

            project_data[c].append(session_data)
    return project_data


# print(load_one_param_file(os.path.join( DATA_DIR, 'SlideAround', 'Session1', 'files.param')))