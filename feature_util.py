'''
Created on Oct 15, 2017

@author: Tuan

===========================
Projecting a segment of data into two dimensional plane 
delineated by the table surface
'''
import numpy as np
from utils import SESSION_OBJECTS
from feature.project_table import project_markers, estimate_cube_2d

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace

# Count the number of finite element in an array
def count_finite( numpy_array ):
    return np.sum( np.isfinite( numpy_array )) 

'''
Area of polygons
===========
Params: numpy_array of size ( 3 x n )

Return: object_data: Dictionary
'''
def area_dimension( numpy_array ):
    s = np.reshape(numpy_array, (len(numpy_array)//3,3))
    x = s[:,0]
    y = s[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

'''
===========
Params: session_data: check the return value of read_utils.load_one_param_file

Return: object_data: Dictionary
    object_data[object_name] = interpolated and 2-d data projected onto the table
    object_data[object_name][frameNo] = Cube-2D (a wrapper of a transform)
'''
def project_to2d ( session_data ):
    object_data = {}

    for object_name in session_data[SESSION_OBJECTS]:
        if object_name == 'table':
            polygon = []
            for frameNo in session_data[SESSION_OBJECTS][object_name]:
                frame_polygon = session_data[SESSION_OBJECTS][object_name][int(frameNo)]

            polygon.append(frame_polygon)

            polygon = np.concatenate(polygon)
            polygon = np.reshape(polygon, (len(polygon)//3, 3) )
            
            table_markers = polygon

            # Just pick the first two points for coordination
            first_point = table_markers[0]
            second_point = table_markers[1]
    
    for object_name in session_data[SESSION_OBJECTS]:
        if object_name != 'table':
            object_data[object_name] = {}
            for frameNo in session_data[SESSION_OBJECTS][object_name]:
                frame_data = session_data[SESSION_OBJECTS][object_name][int(frameNo)]

                # Sort firstly by number of non-finite corners
                # Sort secondly by size of marker (larger marker means better resolution)
                # Size of marker should be only based on first two dimensions
                # The third dimension might be very noisy
                q = [((count_finite(frame_data[face_index]), area_dimension(frame_data[face_index]) ), face_index) 
                    for face_index in frame_data]
                q = sorted(q, key = lambda t: t[0], reverse = True)

                # Pick out the face_index with the most number of non-infinite values
                best_face_index = q[0][1]

                rectangle_projected = project_markers ( frame_data[best_face_index], table_markers )

                object_data[object_name][int(frameNo)] = estimate_cube_2d ( rectangle_projected, first_point, second_point )

    return object_data

'''
Interpolate data for missing frames

===========
Params: 
object_data

Return:
interpolated_object_data: chain of features, one feature vector for each frame (interpolated frames)
'''
def interpolate_object_data( object_data ):


def turn_response_to_features(keys, qsrlib_response_message):
    feature_chain = []
    for t in qsrlib_response_message.qsrs.get_sorted_timestamps():
        features = []
        # print (qsrlib_response_message.qsrs.trace[t].qsrs.keys())
        for k in keys:
            if k in qsrlib_response_message.qsrs.trace[t].qsrs:
                v = qsrlib_response_message.qsrs.trace[t].qsrs[k]

                if 'cardir' in v.qsr:
                    f = v.qsr['cardir']
                    features.append(cardir_index(f))
                if 'argd' in v.qsr:
                    f = int( v.qsr['argd'] )
                    features.append(f)
                if 'mos' in v.qsr:
                    f = v.qsr['mos'] 
                    features.append(mos_index(f))
        # Just to separate qtccs at the end of feature vectors
        
        for k in keys:
            if k in qsrlib_response_message.qsrs.trace[t].qsrs:
                v = qsrlib_response_message.qsrs.trace[t].qsrs[k]
                if 'qtccs' in v.qsr:
                    fs = v.qsr['qtccs']
                    features += [qtcc_index(f) for f  in fs.split(',')]
        
        # print features
        feature_chain.append(features)
    
    if len(feature_chain) == 0:
        return feature_chain

    # The first frame doesn't has mos and qtcc relations
    feature_chain[0] += [0, 0, 0, 0, 0, 0, 0]
    
    diff_feature_chain = [ [feature_chain[t + 1][i] - feature_chain[t][i] 
                            for i in xrange(len(feature_chain[0]) - 7) ] + \
                          [feature_chain[t][i] for i in xrange(len(feature_chain[0]) - 7, len(feature_chain[0]))]
                    for t in xrange(len(feature_chain) - 1)]
    
    diff_feature_chain = [[0 for i in xrange(len(feature_chain[0]))]] +  diff_feature_chain

    # Concatenate features
    # feature_chain = [feature_chain[i] + diff_feature_chain[i] for i in xrange(len(feature_chain))]

    return diff_feature_chain

'''
Get the features from 
- one object as the one mainly under movement (object slot)
- one object that is relatively static (theme slot)

===========
Params: 
qsrlib: check the return value of read_utils.load_one_param_file
object_data: See the return type of project_to2d
object_1: Name of the first object
object_2: Name of the second object

Return:
feature_chain: chain of feature, one feature vector for each frame (interpolated frames)
'''
def qsr_feature_extractor ( qsrlib, object_data, object_1, object_2 ):
    '''
    feature_selection between object_data
    '''
