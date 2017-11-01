'''
Created on Oct 15, 2017

@author: Tuan

===========================
Projecting a segment of data into two dimensional plane 
delineated by the table surface
'''
import numpy as np
import bisect
from utils import SESSION_OBJECTS, SESSION_LEN, BLOCK_SIZE
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

            interpolate_object_data(session_data[SESSION_LEN])
    return object_data

'''
Interpolate data for missing frames

===========
Params: 
object_data: chain of features, one feature vector for each frame (interpolated frames) for one object


'''
def interpolate_object_data( one_object_data, session_len ):
    sorted_keys = sorted(one_object_data.keys())
    for frame in range(session_len):
        if frame not in one_object_data:
            # missing frame
            frame_position = bisect.bisect_left(sorted_keys, frame)

            if frame_position == 0:
                # missing at the beginning
                one_object_data[frame] = one_object_data[sorted_keys[0]]
            elif frame_position == len(sorted_keys):
                # missing at the end
                one_object_data[frame] = one_object_data[sorted_keys[-1]]
            else:
                pre_key = sorted_keys[frame_position - 1]
                nex_key = sorted_keys[frame_position]
                pre = one_object_data[pre_key]
                nex = one_object_data[nex_key]

                p = (frame - pre_key)/(nex_key - pre_key)
                q = (nex_key - frame)/(nex_key - pre_key)
                one_object_data[frame] = Cube2D( transform = nex * p + pre * q)

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
def qsr_feature_extractor ( qsrlib, object_data, object_1, object_2, session_len ):
    '''
    feature_selection between two objects
    # of features = 13
    
    8 features here
    (o1.position, o2.position) - cardir, cardir_diff, argd, argd_diff, qtccs 4 features

    -- other features
    
    2 features
    quantized rotation of two objects
    1 feature
    quantized rotation difference between two objects
    2 features
    quantized difference of rotations btw two frames of two objects
    '''
    o1 = [Object_State(name="o1", timestamp=i, x=object_1.position[0], y=object_1.position[1]) 
            for i in range(session_len)]
    o2 = [Object_State(name="o2", timestamp=i, x=object_2.position[0], y=object_2.position[1]) 
            for i in range(session_len)]

    world = World_Trace()
    world.add_object_state_series(o1)
    world.add_object_state_series(o2)

    qsrlib_request_message = QSRlib_Request_Message(which_qsr=['cardir', 'argd', 'qtccs'], input_data=world, 
                    dynamic_args = {'cardir': {'qsrs_for': [('o1', 'o2')]},
                                    'argd': {'qsrs_for': [('o1', 'o2')], 
                                            'qsr_relations_and_values' : dict(("" + str(i), i * BLOCK_SIZE / 2) for i in xrange(20)) },
                                    'qtccs': {'qsrs_for': [('o1', 'o2')], 
                                              'quantisation_factor': 0.001, 'angle_quantisation_factor' : np.pi / 5,
                                              'validate': False, 'no_collapse': True
                                   }})

    # Number of features that you calculate the difference between two consecutive frames
    diff_feature = 2
    try:
        # pretty_print_world_qsr_trace(['cardir', 'mos', 'argd', 'qtccs'], qsrlib_response_message)
        qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)
        qsr_feature = turn_response_to_features([('o1', 'o2')], qsrlib_response_message, diff_feature)


    except ValueError, e:
        print e
        print 'Problem in data of length ' + str(len_data)
        return []

'''
'''
def turn_response_to_features(keys, qsrlib_response_message, diff_feature, all_feature):
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

    # The first frame doesn't has qtcc relations
    feature_chain[0] += [0, 0, 0, 0]
    
    diff_feature_chain = [ [feature_chain[t + 1][i] - feature_chain[t][i] 
                            for i in xrange(len(feature_chain[0]) - 7) ] + \
                          [feature_chain[t][i] for i in xrange(len(feature_chain[0]) - 7, len(feature_chain[0]))]
                    for t in xrange(len(feature_chain) - 1)]
    
    diff_feature_chain = [[0 for i in xrange(len(feature_chain[0]))]] +  diff_feature_chain

    # Concatenate features
    # feature_chain = [feature_chain[i] + diff_feature_chain[i] for i in xrange(len(feature_chain))]

    return diff_feature_chain