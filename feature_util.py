'''
Created on Oct 15, 2017

@author: Tuan

===========================
Projecting a segment of data into two dimensional plane 
delineated by the table surface
'''
import numpy as np
import bisect

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace

from feature.project_table import project_markers, estimate_cube_2d
from utils import SESSION_OBJECTS, SESSION_LEN, BLOCK_SIZE, ROTATION_QUANTIZATION

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


def project_to2d ( session_data, from_frame = 0, to_frame = 10000 ):
    object_data = {}

    for object_name in session_data[SESSION_OBJECTS]:
        if object_name == 'table':
            polygon = []
            for frameNo in session_data[SESSION_OBJECTS][object_name]:
                frame_polygon = session_data[SESSION_OBJECTS][object_name][frameNo]

            polygon.append(frame_polygon)

            polygon = np.concatenate(polygon)
            polygon = np.reshape(polygon, (len(polygon)//3, 3) )
            
            table_markers = polygon

            # Just pick the first two points for coordination
            first_point = table_markers[0]
            second_point = table_markers[1]
    
    for object_name in session_data[SESSION_OBJECTS]:
        print ('============ ' + object_name)
        if object_name != 'table':
            object_data[object_name] = {}
            for frameNo in session_data[SESSION_OBJECTS][object_name]:
                if int(frameNo) < from_frame or int(frameNo) > to_frame:
                    continue
                frame_data = session_data[SESSION_OBJECTS][object_name][frameNo]

                # Sort firstly by number of non-finite corners
                # Sort secondly by size of marker (larger marker means better resolution)
                # Size of marker should be only based on first two dimensions
                # The third dimension might be very noisy
                q = [((count_finite(frame_data[face_index]), area_dimension(frame_data[face_index]) ), face_index) 
                    for face_index in frame_data]
                q = sorted(q, key = lambda t: t[0], reverse = True)

                # Pick out the face_index with the most number of non-infinite values
                best_face_index = q[0][1]
                #print ('----------- ' + frameNo)
                #print (repr(frame_data[best_face_index]))
                rectangle_projected = project_markers ( frame_data[best_face_index], table_markers )
                #print (repr(rectangle_projected))
                
                try:
                    object_data[object_name][int(frameNo)] = estimate_cube_2d ( rectangle_projected, first_point, second_point )
                 #   print (object_data[object_name][int(frameNo)])
                except:
                    continue

    return object_data


def interpolate_multi_object_data( session_len, object_data, object_names = [] ):
    """
    
    """
    data = {}
    for object_name in object_names:
        data[object_name] = _interpolate_object_data( session_len, object_data[object_name])
    return data


def _interpolate_object_data( session_len, one_object_data ):
    '''
    Interpolate data for missing frames

    ===========
    Params: 
    object_data: chain of features, one feature vector for each frame (interpolated frames) for one object
    '''
    new_one_object_data = {}
    sorted_keys = sorted(one_object_data.keys())
    for frame in range(session_len):
        if frame not in one_object_data:
            # missing frame
            frame_position = bisect.bisect_left(sorted_keys, frame)

            if frame_position == 0:
                # missing at the beginning
                new_one_object_data[frame] = one_object_data[sorted_keys[0]]
            elif frame_position == len(sorted_keys):
                # missing at the end
                new_one_object_data[frame] = one_object_data[sorted_keys[-1]]
            else:
                pre_key = sorted_keys[frame_position - 1]
                nex_key = sorted_keys[frame_position]
                pre = one_object_data[pre_key].transform
                nex = one_object_data[nex_key].transform
                
                p = (frame - pre_key)/(nex_key - pre_key)
                q = (nex_key - frame)/(nex_key - pre_key)
                transfrom = Transform2D ( nex.position * p + pre.position * q , 
                                         nex.rotation * p + pre.rotation * q, 
                                         nex.scale * p + pre.scale * q)
                new_one_object_data[frame] = Cube2D( transfrom )
        else:
            new_one_object_data[frame] = one_object_data[frame]
    return new_one_object_data

cdid = dict( (u, i) for (i, u) in enumerate( ['n', 'nw', 'w', 'sw', 's', 'se', 'e', 'ne', 'eq'] ))
mosd = dict( (u, i) for (i, u) in enumerate( ['s', 'm'] ))
qtcc_relations = dict( (u, i) for (i, u) in enumerate( ['-', '0', '+'] ))

def cardir_index ( cardir ):
    return cdid [cardir]

def mos_index ( mos ):
    return mosd [mos]

def qtcc_index ( qtcc_relation ):
    return qtcc_relations [qtcc_relation] - 1


def qsr_feature_extractor ( qsrlib, object_data, object_1_name, object_2_name, session_len ):
    '''
    Get the features from 
    - one object as the one mainly under movement (object slot)
    - one object that is relatively static (locative slot)

    Params: 
    qsrlib: check the return value of read_utils.load_one_param_file
    object_data: See the return type of project_to2d
    object_1_name: Name of the first object
    object_2_name: Name of the second object
    session_len: Length of session (int)

    Return:
    feature_chain: chain of feature, one feature vector for each frame (interpolated frames)
    
    ============================================================
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
    object_1 = object_data[object_1_name]
    object_2 = object_data[object_2_name]

    o1 = [Object_State(name="o1", timestamp=i, x=object_1[i].transform.position[0][0], y=object_1[i].transform.position[0][1]) 
            for i in range(session_len)]
    o2 = [Object_State(name="o2", timestamp=i, x=object_2[i].transform.position[0][0], y=object_2[i].transform.position[0][1]) 
            for i in range(session_len)]

    world = World_Trace()
    world.add_object_state_series(o1)
    world.add_object_state_series(o2)

    qsrlib_request_message = QSRlib_Request_Message(which_qsr=['cardir', 'argd', 'qtccs'], input_data=world, 
                    dynamic_args = {'cardir': {'qsrs_for': [('o1', 'o2')]},
                                    'argd': {'qsrs_for': [('o1', 'o2')], 
                                            'qsr_relations_and_values' : dict(("" + str(i), i * BLOCK_SIZE / 2) for i in range(20)) },
                                    'qtccs': {'qsrs_for': [('o1', 'o2')], 
                                              'quantisation_factor': 0.001, 'angle_quantisation_factor' : np.pi / 5,
                                              'validate': False, 'no_collapse': True
                                   }})

    # Number of features that you calculate the difference between two consecutive frames
    diff_feature = 2
    try:
        # pretty_print_world_qsr_trace(['cardir', 'mos', 'argd', 'qtccs'], qsrlib_response_message)
        qsrlib_response_message = qsrlib.request_qsrs(req_msg=qsrlib_request_message)

        # (#frame, 8)
        qsr_feature = _turn_response_to_features([('o1,o2')], qsrlib_response_message, diff_feature)

        # rotation features
        quantized_r_1 = np.array([object_1[i].transform.rotation // ROTATION_QUANTIZATION for i in range(session_len)])
        quantized_r_2 = np.array([object_2[i].transform.rotation // ROTATION_QUANTIZATION for i in range(session_len)])
        quantized_diff = quantized_r_1 - quantized_r_2
        diff_quantized_r_1 = np.pad(np.ediff1d(quantized_r_1), (1,0), 'constant', constant_values = (0,))
        diff_quantized_r_2 = np.pad(np.ediff1d(quantized_r_2), (1,0), 'constant', constant_values = (0,))

        # column forms
        quantized_r_1.shape = (session_len, 1)
        quantized_r_2.shape = (session_len, 1)
        quantized_diff.shape = (session_len, 1)
        diff_quantized_r_1.shape = (session_len, 1)
        diff_quantized_r_2.shape = (session_len, 1)

        return np.concatenate([qsr_feature, quantized_r_1, quantized_r_2, quantized_diff, diff_quantized_r_1, diff_quantized_r_2], axis = 1)

    except ValueError as e:
        print (e)
        print ('Problem in data of length ' + str(len_data))
        return []


def _turn_response_to_features(keys, qsrlib_response_message, diff_feature):
    """
    diff_feature: number of features at the beginning that need to create difference between two frames
    all_feature: total number of features
    """
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
    feature_chain[0] += [0 for i in range(4)]

    feature_chain = np.array(feature_chain)
    
    print (feature_chain.shape)
    # number of features
    f_number = feature_chain.shape[1]

    # Feature that need to calculate diff
    # (#frame, diff_feature)
    need_diff_chain = feature_chain[:, :diff_feature]

    # Get the diff
    # (#frame - 1, diff_feature)
    diff_chain = np.diff(need_diff_chain, n=1, axis = 0)

    # (#frame, diff_feature)
    padded_diff_chain = np.pad(diff_chain, [(1,0), (0,0)], 'constant', constant_values = (0,))

    
    
    # Add for the first frame
    # (#frame, 2 * diff_feature + other_feature)
    diff_feature_chain = np.concatenate ( [need_diff_chain, padded_diff_chain, feature_chain[:, diff_feature:]], axis = 1 )
    
    print ('shape of diff_feature_chain %s' % str(diff_feature_chain.shape))
    return diff_feature_chain