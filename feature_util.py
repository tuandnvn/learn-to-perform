'''
Created on Oct 15, 2017

@author: Tuan

===========================
Projecting a segment of data into two dimensional plane 
delineated by the table surface

Also handling QSR features 
'''
import numpy as np
import bisect

from feature.project_table import project_markers, estimate_cube_2d
from utils import SESSION_OBJECTS, SESSION_LEN, BLOCK_SIZE, ROTATION_QUANTIZATION, SESSION_OBJ_2D, SESSION_FEAT
from session_util import calculate_distance, calculate_distance_btw

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_trace import Object_State, World_Trace


cdid = dict( (u, i) for (i, u) in enumerate( ['n', 'nw', 'w', 'sw', 's', 'se', 'e', 'ne', 'eq'] ))
mosd = dict( (u, i) for (i, u) in enumerate( ['s', 'm'] ))
qtcc_relations = dict( (u, i) for (i, u) in enumerate( ['-', '0', '+'] ))

def cardir_index ( cardir ):
    return cdid [cardir]

def mos_index ( mos ):
    return mosd [mos]

def qtcc_index ( qtcc_relation ):
    return qtcc_relations [qtcc_relation] - 1


def get_location_objects_default(object_data, object_names, session_len):
    """
    This method create two Object_State lists corresponding to 
    location streams of two objects

    The default case is when the first object is always the moving object
    while the second object is always the static object.

 
    Params:
    ======== 
    object_data: See the return type of project_to2d
    object_names: Names of the objects

    Return:
    ========
    o1: list of qsrlib_io.world_trace.Object_State of the first (moving) object
    o2: list of qsrlib_io.world_trace.Object_State of the second (static) object
    """
    
    if len(object_names) < 2:
        raise Exception ("len(object_names)  < 2")


    object_1_name = object_names[0]
    object_2_name = object_names[1]

    object_1 = object_data[object_1_name]
    object_2 = object_data[object_2_name]

    return (object_1, object_2)

def get_location_objects_most_active(object_data, object_names, session_len):
    """
    Same as get_location_objects_default

    But we calculate the movement of each object for a short period of time
    to pick which one is the moving object (more salient object)


    """
    if len(object_names) < 2:
        raise Exception ("len(object_names)  < 2")

    step  = 5

    object_1 = []
    object_2 = []

    for start in range(0,int(session_len),step):
        d_s = []
        end = int(min(session_len, start + step))

        # We just calculate distance has been travelled by each object
        for object_name in object_names:
            one_object_data = object_data[object_name]

            
            # Calculate travelling distance of this object for 5 frames
            d = calculate_distance( one_object_data, 1, start, end)

            d_s.append((object_name, d))


        # Sort the objects by longest travelled distance, longest first
        d_s = sorted(d_s, key = lambda v:v[1], reverse = True)

        object_1_name = d_s[0][0]

        # For the second object, it should be the closest object to the moving object
        d_2_s = []
        for name, _ in d_s[1:]:
            d = calculate_distance_btw(object_data[object_1_name], object_data[name], 1, start, end)
            d_2_s.append((name, d))
        
        object_2_name = d_2_s[0][0]
        #print ('start = %d ; object_1_name = %s ; object_2_name = %s'%(start, object_1_name, object_2_name))

        object_1 += object_data[object_1_name][start:end]
        object_2 += object_data[object_2_name][start:end]  

    return (object_1, object_2)        

def qsr_feature_extractor ( session, get_location_objects = get_location_objects_default, qsrlib = None):
    '''
    Get the features from 
    - one object as the one mainly under movement (object slot)
    - one object that is relatively static (locative slot)

    Params:
    ======== 
    qsrlib: check the return value of read_utils.load_one_param_file
    object_data: See the return type of project_to2d
    object_names: names of objects that need to find features
    session_len: Length of session (int)
    get_location_objects: a function (object_data, object_names) -> (o1, o2)

    Return:
    ========
    None 
    Add session[SESSION_FEAT] = feature_chain: chain of feature, one feature vector for each frame (interpolated frames)
    
    ============================================================
    feature_selection between two objects
    # of features = 13
    
    8 features here
    (o1.position, o2.position) - cardir, argd, cardir_diff, argd_diff, qtccs 4 features

    -- other features
    
    2 features
    quantized rotation of two objects
    1 feature
    quantized rotation difference between two objects
    2 features
    quantized difference of rotations btw two frames of two objects
    '''
    if qsrlib == None:
        qsrlib = QSRlib()
    object_data = session[SESSION_OBJ_2D]
    session_len = session[SESSION_LEN]
    object_names = object_data.keys()

    object_1, object_2 = get_location_objects(object_data, object_names, session_len)

    o1 = [Object_State(name="o1", timestamp=i, x=object_1[i].transform.position[0][0], y=object_1[i].transform.position[0][1]) 
            for i in range(0, session_len)]
    o2 = [Object_State(name="o2", timestamp=i, x=object_2[i].transform.position[0][0], y=object_2[i].transform.position[0][1]) 
            for i in range(0, session_len)]

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

        session[SESSION_FEAT] = np.concatenate([qsr_feature, quantized_r_1, quantized_r_2, quantized_diff, diff_quantized_r_1, diff_quantized_r_2], axis = 1)

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