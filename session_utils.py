'''
Created on December 2, 2017

@author: Tuan

===========================
This file will store methods that related to handling session data

In general, all the methods related to processing of session_data like
downsampling, calculating speed, etc.

read_utils : read data from the .param file
    ||
    \/
session_utils : session data standardize the data
    ||
    \/
feature_utils : read data from the .param file
    ||
    \/
generate_utils : generate training and testing data
    ||
    \/
learning_module
'''
import bisect
import numpy as np
from utils import SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, SESSION_OBJ_2D,\
    START, END, LABEL
from feature.project_table import estimate_cube_2d, project_markers
from simulator.utils import Cube2D, Transform2D

# Count the number of finite element in an array
def count_finite( numpy_array ):
    return np.sum( np.isfinite( numpy_array )) 


def area_dimension( numpy_array ):
    '''
    Area of polygons
    ===========
    Params: numpy_array of size ( 3 x n )

    Return: object_data: Dictionary
    '''
    s = np.reshape(numpy_array, (len(numpy_array)//3,3))
    x = s[:,0]
    y = s[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def project_to2d ( session, from_frame = 0, to_frame = 10000 ):
    """
    This method project the data of session into 2d Cube
    and insert a new object into session
    which is session[SESSION_OBJ_2D]
    """
    object_data = {}

    for object_name in session[SESSION_OBJECTS]:
        if object_name == 'table':
            polygon = []
            for frameNo in session[SESSION_OBJECTS][object_name]:
                frame_polygon = session[SESSION_OBJECTS][object_name][frameNo]

            polygon.append(frame_polygon)

            polygon = np.concatenate(polygon)
            polygon = np.reshape(polygon, (len(polygon)//3, 3) )
            
            table_markers = polygon

            # Just pick the first two points for coordination
            first_point = table_markers[0]
            second_point = table_markers[1]
    
    for object_name in session[SESSION_OBJECTS]:
        print ('============ ' + object_name)
        if object_name != 'table':
            object_data[object_name] = {}
            for frameNo in session[SESSION_OBJECTS][object_name]:
                if int(frameNo) < from_frame or int(frameNo) > to_frame:
                    continue
                frame_data = session[SESSION_OBJECTS][object_name][frameNo]

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
                    #print (str(frameNo) + ' ' +  str(object_data[object_name][int(frameNo)]))
                except:
                    continue

    session[SESSION_OBJ_2D] = object_data


def interpolate_multi_object_data( session, object_names = [] ):
    """
    
    """
    session_len = session[SESSION_LEN]
    object_data = session[SESSION_OBJ_2D]

    data = {}
    for object_name in object_names:
        data[object_name] = _interpolate_object_data( session_len, object_data[object_name])
    
    session[SESSION_OBJ_2D] = data


def _interpolate_object_data( session_len, one_object_data ):
    '''
    Interpolate data for missing frames

    ===========
    Params: 
    object_data: chain of features, one feature vector for each frame (interpolated frames) for one object
    '''
    new_one_object_data = []
    sorted_keys = sorted(one_object_data.keys())
    for frame in range(session_len):
        if frame not in one_object_data:
            # missing frame
            frame_position = bisect.bisect_left(sorted_keys, frame)

            if frame_position == 0:
                # missing at the beginning
                new_one_object_data.append(one_object_data[sorted_keys[0]])
            elif frame_position == len(sorted_keys):
                # missing at the end
                new_one_object_data.append(one_object_data[sorted_keys[-1]])
            else:
                pre_key = sorted_keys[frame_position - 1]
                nex_key = sorted_keys[frame_position]
                pre = one_object_data[pre_key].transform
                nex = one_object_data[nex_key].transform
                
                p = float(frame - pre_key)/(nex_key - pre_key)
                q = float(nex_key - frame)/(nex_key - pre_key)
                transform = Transform2D ( nex.position * p + pre.position * q , 
                                         nex.rotation * p + pre.rotation * q, 
                                         nex.scale * p + pre.scale * q)
                new_one_object_data.append(Cube2D( transform ))
        else:
            new_one_object_data.append(one_object_data[frame])
    return new_one_object_data

def calculate_distance( one_object_data, down_sample, start, end ):
    """
    Distance one object has travelled from start frame to end frame
     (only sampling at (frame - start) % down_sample == 0)
    
    Params: 
    ===========
        one_object_data: list of Cube2D
        down_sample: int
        start: inclusive
        end: exclusive
    Returns:
    ===========
    """
    prev_loc = None
    sum_d = 0
    
    for i in range(start, end, down_sample):
        cur_loc = one_object_data[i].transform.position
        
        if isinstance(prev_loc, np.ndarray):
            d = np.linalg.norm(cur_loc-prev_loc)
            sum_d += d
        prev_loc = cur_loc
        
    return sum_d

def calculate_distance_btw( first_object_data, second_object_data, down_sample, start, end ):
    """
    Average distance between two objects from start frame to end frame
    down_sample (only sampling at (frame - start) % down_sample == 0)

    """
    d_s = []
    
    for i in range(start, end, down_sample):
        first_loc = first_object_data[i].transform.position
        second_loc = second_object_data[i].transform.position
        
        d = np.linalg.norm(second_loc-first_loc)
        d_s.append(d)
        
    return np.average(d_s)

def get_down_sample_quotient(project, num_steps = 20):
    # First pass to calculate downsampling value
    # Problem if we use the original frame number is that it would make very different kind
    # of learning models
    # for example, for different actions, you have different number of frames per action, 
    # so if we need to make a model to distinguish them, it's not very robust
    # Moreover, we need to find an appropriate speed of sampling 
    # for the simulator
    
    # Finding down_sample_quotient
    # We will only keep frames % down_sample_quotient == 0
    
    lens = []
    for session in project:
        for event in session[SESSION_EVENTS]:
            # {'start': 4, 'label': ['Stella Artois', 'Shell'], 'end': 168}
            lens.append(event[END] - event[START])
    
    down_sample_quotient = int(np.average(lens) / num_steps)
    
    return down_sample_quotient

def get_action_speed(project, down_sample_quotient):
    total_travelling_dist = 0
    lens = []
    # Second pass
    # We also calculate average speed per downsampled frame
    for session in project:
        for event in session[SESSION_EVENTS]:
            end = event[END]
            start = event[START]
            lens.append(event[END] - event[START])
            
            d_s = []
            for object_name in session[SESSION_OBJ_2D]:
                d = calculate_distance(session[SESSION_OBJ_2D][object_name],
                                  down_sample_quotient, start, end)
                
                d_s.append(d)
            
            longest_dis = np.max(d_s)
            
            total_travelling_dist += longest_dis
    
    # Real speed = total_travelling_dist / np.sum(lens) unit/frame
    # Downsampled speed = Real speed * down_sample_quotient
    speed = total_travelling_dist * down_sample_quotient / np.sum(lens)
    
    return speed

def down_sample(project, down_sample_quotient):
    # In downsampling, we have to downsample the frames 
    # in session[SESSION_OBJECTS], downsample the start and end 
    # of each event in session[SESSION_EVENTS]
    
    new_project_data = []
    
    for session in project:
        new_session_data = {}
        
        new_session_data[SESSION_OBJ_2D] = {}

        # downsample session[SESSION_OBJ_2D]
        for object_name in session[SESSION_OBJ_2D]:
            new_session_data[SESSION_OBJ_2D][object_name] = session[SESSION_OBJ_2D][object_name][::down_sample_quotient]
        
        # downsample session[SESSION_EVENTS]
        new_session_data[SESSION_EVENTS] = []
        for event in session[SESSION_EVENTS]:
            end = event[END] // down_sample_quotient
            start = event[START] // down_sample_quotient
            
            new_event = {}
            new_event[LABEL] = event[LABEL]
            new_event[START] = start
            new_event[END] = end
            
            new_session_data[SESSION_EVENTS].append(new_event)
        
        # downsample session[SESSION_LEN]
        new_session_data[SESSION_LEN] = int(np.ceil(session[SESSION_LEN] // down_sample_quotient))
        
        new_session_data[SESSION_NAME] = session[SESSION_NAME]
        
        # Add back to project data
        new_project_data.append(new_session_data)
        
    return new_project_data