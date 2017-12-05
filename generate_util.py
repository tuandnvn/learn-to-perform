import numpy as np
from utils import SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, SESSION_FEAT,\
            START, END, LABEL

def generate_data(rearranged_data, rearranged_lbls, config) :
    """
    
    Parameters:
    -----------
    rearranged_data:  num_steps, data_point_size)
    
    config has values to create training and testing data
    
    training_data and testing_data just break apart the rearranged_data and rearranged_lbl
    
    Return:
    -------
    training_data: (train_epoch_size, train_batch_size, num_steps, data_point_size)
    training_lbl:  (train_epoch_size, train_batch_size, num_steps)
                   
    testing_data:  (test_epoch_size, test_batch_size, num_steps, data_point_size)
    testing_lbl:   (test_epoch_size, test_batch_size, num_steps)
    """
    training_data = []
    testing_data = []




def linear_progress_lbl_generator(session_data):
    """
    Parameters:
    -----------
    session_data: 
    each session is a dictionary
    dict_keys(['session_events', 'session_name', 'session_objects', 'session_length'])
    
    Return:
    -------
    labels: [float] progress for each frame, from 0 to 1 based on session_events
    """
    
    # Just in case they are not sorted
    sorted_events = sorted( session_data[SESSION_EVENTS], key = lambda event: event[START] )
    
    lbls = []
    last_p = 0
    for event in sorted_events:
        prev_p = event[START]
        next_p = event[END]
        
        lbls += [0] * (prev_p - last_p)
        lbls += list(np.arange(0.0, 1.0 + 1.0 / (next_p - prev_p), 1.0 / (next_p - prev_p)))
    
        last_p = next_p + 1
    return np.array(lbls)

def turn_to_intermediate_data(project_data, data_point_size, num_steps, hop_step):
    """
    A function to generate a pair of batch-data (x, y)
    
    Parameters:
    -----------
    - project_data: a list of session_data (in ecat-learning, )
    each session is a dictionary
    dict_keys(['session_events', 'session_name', 'session_objects', 'session_length'])
    
    Notice that you need to preprocess session_data before passing it
    for each session_data in data:
        preprocess session_data[SESSION_OBJECTS] into session_data[SESSION_FEAT]:
        - projecting them into 2d using session_util.project_to2d
        - interpolate them using session_util.interpolate_multi_object_data
        - downsample them to an appropriate speed
        - get feature by running qsr_feature_extractor with the help of get_location_objects_most_active
    
    - data_point_size: Vector feature size
    - num_steps: A fix number of steps for each sample
    - hop_step: A fix number of frame offset btw two events
    
    Return:
    -------
    rearranged_data: (# samples, num_steps, data_point_size)
    rearranged_lbls: (# samples, num_steps)
    """
    samples = 0   # Number of samples of interpolating
    
    for session_data in project_data:
        correct_no_samples = ( session_data[SESSION_LEN] - num_steps ) // hop_step + 1
        
        samples += correct_no_samples
        
    print('Total number of samples' + str(samples))
    
    # At any time, 
    interpolated_data = np.zeros([samples * num_steps, data_point_size], dtype=np.float32)
    interpolated_lbls = np.zeros([samples * num_steps], dtype=np.float32)
    
    sample_counter = 0
    for session_data in project_data:
        feature_data = session_data[SESSION_FEAT]
               
        correct_no_samples = ( len(feature_data) - num_steps ) // hop_step + 1
    
        lbls = linear_progress_lbl_generator(session_data)
        
        for i in range(correct_no_samples):
            for j in range(num_steps):
                interpolated_data[( ( sample_counter + i ) * num_steps + j)] =\
                             feature_data[i * hop_step + j]
                    
                interpolated_lbls[( ( sample_counter + i ) * num_steps + j)] = lbls[i * hop_step + j]
            
        sample_counter += correct_no_samples
    
    # Divide the first dimension from samples * num_steps -> (samples, num_steps)
    rearranged_data = interpolated_data.reshape((samples, num_steps, data_point_size))
    
    rearranged_lbls = interpolated_lbls.reshape((samples, num_steps))
    
    return (rearranged_data, rearranged_lbls)