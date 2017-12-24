import numpy as np
from utils import SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, SESSION_FEAT,\
            START, END, LABEL, TRAINING, VALIDATING, TESTING
from simulator.utils import Cube2D

def generate_data(rearranged_data, rearranged_lbls, config) :
    """
    
    Parameters:
    -----------
    rearranged_data:  num_steps, n_input)
    
    config has values to create training and testing data
    
    training_data and testing_data just break apart the rearranged_data and rearranged_lbl
    
    Return:
    -------
    training_data: (train_epoch_size, train_batch_size, num_steps, n_input)
    training_lbl:  (train_epoch_size, train_batch_size, num_steps)
                   
    testing_data:  (test_epoch_size, test_batch_size, num_steps, n_input)
    testing_lbl:   (test_epoch_size, test_batch_size, num_steps)
    """
    samples = rearranged_data.shape[0]
    num_steps = rearranged_data.shape[1]
    n_input = rearranged_data.shape[2]
    
    training_data = []
    testing_data = []

    batch_size = config.batch_size
    
    train_percentage = config.train_percentage
    validation_percentage = config.validation_percentage    
    
    random_indices = np.arange(samples)
    np.random.shuffle(random_indices)
    
    split_1 = int(train_percentage * samples)
    split_2 = int((train_percentage + validation_percentage) * samples)

    t = {TRAINING : (0, split_1), VALIDATING : (split_1, split_2), TESTING : (split_2, samples)}
    
    training_data = rearranged_data[random_indices[:split_1]]
    training_lbl = rearranged_lbls[random_indices[:split_1]]

    validation_data = rearranged_data[random_indices[split_1:split_2]]
    validation_lbl = rearranged_lbls[random_indices[split_1:split_2]]
    
    testing_data = rearranged_data[random_indices[split_2:]]
    testing_lbl = rearranged_lbls[random_indices[split_2:]]

    training_data, training_lbl = get_data_lbl(training_data, training_lbl, batch_size , num_steps, n_input )
    validation_data, validation_lbl = get_data_lbl(validation_data, validation_lbl, batch_size , num_steps, n_input )
    testing_data, testing_lbl = get_data_lbl(testing_data, testing_lbl, batch_size , num_steps, n_input )

    return (training_data, training_lbl, validation_data, validation_lbl, testing_data, testing_lbl)


def generate_data_info(rearranged_data, rearranged_lbls, rearranged_info, config) :
    samples = rearranged_data.shape[0]
    num_steps = rearranged_data.shape[1]
    n_input = rearranged_data.shape[2]
    
    training_data = []
    testing_data = []

    batch_size = config.batch_size
    
    train_percentage = config.train_percentage
    validation_percentage = config.validation_percentage    
    
    random_indices = np.arange(samples)
    np.random.shuffle(random_indices)
    
    split_1 = int(train_percentage * samples)
    split_2 = int((train_percentage + validation_percentage) * samples)

    t = {TRAINING : (0, split_1), VALIDATING : (split_1, split_2), TESTING : (split_2, samples)}

    all_data = {}
    for data_type in [TRAINING, VALIDATING, TESTING]:
        data = rearranged_data[random_indices[t[data_type][0]:t[data_type][1]]] 
        lbl = rearranged_lbls[random_indices[t[data_type][0]:t[data_type][1]]]
        info = rearranged_info[random_indices[t[data_type][0]:t[data_type][1]]]

        all_data[data_type] = (get_reshape(data, batch_size), get_reshape(lbl, batch_size), get_reshape(info, batch_size))

    return all_data

def get_data_lbl( data , lbl, batch_size , num_steps, n_input ) :
    epoch_size = int(len(data) // batch_size)

    data = data[:epoch_size * batch_size].\
                    reshape((epoch_size, batch_size, num_steps, n_input))
    lbl = lbl[:epoch_size * batch_size].\
                    reshape((epoch_size, batch_size, num_steps))

    return (data, lbl)

def get_reshape(data, batch_size):
    # Split the first dimension into multiple batch_size
    epoch_size = int(len(data) // batch_size)
    new_shape = [epoch_size, batch_size] + list(data.shape[1:])
    return np.reshape(data[:epoch_size * batch_size], new_shape)

def gothrough(data, lbl):
    for i in range(np.shape(data)[0]):
        x = data[i]
        y = lbl[i]
        yield (x, y)

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
    
    lbls = np.zeros(session_data[SESSION_LEN])
    
    for event in sorted_events:
        prev_p = event[START]
        next_p = event[END]
        
        lbls[prev_p:next_p+1] = np.arange(next_p - prev_p + 1, dtype=np.float32) / (next_p - prev_p)
        
    return np.array(lbls)

def linear_progress_lbl_generator_retreat(session_data, retreat_phase = 4):
    """
    Adding a retreat phase that allows smooth decrease of progress value
    For example: if retreat_phase = 4, the value of event progress would reduce from 1 
    to 0 through 1, 0.75, 0.5, 0.25, 0

    Parameters:
    -----------
    session_data: 
    each session is a dictionary
    dict_keys(['session_events', 'session_name', 'session_objects', 'session_length'])
    
    Return:
    -------
    labels: [float] progress for each frame, from 0 to 1 based on session_events
    """
    sorted_events = sorted( session_data[SESSION_EVENTS], key = lambda event: event[START] )
    
    lbls = np.zeros(session_data[SESSION_LEN])
    
    for event in sorted_events:
        prev_p = event[START]
        next_p = event[END]
        
        lbls[prev_p:next_p+1] = np.arange(next_p - prev_p + 1, dtype=np.float32) / (next_p - prev_p)

        # Sometimes next_p + retreat_phase + 1 is more than the length of session
        q = len(lbls[next_p+1:next_p+retreat_phase+1])

        lbls[next_p+1:next_p+retreat_phase+1] = np.arange(retreat_phase - 1, -1, -1, dtype=np.float32)[:q] / retreat_phase
        
    return np.array(lbls)


def turn_to_intermediate_data(project_data, n_input, num_steps, hop_step, 
        linear_progress_lbl_func = linear_progress_lbl_generator):
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
    
    - n_input: Vector feature size
    - num_steps: A fix number of steps for each sample
    - hop_step: A fix number of frame offset btw two events
    - linear_progress_lbl_func: a function that when you give it a session, it gives 
    you a lbls list of length the same as the session
    
    Return:
    -------
    rearranged_data: (# samples, num_steps, n_input)
    rearranged_lbls: (# samples, num_steps)
    """
    print ('turn_to_intermediate_data with n_input = %d, num_steps = %d, hop_step = %d' % (n_input, num_steps, hop_step))
    samples = 0   # Number of samples of interpolating
    
    for session_data in project_data:
        session_sample = ( session_data[SESSION_LEN] - num_steps ) // hop_step + 1
        
        samples += session_sample
        
    print('Total number of samples ' + str(samples))

    # At any time, 
    interpolated_data = np.zeros([samples * num_steps, n_input], dtype=np.float32)
    interpolated_lbls = np.zeros([samples * num_steps], dtype=np.float32)
    
    sample_counter = 0
    for session_data in project_data:
        feature_data = session_data[SESSION_FEAT]
               
        session_sample = ( len(feature_data) - num_steps ) // hop_step + 1
    
        lbls = linear_progress_lbl_func(session_data)
        print (lbls.shape)
        
        for i in range(session_sample):
            for j in range(num_steps):
                interpolated_data[( ( sample_counter + i ) * num_steps + j)] =\
                             feature_data[i * hop_step + j]
                    
                interpolated_lbls[( ( sample_counter + i ) * num_steps + j)] = lbls[i * hop_step + j]
            
        sample_counter += session_sample
    
    # Divide the first dimension from samples * num_steps -> (samples, num_steps)
    rearranged_data = interpolated_data.reshape((samples, num_steps, n_input))
    
    rearranged_lbls = interpolated_lbls.reshape((samples, num_steps))
    
    return (rearranged_data, rearranged_lbls)

def get_rescale ( session_data, from_frame, to_frame, scale):
    """
    To generate some data from_frame, to_frame at a scale
    
    For example, we can generate data from 0 to 30 (exclusive) at scale 1.5 -> result is 20 frames

    We should only accept .5

    Parameters:
    -----------
    session_data: typical session data

    Return:
    ----------
    
    """
    assert np.isclose(scale * 2, int(scale * 2))

    result = []
    if np.isclose(scale, int(scale)):
        # Downscaling
        t = int(scale)
        for i in range(from_frame, to_frame, t):
            result.append( session_data[SESSION_FEAT][i] )

        return result

    else:
        t = int(scale * 2)
        for i in range(from_frame, to_frame, t):
            result.append( session_data[SESSION_FEAT][i] )
            middle = np.array(session_data[SESSION_FEAT][int(i + scale)]) * 0.5 + np.array(session_data[SESSION_FEAT][int(i + scale + 1)]) * 0.5
            result.append( middle )
        return result

def get_rescale_lbls(lbls, from_frame, to_frame, scale, num_steps):
    """
    To generate labels from from_frame to to_frame at a scale
    
    For example, we can generate labels for frames from 0 to 30 (exclusive) at scale 1.5 -> result is labels for 20 frames

    We should only accept .5

    We always give some discount of the label over the segment
    Example:
    Event: 5 to 33
    
    Take segment: 
    -    0 to 20, scale = 1, -> lbls would be 0 at the begining, than increase to 15/28 at frame 19
    -    15 to 35, scale = 1 -> it use to be that it would increase to 1 than start to decrease
                            -> It now only increases to (33-15)/28 before start to decrease. It decreases with twice the rate of increasing

    This way it would be much harder to get to maximum progress value of 1. However, it seems like we give higher progress for 
    action with no-move at the beginning to action with no-move at the end? That's probably ok because we don't actually have 
    no-move at the end when we interpolate.

    If this segment crosses two action-demonstrations, we should interpolate according to the last event.

    Parameters:
    -----------
    lbls: linear progress labels

    Return:
    ----------
    lbsl:  size = num_steps = (to_frame - from_frame) // scale
    """
    assert num_steps == (to_frame - from_frame) // scale
    assert np.isclose(scale * 2, int(scale * 2))

    new_lbls = []

    result = []
    if np.isclose(scale, int(scale)):
        # Downscaling
        t = int(scale)
        for i in range(from_frame, to_frame, t):
            result.append( lbls[i] )
    else:
        t = int(scale * 2)
        for i in range(from_frame, to_frame, t):
            result.append( lbls[i] )
            lbl = (lbls[int(i + scale)] + lbls[int(i + scale + 1)]) / 2
            result.append( lbl )

    return np.maximum(result - result[0],[0] * num_steps)
        

def turn_to_intermediate_data_multiscale(project_data, n_input, num_steps, hop_step, scales = [1.0, 1.5, 2.0]):
    """
    A function to generate a pair of batch-data (x, y)

    The difference between this and turn_to_intermediate_data is that
    this would generate data with different scaling level (1, 1.5, 2 etc.)
    and also with different kind of progress lbls.
    For example, if the action is from frame 1 to frame 33
    [0, 20] ~ 0.7
    [0, 30] ~ 1 (rescale)
    [0, 40] ~ 0.9 (rescale)

    We also generate more data than normal because we rescale at different scales
    
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
    
    - n_input: Vector feature size
    - num_steps: A fix number of steps for each sample
    - hop_step: A fix number of frame offset btw two events
    - linear_progress_lbl_func: a function that when you give it a session, it gives 
    you a lbls list of length the same as the session
    - scales: 
    
    Return:
    -------
    rearranged_data: (# samples, num_steps, n_input)
    rearranged_lbls: (# samples, num_steps)
    """
    print ('turn_to_intermediate_data_multiscale with n_input = %d, num_steps = %d, hop_step = %d, scales = %s' % (n_input, num_steps, hop_step, str(scales)))
    samples = 0   # Number of samples of interpolating
    
    for session_data in project_data:
        for scale in scales:
            session_sample_scale = ( session_data[SESSION_LEN] - int(num_steps * scale) ) // hop_step + 1
            
            print ('For scale = %.1f ; session_sample_scale = %d' % (scale, session_sample_scale) )
            samples += session_sample_scale
        
    print('Total number of samples ' + str(samples))

    # At any time, 
    interpolated_data = np.zeros([samples * num_steps, n_input], dtype=np.float32)
    interpolated_lbls = np.zeros([samples * num_steps], dtype=np.float32)
    
    sample_counter = 0
    for session_data in project_data:
        lbls = linear_progress_lbl_generator_retreat(session_data, retreat_phase = 10)

        for scale in scales:
            feature_data = session_data[SESSION_FEAT]
                   
            session_sample_scale = ( len(feature_data) - int(num_steps  * scale) ) // hop_step + 1
            
            for i in range(session_sample_scale):
                # num_steps features
                scaled_features = get_rescale (session_data, i, i + int(num_steps  * scale), scale)
                # num_steps lbls
                scaled_lbls = get_rescale_lbls(lbls, i, i + int(num_steps  * scale), scale, num_steps)

                interpolated_data[( sample_counter + i ) * num_steps:  ( sample_counter + i + 1) * num_steps ] = scaled_features        
                interpolated_lbls[( sample_counter + i ) * num_steps:  ( sample_counter + i + 1) * num_steps ] = scaled_lbls
                
            sample_counter += session_sample_scale
    
    # Divide the first dimension from samples * num_steps -> (samples, num_steps)
    rearranged_data = interpolated_data.reshape((samples, num_steps, n_input))
    
    rearranged_lbls = interpolated_lbls.reshape((samples, num_steps))
    
    return (rearranged_data, rearranged_lbls)