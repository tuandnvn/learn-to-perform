import numpy as np
from utils import SESSION_NAME, SESSION_OBJECTS, SESSION_EVENTS, SESSION_LEN, SESSION_FEAT,\
            START, END, LABEL

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

    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    
    train_percentage = config.train_percentage
    test_percentage = config.test_percentage
    
    random_indices = np.arange(samples)
    np.random.shuffle(random_indices)
    
    split = int(train_percentage * samples)
    
    training_data = rearranged_data[random_indices[:split]]
    training_lbl = rearranged_lbls[random_indices[:split]]
    
    testing_data = rearranged_data[random_indices[split:]]
    testing_lbl = rearranged_lbls[random_indices[split:]]
    
    train_epoch_size = int(split // train_batch_size)
    test_epoch_size = int(test_percentage * samples // test_batch_size)
    
    training_data = training_data[:train_epoch_size * train_batch_size].\
                    reshape((train_epoch_size, train_batch_size, num_steps, n_input))
    training_lbl = training_lbl[:train_epoch_size * train_batch_size].\
                    reshape((train_epoch_size, train_batch_size, num_steps))
    
    testing_data = testing_data[:test_epoch_size * test_batch_size].\
                    reshape((test_epoch_size, test_batch_size, num_steps, n_input))
    testing_lbl = testing_lbl[:test_epoch_size * test_batch_size].\
                    reshape((test_epoch_size, test_batch_size, num_steps))
                    
    return (training_data, training_lbl, testing_data, testing_lbl)
                    
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
        
        lbls[prev_p:next_p+1] = np.arange(next_p - prev_p + 1) / (next_p - prev_p)
        
    return np.array(lbls)

def turn_to_intermediate_data(project_data, n_input, num_steps, hop_step):
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
    
    Return:
    -------
    rearranged_data: (# samples, num_steps, n_input)
    rearranged_lbls: (# samples, num_steps)
    """
    samples = 0   # Number of samples of interpolating
    
    for session_data in project_data:
        correct_no_samples = ( session_data[SESSION_LEN] - num_steps ) // hop_step + 1
        
        samples += correct_no_samples
        
    print('Total number of samples ' + str(samples))
    
    # At any time, 
    interpolated_data = np.zeros([samples * num_steps, n_input], dtype=np.float32)
    interpolated_lbls = np.zeros([samples * num_steps], dtype=np.float32)
    
    sample_counter = 0
    for session_data in project_data:
        feature_data = session_data[SESSION_FEAT]
               
        correct_no_samples = ( len(feature_data) - num_steps ) // hop_step + 1
    
        lbls = linear_progress_lbl_generator(session_data)
        print (lbls.shape)
        
        for i in range(correct_no_samples):
            for j in range(num_steps):
                interpolated_data[( ( sample_counter + i ) * num_steps + j)] =\
                             feature_data[i * hop_step + j]
                    
                interpolated_lbls[( ( sample_counter + i ) * num_steps + j)] = lbls[i * hop_step + j]
            
        sample_counter += correct_no_samples
    
    # Divide the first dimension from samples * num_steps -> (samples, num_steps)
    rearranged_data = interpolated_data.reshape((samples, num_steps, n_input))
    
    rearranged_lbls = interpolated_lbls.reshape((samples, num_steps))
    
    return (rearranged_data, rearranged_lbls)