import numpy as np

QUAN = 'QUANTITATIVE'
QUAL = 'QUALITATIVE'

class Config (object):
	# For reinforce learning
	playground_x = [-1.1,-1.1, 0]
	playground_dim = [2.2, 2.2, np.pi/2]

	n_objects = 2 #
	# Real block_size = 0.15
	block_size = 0.20
	progress_threshold = 0.95

	num_steps = 20
	
	state_dimension = 4
	action_dimension = 2

	num_episodes = 20
	discount_factor = 1

	# Event progress learner parameters
	hop_step = 1
	batch_size = 30
	train_percentage = 0.8
	validation_percentage = 0.1
	test_percentage = 0.1
	
	# Progress learner hyper-parameters
	s2s = False
	# Typical value for LSTM
	keep_prob = 0.6
	hidden_size = 200
	num_layers = 2
	max_epoch = 10
	max_max_epoch = 50
	lr_decay = 0.95
	learning_rate = 0.001
	max_grad_norm = 5
	epoch_shuffle = False
	optimizer = 'adam'

	# Whether we shuffle training data in each epoch
	epoch_shuffle = False

	# value estimator
	weight_regularizer_scale = 0.1
	policy_learning_rate = 0.1
	policy_decay = 0.92
	policy_decay_every = 100
	value_learning_rate = 0.01
	value_decay = 0.92
	value_decay_every = 100
	value_estimator_hidden_size = 20
	# Set policy decay and value_decay to 1 makes nan values?

	# values used for sigma
	constraint_sigma = 0.0
	start_sigma = np.array([2.0, 2.0])
	end_sigma = np.array([1.0, 1.0])

	# Fail action penalty
	failed_action_penalty = 0.00

	# searching with branching configuration parameters
	no_of_start_setups = 10 # In each loop, we start with 10 random setups - train setups
	no_of_test_setups = 10 # We also want to test the learned algorithm on some test configurations
	# strong_progress_threshold = 0.9 # For this, because we will search in a large space, and going deep
									# so we will raise the value of progress_threshold
	sigma_discount_factor = 0.7 # In each loop, we reduce sigma by this factor.
								# However, we should stop 
	no_of_loops = 10 

	keep_branching = 9
	branching = 36 # Target is so that at the last loop, we reduce branching down so that we lower
				   # the number of searching steps we spend

	interactive_alpha = 0.1


class Raw_Config (Config):
	n_input = 40

class Qual_Config (Config):
	n_input = 7
	input_type = QUAL

class Quan_Config (Config):
	n_input = 40
	input_type = QUAN

class Next_Frame_Config (object):
	# Probably just simple location/orientation difference btw two objects
	n_input = 3
	n_output = 3

class Qual_Plan_Config (Config):
	state_dimension = 150 # = 5 () * 6 () * 5 ()
	action_dimension = 5