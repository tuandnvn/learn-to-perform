import numpy as np

class Config (object):
	# For reinforce learnign
	playground_x = [-1,-1, 0]
	playground_dim = [2, 2, np.pi/2]

	n_objects = 2 #
	block_size = 0.15
	progress_threshold = 0.6

	num_steps = 20
	n_input = 13
	
	state_dimension = 12
	action_dimension = 3

	num_episodes = 1000
	discount_factor = 1

	# Event progress learner parameters
	hop_step = 1
	batch_size = 10
	train_percentage = 0.8
	validation_percentage = 0.1
	test_percentage = 0.1
	
	# Progress learner hyper-parameters
	s2s = False
	# Typical value for LSTM
	keep_prob = 0.6
	hidden_size = 100
	num_layers = 2
	max_epoch = 10
	max_max_epoch = 50
	lr_decay = 0.97
	learning_rate = 0.005
	max_grad_norm = 5
	epoch_shuffle = False
	optimizer = 'adam'

	# Whether we shuffle training data in each epoch
	epoch_shuffle = False

	# value estimator
	weight_regularizer_scale = 0.1
	policy_learning_rate = 0.001
	policy_decay = 0.96
	value_learning_rate = 0.001
	value_decay = 0.96

class Raw_Config (Config):
	n_input = 40