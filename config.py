import numpy as np

class Config (object):
	playground_x = [-1,-1, 0]
	playground_dim = [2, 2, np.pi/2]

	n_objects = 2 #
	block_size = 0.15
	progress_threshold = 0.8

	num_steps = 20
	n_input = 13
	
	hop_step = 2
	
	train_batch_size = 10
	test_batch_size = 2
	
	train_percentage = 0.8
	test_percentage = 0.2
	
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


class Raw_Config (Config):
	n_input = 40