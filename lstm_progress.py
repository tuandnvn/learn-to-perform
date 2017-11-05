from collections import deque

import numpy as np
import tensorflow as tf

try:
    from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell
except:
    from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

class LSTM_Progress(object):
	def __init__(self, is_training, config):
		pass