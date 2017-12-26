import time
import numpy as np
import tensorflow as tf
try:
    from tensorflow.nn.rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell
except:
    from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

import stateful_lstm

from config import Config, Raw_Config
from project import Project
from generate_utils import gothrough

class NextFramePredictor(object):
	def __init__(self, is_training, name=None, config = Config()):
		
