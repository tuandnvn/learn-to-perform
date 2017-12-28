"""
Predict location of the most active object given the current configuration
This prediction function should be independent of the progress function.

This preditive function could predict the next mode of the move (with small standard deviation)
where as the progress function decides when to stop.
"""
class NextFramePredictor(object):
	def __init__(self, config = Next_Frame_Config(), scope="next_frame_estimator"):
		self.config = config
        self.num_steps = num_steps = config.num_steps
		self.n_input = n_input = config.n_input
        self.size = size = config.hidden_size

        with tf.variable_scope(name):
        	self._input_data = tf.placeholder(tf.float32, [None, num_steps, n_input])

        	self._targets = tf.placeholder(tf.float32, [None, n_output])

        	if is_training:
                self.lr = tf.Variable(0.0, trainable=False)
            
            lstm_cell = BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple=True)
            
            if is_training and config.keep_prob < 1:
                lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
                
            multi_lstm_cell = MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
            

