import numpy as np
import tensorflow as tf
from config import Config

class EventProgressEstimator(object):
    """
    Estimate the progress of event using LSTM
    """
    def __init__(self, is_training, name=None, config = Config()):
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.size = size = config.hidden_size
        # This is an option, if self.s2s = True -> Use all progress values
        # otherwise just use the last progress value
        self.s2s = config.s2s 
        
        with tf.variable_scope(name):
            "Declare all placeholders"
            "Placeholder for input"
            
            """
            batch_size
            num_steps: length of sequence
            n_input: size of feature vectors
            """
            self._input_data = tf.placeholder(tf.float32, [None, num_steps, n_input])
            
            """
            (batch_size x num_steps) for sequence to sequence
            batch_size for 
            """
            if self.s2s:
                self._targets = tf.placeholder(tf.float32, [None, num_steps])
            else:
                self._targets = tf.placeholder(tf.float32, [None])
            
            lstm_cell = BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple=True)
            
            if is_training and config.keep_prob < 1:
                lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
                
            multi_lstm_cell = MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
            
            # Initial states of the cells
            # cell.state_size = config.num_layers * 2 * size
            # Size = ( batch_size x cell.state_size )
            
            # We don't know the batch_size here, so don't need
            # to specify initial_state
            # self._initial_state = multi_lstm_cell.zero_state(batch_size, tf.float32)
            
            inputs = tf.reshape(self._input_data, [-1, n_input]) # (batch_size * num_steps, n_input)
            
            with tf.variable_scope("linear"):
                weight = tf.get_variable("weight", [n_input, size])
                bias = tf.get_variable("bias", [size])

                # (batch_size * num_steps, size)
                inputs = tf.matmul(inputs, weight) + bias
                
            inputs = tf.reshape(inputs, (-1, num_steps, size)) # (batch_size, num_steps, size)
            
            # (output, state)
            # output is of size:  ( batch_size, num_steps, size ) or (( batch_size, size ))
            # state is of size:   ( batch_size, cell.state_size ) (last state only)
            with tf.variable_scope("lstm"):
                output_and_state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs, dtype=tf.float32)
            
            output = output_and_state[0]
            # we will pass the hidden state to next run of lstm
            self._final_state = output_and_state[1]
            
            with tf.variable_scope("output_linear"):
                weight = tf.get_variable("weight", [size, 1])
                bias = tf.get_variable("bias", [1])

                
                if self.s2s:
                    # Need to reshape to 2 dimensions
                    output = tf.reshape(output, [-1, size])
                    output = output @ weight + bias
                    # ( batch_size, num_steps )  
                    output = tf.reshape(output, [-1, num_steps])
                else:
                    #( batch_size, 1)
                    # @ is the same as matmul
                    output = output @ weight + bias
                    
            # Remove all 1 dimension
            # ( batch_size, num_steps ) or (batch_size)
            self.output = tf.squeeze(output)
            
            # Loss = mean squared error of target and predictions
            self.loss = tf.losses.mean_squared_error(self._targets, output)
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def checkInputs(self, inputs):
        assert isinstance(inputs, np.ndarray)
        
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.num_steps
        assert inputs.shape[2] == self.n_input
    
    def checkOutputs(self, outputs, batch_size):
        assert isinstance(outputs, np.ndarray)
        if self.s2s:
            assert len(outputs.shape) == 2
            assert outputs[0] == batch_size
            assert outputs[1] == self.num_steps
        else:
            assert len(outputs.shape) == 1
            assert outputs[0] == batch_size
            
        
    def update(self, inputs, outputs, sess=None):
        """
        inputs: np.array (batch_size, num_steps, n_input)
        outputs: np.array (batch_size, num_steps) or (batch_size)
        
        We need to run train_op to update the parameters
        We also need to return its loss
        """
        self.checkInputs(inputs)
        
        batch_size = inputs.shape[0]
        
        self.checkOutputs(outputs, batch_size)
        
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], 
                           {self._input_data: inputs, self._targets: outputs})
        
        return loss
    
    def predict(self, inputs, outputs = None, sess=None):
        """
        inputs: np.array (batch_size, num_steps, n_input)
        outputs: np.array (batch_size, num_steps) or (batch_size)
        
        This function would not run train_op
        outputs is only optional if we want to get the loss
        """
        self.checkInputs(inputs)
        
        if outputs != None:
            batch_size = inputs.shape[0]
            self.checkOutputs(outputs, batch_size)
        
        sess = sess or tf.get_default_session()
        
        if outputs != None:
            predicted, loss = sess.run([self.output, self.loss],
                    {self._input_data: inputs, self._targets: outputs})
            return (predicted, loss)
        else:
            predicted = sess.run(self.output,
                    {self._input_data: inputs})
            return predicted