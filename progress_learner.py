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

class EventProgressEstimator(object):
    """
    Estimate the progress of event using LSTM
    """
    def __init__(self, is_training, name=None, config = Config()):
        self.config = config
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

            # Sample's weights, NOT network's weights
            self._weights = tf.placeholder(tf.float32, [None])
                
            
            if is_training:
                self.lr = tf.Variable(0.0, trainable=False)
            
            lstm_cell = BasicLSTMCell(size, forget_bias = 1.0, state_is_tuple=True)
            
            if is_training and config.keep_prob < 1:
                lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
                
            multi_lstm_cell = MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
            
            # multi_lstm_cell.state_size = config.num_layers * 2 * size
            # ( batch_size x cell.state_size )
            if is_training:
                self.initial_state = multi_lstm_cell.zero_state(config.batch_size, tf.float32)
            else:
                self.initial_state = multi_lstm_cell.zero_state(config.batch_size, tf.float32)
                
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
            
            print ("self.inputs.shape = %s " % str(inputs.shape) + " after linear layer")
            
            
                         
            # (output, state)
            # output is of size:  ( batch_size, num_steps, size )
            # state is of size:   ( batch_size, cell.state_size ) (last state only)
            with tf.variable_scope("lstm"):
                output_and_state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs, dtype=tf.float32,
                                                     initial_state = self.initial_state)
            
            self.final_state = output_and_state[1]
            
            if self.s2s:
                # ( batch_size, num_steps, size )
                output = output_and_state[0]
            else:
                # ( num_steps, batch_size, size )
                output = tf.transpose(output_and_state[0], [1, 0, 2])
                
                # ( batch_size, size )
                output = tf.gather(output, int(output.get_shape()[0]) - 1)
                
            
            print ("output.shape = %s after LSTM" % str(output.shape))
            
            # we will pass the hidden state to next run of lstm
            self._final_state = output_and_state[1]
            
            with tf.variable_scope("output_linear"):
                weight = tf.get_variable("weight", [size, 1])
                bias = tf.get_variable("bias", [1])

                
                if self.s2s:
                    # Need to reshape to 2 dimensions
                    output = tf.reshape(output, [-1, size])
                    output = tf.matmul(output, weight) + bias
                    # ( batch_size, num_steps )  
                    output = tf.reshape(output, [-1, num_steps])
                else:
                    #( batch_size, 1)
                    # @ is the same as matmul
                    output = tf.matmul(output, weight) + bias
                    
            # Remove all 1 dimension and squash the function down to [0..1]
            # ( batch_size, num_steps ) or (batch_size)
            self.output = tf.sigmoid(tf.squeeze(output))
            
            print ("self.output.shape = %s after linear" % str(self.output.shape))
            
            print ("self._targets.shape = %s " % str(self._targets.shape))
            
            # Loss = mean squared error of target and predictions
            self.loss = tf.losses.mean_squared_error(self._targets, self.output, self._weights)
            
            if is_training:
                # 
                # optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
                
                # self.train_op = optimizer.minimize(self.loss)

                if self.config.optimizer == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                    tvars = tf.trainable_variables()
                    self.train_op = []
                        
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                                      self.config.max_grad_norm)
                    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

                elif self.config.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                    
                    self.train_op = optimizer.minimize(self.loss)

                elif self.config.optimizer == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
                    
                    self.train_op = optimizer.minimize(self.loss)
                
    
    def checkInputs(self, inputs):
        assert isinstance(inputs, np.ndarray)
        
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.num_steps
        assert inputs.shape[2] == self.n_input
    
    def checkOutputs(self, outputs, batch_size):
        assert isinstance(outputs, np.ndarray)
        if self.s2s:
            assert len(outputs.shape) == 2
            assert outputs.shape[0] == batch_size
            assert outputs.shape[1] == self.num_steps
        else:
            assert len(outputs.shape) == 1
            assert outputs.shape[0] == batch_size
        
    def update(self, inputs, outputs, weights = None, initial_state = None, sess=None):
        """
        inputs: np.array (batch_size, num_steps, n_input)
        outputs: np.array (batch_size, num_steps) or (batch_size)
        weights: (Optional) weight of samples np.array (batch_size)
        
        We need to run train_op to update the parameters
        We also need to return its loss
        """
        self.checkInputs(inputs)
        
        batch_size = inputs.shape[0]
        
        self.checkOutputs(outputs, batch_size)
        
        sess = sess or tf.get_default_session()
        
        feed_dict = {self._input_data: inputs, self._targets: outputs}
        
        if not initial_state is None:
            feed_dict[self.initial_state] = initial_state

        if weights is None:
            weights = np.ones(batch_size, dtype=np.float32)
            
        feed_dict[self._weights] = weights
        
        _, loss, state = sess.run([self.train_op, self.loss, self.final_state], 
                           feed_dict)
        
        return loss, state
    
    def predict(self, inputs, outputs = None, weights = None, sess=None):
        """
        inputs: np.array (batch_size, num_steps, n_input)
        outputs: np.array (batch_size, num_steps) or (batch_size)
        
        This function would not run train_op
        outputs is only optional if we want to get the loss
        """
        self.checkInputs(inputs)
        
        batch_size = inputs.shape[0]

        if not outputs is None:
            self.checkOutputs(outputs, batch_size)
        
        sess = sess or tf.get_default_session()
        
        if weights is None:
            weights = np.ones(batch_size, dtype=np.float32)

        if not outputs is None:
            predicted, loss = sess.run([self.output, self.loss],
                    {self._input_data: inputs, self._targets: outputs, self._weights: weights})
            return (predicted, loss)
        else:
            predicted = sess.run(self.output,
                    {self._input_data: inputs})
            return predicted
        
    def assign_lr(self, lr_value, sess=None):
        sess = sess or tf.get_default_session()
        
        sess.run(tf.assign(self.lr, lr_value))
        
    def get_state(self, sess=None):
        """
        This basically gives the state of the cell
        """
        sess = sess or tf.get_default_session()

# LSTM cell states        
# state = None

def none_info ():
    while True:
        yield None

def run_epoch(m, data, lbl, info = none_info(), verbose=False, training = True):
    state = None
    costs = 0
    cost_iters = 0
    
    for step, (x, y, z) in enumerate( zip(data, lbl, info) ):
        y_prime = y
        if not m.config.s2s:
            # Just use the last label
            y_prime = y[:,-1]
        
        if training:
            cost, state = m.update(x, y_prime, weights = z, initial_state = state)
        else:
            predicted, cost = m.predict(x, y_prime, weights = z)
            
            if verbose:
                print ('Predicted = ' +str(predicted))
                print ('Labels = ' +str(y_prime))
                print ('Infos = ' +str(z))
        
        costs += cost
        cost_iters += 1
        
    print("costs %.3f, cost_iters %d, cost %.3f" % 
          (costs, cost_iters, costs / cost_iters))

    return costs / cost_iters
    
if __name__ == "__main__":
    p = Project.load("slideaway_project.proj")
    
    config = Raw_Config()
    
    np.random.seed()

    print ('p.training_data.shape = ' + str(p.training_data.shape))
    print ('p.validation_data.shape = ' + str(p.validation_data.shape))
    print ('p.testing_data.shape = ' + str(p.testing_data.shape))

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model") as scope:
            print('-------- Setup m model ---------')
            m = EventProgressEstimator(is_training=True, name = p.name, config = config)
        
        with tf.variable_scope("model", reuse = True) as scope:    
            print('-------- Setup mtest model ---------')
            mtest = EventProgressEstimator(is_training=False, name = p.name, config = config)
        
        session.run(tf.global_variables_initializer())
        
        """
        Training first
        """
        train_losses = []
        validate_losses = []

        for i in range(config.max_max_epoch):
            print('-------------------------------')
            start_time = time.time()
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
            
            indices = np.arange(p.training_data.shape[0])

            if config.epoch_shuffle:
                np.random.shuffle(indices)

            train_loss = run_epoch(m, p.training_data[indices], p.training_lbl[indices], training = True)
        
            "Validating"
            # [:,:,:,:8]
            validate_loss = run_epoch(mtest, p.validation_data, p.validation_lbl, training = False)

            train_losses.append(train_loss)
            validate_losses.append(validate_loss)

        print (repr(train_losses))
        print (repr(validate_losses))

        print ('------- TEST -------')
        run_epoch(mtest, p.testing_data, p.testing_lbl, training = False, verbose = True)
