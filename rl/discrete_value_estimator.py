import tensorflow as tf
from .value_estimator import PolicyEstimator

class DiscretePolicyEstimator ( PolicyEstimator ):
    """
    This is a simple version in which
    there would be no rotation.
    
    Let's the state to be a discretized version of the original state
    
    The prediction value would be converted to discretized values
    """
    def __init__(self, config, scope="policy_estimator", reuse = False): 
        # This state dimension would be of size 162
        state_dimension = config.state_dimension

        # Action dimension would be of size 9
        action_dimension =  config.action_dimension

        self.lr = tf.Variable(0.0, trainable=False)

        with tf.variable_scope(scope, reuse): 
            "Declare all placeholders"
            "Placeholder for input"
            """
            Now in this case, if the state is represented as a number ranking from
            start point of the map to the end, we lost the locality between
            cells of two consecutive rows.
            So let's make it a row and column.
            """
            # No batch
            self.state = tf.placeholder(shape=[state_dimension], name="state", dtype = tf.float32)
            
            "Placeholder for Monte Carlo action"
            self.action = tf.placeholder(shape=[1], name="action", dtype = tf.float32)
            
            "Placeholder for target"
            self.target = tf.placeholder(name="target", dtype = tf.float32)
            
            """
            mu_layer is a fully connected layer, produce location/rotation of the action
            
            activation_fn=None because we want a linear function

            Currently the whole problem is that a linear function might not be helpful to learn 
            this kind of problem
            """
            self.output_layer = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=action_dimension,
                activation_fn=None))

            # The action probability is the product of component probabilities
            # Notice that the formula for REINFORCE update is (+) gradient of log-prob function
            # so we minimize the negative log-prob function instead
            self.loss = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_layer, labels = self.action) * self.target
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    def update(self, state, target, action, sess=None):
        """
        state: input state
        target: return from time t of episode * discount factor
        action: input action
        
        We need to run train_op to update the parameters
        We also need to return its loss
        """
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action, self.target: target})
        
        return loss

    def predict(self, state, sess=None):
        """
        In prediction, just need to produce the multivariate distribution
        """
        sess = sess or tf.get_default_session()
        return sess.run([self.output_layer], {self.state: state})