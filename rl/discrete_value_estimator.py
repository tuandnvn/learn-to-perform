from value_estimator import PolicyEstimator, ValueEstimator

class DiscretePolicyEstimator ( PolicyEstimator ):
	"""
	This is a simple version in which
	there would be no rotation.
	
	Let's the state to be a discretized version of the original state
	
	The prediction value would be converted to discretized values
	"""
	def __init__(self, config, scope="policy_estimator"):
		# This state dimension would be 6
		# Current discretized transform of moving object + Discretized form of velocity vector
		# For example, state of a block just north of the static block, and straightly align
		# with the static block would be (0, 1, 0)
		# straightly south, and rotate 45 degree would be (0, -1, 2)
        state_dimension = config.state_dimension

        # Moving-to discretized transform of moving object 
        action_dimension =  config.action_dimension

        self.lr = tf.Variable(0.0, trainable=False)

        self.sigma_layer = tf.Variable([1,1,1], dtype = tf.float32, trainable=False)

        with tf.variable_scope(scope): 
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
            self.action = tf.placeholder(shape=[action_dimension], name="action", dtype = tf.float32)
            
            "Placeholder for target"
            self.target = tf.placeholder(name="target", dtype = tf.float32)
            
            """
            mu_layer is a fully connected layer, produce location/rotation of the action
            
            activation_fn=None because we want a linear function

            Currently the whole problem is that a linear function might not be helpful to learn 
            this kind of problem
            """
            hidden_layer = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=hidden_size,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_uniform_initializer(minval=-2.0, maxval=2.0)))

            self.mu_layer = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(hidden_layer, 0),
                num_outputs=action_dimension,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)))

            # Using a mvn to predict action probability
            mvn = tf.contrib.distributions.Normal(
                loc=self.mu_layer,
                scale=self.sigma_layer)

            # (action_dimension)
            self.picked_action_prob = mvn.prob(self.action)

            # The action probability is the product of component probabilities
            # Notice that the formula for REINFORCE update is (+) gradient of log-prob function
            # so we minimize the negative log-prob function instead
            self.loss = -tf.reduce_sum(tf.log(self.picked_action_prob)) * self.target
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())