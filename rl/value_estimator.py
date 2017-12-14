# -*- coding: utf-8 -*-
import tensorflow as tf 

class PolicyEstimator():
    """
    Policy Function approximator.
    
    A policy estimator is a function that produce the following distribution
    π(a|s, θ)
    
    s is the current state
    a is an action that has been selected from the action distribution 
    θ is the set of parameters, in this case, these are a neural network weight
    
    Each time, we have two inputs:
        - A state generated by Monte Carlo
        - An action generated by Monte Carlo
    The model would produce (from the state) an output layer
    that correspond to a preference function h(s,a).
    
    Following is the formula for updating:
    
    θ += α*γ^t*G * DELTA_θ (log π(At|St, θ))
    
    or, more succintly, for baseline also:
    
    θ += α * TARGET_WEIGHT * DELTA_θ (log π(At|St, θ))
    
    where TARGET_WEIGHT could be the (discounted) return from time t
    adjusted with baseline.
    
    This corresponding to the following graph:
    
    The graph basically predicts the action probability, BUT the loss function
    needs to be reweighted with the TARGET_WEIGHT, so that the updating 
    follows the formula
    """
    
    def __init__(self, config, scope="policy_estimator"):
        """
        The code to declare your tensorflow graph comes here
        """

        # This state dimension would probably be 12
        # location + rotation of two most objects
        state_dimension = config.state_dimension

        # This would be 3. 2 for locations, 1 for rotation
        action_dimension =  config.action_dimension

        # sigma_dimension is simplified to 2
        # in a full model, this value would be 9
        # taking in all covariances between all variables
        # In this model, we simplify that to a diagonal matrix
        # which means we actually generate each value independently
        # Covariance matrix  = [ sigma_1, 0, 0 ]
        #                      [ 0, sigma_2, 0 ]
        #                      [ 0, 0, sigma_3 ]
        sigma_dimension = config.action_dimension

        # short for weight_regularizer_scale
        wrs = config.weight_regularizer_scale

        self.lr = tf.Variable(0.0, trainable=False)
        
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
            
            state_expanded = tf.expand_dims(self.state, 0)
            """
            mu_layer is a fully connected layer, produce location/rotation of the action
            
            activation_fn=None because we want a linear function
            """
            self.mu_layer = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=state_expanded,
                num_outputs=action_dimension,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer))
            
            """
            Using softplus so that the output would be > 0 but we also don't want 0
            """
            self.sigma_layer = 0.5 * tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=state_expanded,
                num_outputs=sigma_dimension,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_uniform_initializer(minval=1.0/(5 * state_dimension), maxval=2.0/(5 * state_dimension))))

            # Using a mvn to predict action probability
            mvn = tf.contrib.distributions.Normal(
                loc=self.mu_layer,
                scale=self.sigma_layer)

            # (action_dimension)
            self.picked_action_prob = mvn.prob(self.action) 
            
            #print (tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.regularizer_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.sigma_constraint = tf.norm(self.sigma_layer)

            # The action probability is the product of component probabilities
            # Notice that the formula for REINFORCE update is (+) gradient of log-prob function
            # so we minimize the negative log-prob function instead
            self.loss = -tf.reduce_sum(tf.log(self.picked_action_prob)) * self.target + config.constraint_sigma * self.sigma_constraint
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        """
        In prediction, just need to produce the multivariate distribution
        """
        sess = sess or tf.get_default_session()
        return sess.run([self.mu_layer, self.sigma_layer], {self.state: state})
    
    def update(self, state, target, action, sess=None):
        """
        state: input state
        target: return from time t of episode * discount factor
        action: input action
        
        We need to run train_op to update the parameters
        We also need to return its loss
        """
        sess = sess or tf.get_default_session()
        _, loss, regularizer_loss = sess.run([self.train_op, self.loss, self.regularizer_loss], {self.state: state, self.action: action, self.target: target})
        
        return loss, regularizer_loss

    def assign_lr(self, lr_value, sess=None):
        sess = sess or tf.get_default_session()
        
        sess.run(tf.assign(self.lr, lr_value))


class ValueEstimator():
    """
    Value Function approximator.
    
    This is to calculate the baseline, otherwise Policy Estimator is enough
    
    We need another set of parameter w for state-value approximator.
    
    Target is (discounted) return
    
    Just use a very simple linear fully connected layer between state and output
    """
    
    def __init__(self, config, scope="value_estimator"):
        # This state dimension would probably be 12
        # location + rotation of two most objects
        state_dimension = config.state_dimension

        self.lr = tf.Variable(0.0, trainable=False)

        with tf.variable_scope(scope): 
            # No batch
            self.state = tf.placeholder(shape=[state_dimension], name="state", dtype = tf.float32)
            
            "Placeholder for target"
            self.target = tf.placeholder(name="target", dtype = tf.float32)
            
            """
            Using a tanh output activation, because it predicts a value between -1 and 1
            """
            self.output_layer = tf.contrib.layers.fully_connected(
                tf.expand_dims(self.state,0),
                1,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.zeros_initializer)
            
            self.value = tf.squeeze(self.output_layer)
            
            self.loss = tf.squared_difference(self.value, self.target) 
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        """
        """
        sess = sess or tf.get_default_session()
        return sess.run(self.value, {self.state: state})

    def update(self, state, target, sess=None):
        """
        """
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.target: target})
        
        return loss

    def assign_lr(self, lr_value, sess=None):
        sess = sess or tf.get_default_session()
        
        sess.run(tf.assign(self.lr, lr_value))