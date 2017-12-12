import tensorflow as tf
from rl import action_learner, value_estimator, block_movement_env

from progress_learner import EventProgressEstimator
import config
import project
# Need to add this import to load class
from project import Project

from importlib import reload

if __name__ == '__main__':
    p = project.Project.load("../slidearound_hopstep_1.proj")

    c = config.Config()

    tf.reset_default_graph()

	global_step = tf.Variable(0, name="global_step", trainable=False)

	with tf.Session() as sess:
	    policy_est = value_estimator.PolicyEstimator(c)
	    value_est = value_estimator.ValueEstimator(c)
	    
	    sess.run(tf.global_variables_initializer())
	    
	    with tf.variable_scope("model") as scope:
	        print('-------- Load progress model ---------')
	        progress_estimator = EventProgressEstimator(is_training=False, name = p.name, config = c)  
	    
	    # Print out all variables that would be restored
	    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
	        print (variable.name)

	    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
	    saver.restore(sess, '../progress.mod')
	    
	    action_ln = action_learner.ActionLearner(c, p, progress_estimator, 
	                                   policy_est, value_est)
	    
	    stats = action_ln.reinforce()

	    with open('', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print('----Done saving stats data ---')