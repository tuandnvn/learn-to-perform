import pickle
import tensorflow as tf
from rl import action_learner, value_estimator, block_movement_env

from progress_learner import EventProgressEstimator
import config
import project
# Need to add this import to load class
from project import Project

if __name__ == '__main__':
    p = project.Project.load("slidearound_hopstep_1.proj")

    c = config.Config()
    c.num_episodes = 20

    tf.reset_default_graph()

    params = []
    for policy_learning_rate in [0.001, 0.003, 0.01]:
        for policy_decay in [0.95, 0.96, 0.97]:
            for value_learning_rate in [0.001, 0.003, 0.01]:
                for value_decay in [0.95, 0.96, 0.97]:
                    for breadth in range(2,5):
                        params.append((policy_learning_rate, policy_decay, 
                            value_learning_rate, value_decay, breadth))

    for policy_learning_rate, policy_decay, value_learning_rate, value_decay, breadth in params:
        print ('==================================================')
        print (param)
        c.policy_learning_rate = policy_learning_rate
        c.policy_decay = policy_decay
        c.value_learning_rate = value_learning_rate
        c,value_decay = value_decay

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
            saver.restore(sess, 'progress.mod')
            
            action_ln = action_learner.ActionLearner(c, p, progress_estimator, 
                                           policy_est, value_est, session = sess)
            action_policy = action_learner.random_action

            _, stats = action_ln.reinforce(action_policy, breadth = breadth, verbose = False)

            stat_file = 'session.data._%.4f_%.4f_%.4f_%.4f' % (policy_learning_rate, policy_decay, 
                            value_learning_rate, value_decay)
            with open(stat_file, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

            print('----Done saving stats data ---')
