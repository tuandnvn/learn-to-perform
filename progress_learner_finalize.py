import numpy as np
import tensorflow as tf
import time

from project import Project
from progress_learner import run_epoch, EventProgressEstimator

from config import Config

class Final_Config(Config):
	keep_prob = 0.6
	hidden_size = 100
	num_layers = 2
	max_epoch = 10
	max_max_epoch = 50
	lr_decay = 0.97
	learning_rate = 0.005
	max_grad_norm = 5
	epoch_shuffle = False
	optimizer = 'adam'

if __name__ == "__main__":
    p = Project.load("slidearound_hopstep_1.proj")
    
    config = Final_Config()
    
    # Merge all data
    total_data = np.concatenate( [p.training_data, p.validation_data, p.testing_data], axis = 0 )
    total_lbl = np.concatenate( [p.training_lbl, p.validation_lbl, p.testing_lbl], axis = 0 )

    print (total_data.shape)
    print (total_lbl.shape)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model") as scope:
            print('-------- Setup m model ---------')
            m = EventProgressEstimator(is_training=True, name = p.name, config = config)

        session.run(tf.global_variables_initializer())

        for i in range(config.max_max_epoch):
            print('-------------------------------')
            start_time = time.time()
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))

            train_loss = run_epoch(m, total_data, total_lbl, training = True)


        saver = tf.train.Saver()
        saver.save(session, "progress.mod")

        print('-------- Saved progress.mod ---------')