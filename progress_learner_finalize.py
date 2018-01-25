import numpy as np
import tensorflow as tf
import time

from project import Project, Multi_Project
from progress_learner import run_epoch, EventProgressEstimator
from utils import TRAINING, VALIDATING, TESTING
from config import Config

class Final_Config(Config):
	max_epoch = 10
	max_max_epoch = 50

if __name__ == "__main__":
    def learn_all_simple(project_name):
        print ('==============================================')
        print ('Learn progress model for ' + project_name)
        file_name = project_name.lower() + "_project.proj"

        p = Project.load(file_name)
        
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
            saver.save(session, "progress_" + project_name + ".mod")

            print('-------- Saved progress.mod ---------')

    for project_name in ["SlideToward", "SlideAway", "SlideNext", "SlidePast", "SlideAround"]:
        learn_all_simple(project_name)


    def learn_all_negative_samples():
        multi_p = Multi_Project.load("all_actions.proj")
    
        config = Config()
        
        np.random.seed()

        with tf.Graph().as_default(), tf.Session() as session:
            for p in multi_p:
                if p.name != 'SlideAround':
                    continue

                print ('-------------TRAINING FOR ' + p.name + ' -----------------')
                project_data = multi_p.data[p.name]

                total_data = np.concatenate( [project_data[TRAINING][0], project_data[VALIDATING][0], project_data[TESTING][0]], axis = 0 )
                total_lbl = np.concatenate( [project_data[TRAINING][1], project_data[VALIDATING][1], project_data[TESTING][1]], axis = 0 )
                total_info = np.concatenate( [project_data[TRAINING][2], project_data[VALIDATING][2], project_data[TESTING][2]], axis = 0 )

                s = "model_" + p.name
                with tf.variable_scope(s) as scope:
                    print('-------- Setup m model ---------')
                    m = EventProgressEstimator(is_training=True, name = p.name, config = config)
                
                with tf.variable_scope(s, reuse = True) as scope:    
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
                    
                    indices = np.arange(total_data.shape[0])

                    if config.epoch_shuffle:
                        np.random.shuffle(indices)

                    train_loss = run_epoch(m, total_data[indices], total_lbl[indices], info = total_info[indices], training = True)
                
            
            saver = tf.train.Saver()
            saver.save(session, "progress_all.mod")

            print('-------- Saved progress.mod ---------')

    # learn_all_negative_samples()