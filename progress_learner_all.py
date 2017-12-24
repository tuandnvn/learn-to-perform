import time
import numpy as np
import tensorflow as tf
from config import Config, Raw_Config
from project import Project, Multi_Project
from generate_utils import gothrough
from progress_learner import EventProgressEstimator, run_epoch
from utils import TRAINING, VALIDATING, TESTING

if __name__ == "__main__":
    multi_p = Multi_Project.load("all_actions.proj")
    
    config = Config()
    
    np.random.seed()

    with tf.Graph().as_default(), tf.Session() as session:
        for p in multi_p:
            if p.name != 'SlideAround':
                continue

            print ('-------------TRAINING FOR ' + p.name + ' -----------------')
            project_data = multi_p.data[p.name]

            print ('p.training_data.shape = ' + str(project_data[TRAINING][0].shape))
            print ('p.validation_data.shape = ' + str(project_data[VALIDATING][0].shape))
            print ('p.testing_data.shape = ' + str(project_data[TESTING][0].shape))

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
                
                indices = np.arange(project_data[TRAINING][0].shape[0])

                if config.epoch_shuffle:
                    np.random.shuffle(indices)

                train_loss = run_epoch(m, project_data[TRAINING][0][indices], project_data[TRAINING][1][indices], info = project_data[TRAINING][2][indices], training = True)
            
                "Validating"
                # [:,:,:,:8]
                validate_loss = run_epoch(mtest, project_data[VALIDATING][0], project_data[VALIDATING][1], info = project_data[VALIDATING][2], training = False)

                train_losses.append(train_loss)
                validate_losses.append(validate_loss)

            print (repr(train_losses))
            print (repr(validate_losses))

            print ('------- TEST -------')
            run_epoch(mtest, project_data[TESTING][0], project_data[TESTING][1], info = project_data[TESTING][2], training = False, verbose = True)