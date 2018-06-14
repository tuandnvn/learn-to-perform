import os
import numpy as np
import tensorflow as tf
import time
import argparse

from project import Project, Multi_Project
from progress_learner import run_epoch, EventProgressEstimator
from utils import TRAINING, VALIDATING, TESTING
from config import Config, Qual_Config, Quan_Config

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save a learned model from combining TRAIN/VALIDATE/TEST data.')

    parser.add_argument('-a', '--action', action='store', metavar = ('ACTION'),
                                help = "Action type. Choose from 'SlideToward', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideAround'" )
    parser.add_argument('-t', '--type', action='store', metavar = ('TYPE'), default='QUAL',
                                help = "Choose one of the followings: QUAL (qualitative), QUAN (quantitative). Default is QUAL" )
    parser.add_argument('-p', '--project', action='store', metavar = ('PROJECT'),
                                help = "Location of project file." )
    parser.add_argument('-e', '--epoch', action='store', metavar = ('EPOCH'), default=50,type=int,
                                help = "Number of epochs." )
    parser.add_argument('-s', '--save', action='store', metavar = ('SAVE'),
                                help = "Where to save the progress model. Default is 'learned_models/progress_' + action + '.mod'" )

    args = parser.parse_args()
    
    project_name = args.action

    feature_type = args.type
    if feature_type == 'QUAL':
        config = Qual_Config()
    elif feature_type == 'QUAN':
        config = Quan_Config()

    config.max_epoch = 10
    config.max_max_epoch = args.epoch

    project_file = args.project

    if project_file is None:
        if feature_type == 'QUAL':
            project_file = os.path.join('learned_models', project_name.lower() + "_project.proj")
        elif feature_type == 'QUAN':
            project_file = os.path.join('learned_models', project_name.lower() + "_raw.proj")

    progress_path = args.save
    if progress_path is None:
        progress_path = os.path.join('learned_models', "progress_" + project_name + ".mod")

    print (' Create progress model by loading project file from %s, running %d epochs, and saving into %s ' % (project_file, config.max_max_epoch, progress_path) )
    
    p = Project.load(project_file)

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
        saver.save(session, progress_path)

        print('-------- Saved progress file to ' + progress_path)