import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import SGD, Adam
from config import Config, Raw_Config
from project import Project

class EventProgressEstimatorKeras(object):
    def __init__(self, config = Config()):
        self.config = config
        self.num_steps = num_steps = config.num_steps
        self.n_input = n_input = config.n_input
        self.size = size = config.hidden_size

        self.model = Sequential()

        # Additional layers
        for layer in range(0, config.num_layers):
            # s2s for lower layers
            if layer < config.num_layers - 1:
                rs=True
            else:
                rs=False

            do = 1-config.keep_prob

            if layer == 0:
                self.model.add(LSTM(self.size, input_shape = (self.num_steps, self.n_input), 
                    dropout=do, return_sequences = rs))
            else:
                self.model.add(LSTM(self.size, 
                    dropout=do, return_sequences = rs))

        self.model.add(Dense(1, activation='sigmoid'))

        # Using Mean Square Error for loss
        # Using Mean Absolute Error for metrics
        # Notice that decay in keras is the part that got discounted
        decay = 1 - self.config.lr_decay

        if config.optimizer == 'sgd':
            optimizer = SGD(lr = self.config.learning_rate, decay = decay)

        elif config.optimizer == 'adam':
            optimizer = Adam(lr = self.config.learning_rate, decay = decay)
            # optimizer = Adam(lr = self.config.learning_rate, decay = self.config.lr_decay)
        

        self.model.compile(optimizer=optimizer, loss='mse', metrics = ['mae'])

    def update(self, inputs, outputs):
        """
        This is different from the version using tensorflow which
        explicitly updates each mini-batch
        Here we input the whole data, which means 
        both the training and testing data 
        
        Params:
        ========
        inputs: np.array (#samples, num_steps, n_input)
        outputs: np.array (#samples)
        
        We need to run train_op to update the parameters
        We also need to return its loss

        Return:
        ========
        a History object
        """

        return self.model.fit(X, y, epochs=self.config.max_max_epoch, 
            validation_split = self.config.test_percentage, verbose=2, shuffle=False,
            batch_size = self.config.train_batch_size)


    def predict(self, inputs):
        """

        Params:
        ========
        inputs: np.array (#samples, num_steps, n_input)

        Return:
        ========
        outputs: np.array (#samples)
        """
        return self.model.predict(inputs)



if __name__ == '__main__':
    p = Project.load("slidearound_p2.proj")
    
    config = Config()

    X = p.rearranged_data
    y = p.rearranged_lbls[:,-1]

    print (X.shape)
    print (y.shape)

    m = EventProgressEstimatorKeras(config = Config())

    history = m.update(X, y)

    print(history.history['loss'])
    print(history.history['val_loss'])