"""
Controller base class that allows to define and train a simple LSTM or an hybrid one with the accuracy predictor.
"""

import os
import numpy as np

from keras import optimizers
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM
from keras.models import Model
from keras.engine.input_layer import Input

from mlp_generator import MLPSearchSpace
from constants import MAX_ARCHITECTURE_LENGTH, CONTROLLER_LSTM_DIM, CONTROLLER_OPTIMIZER, CONTROLLER_LEARNING_RATE, CONTROLLER_DECAY, CONTROLLER_MOMENTUM, CONTROLLER_USE_PREDICTOR

class Controller(MLPSearchSpace):

    def __init__(self, num_classes):
        self.max_len = MAX_ARCHITECTURE_LENGTH

        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR

        self.controller_weights = 'LOGS/controller_weights.h5'
        self.seq_data = [] # Flattened NAS data across epochs

        super().__init__(num_classes)

        self.controller_classes = len(self.vocab) + 1 # +1 to account for the <pad> token

    def sample_architecture_sequences(self, model, number_of_samples):
        # Define values needed for sampling 
        final_layer_id = len(self.vocab) # <end> token := sigmoid or softmax
        dropout_id = final_layer_id - 1
        vocab_idx = [0] + list(self.vocab.keys()) # Put <pad> (0), up-front

        # Sampled architectures per-epoch
        samples = []
        print("Generating architecture samples...")
        print('------------------------------------------------------')

        # Sample up to the required number of samples
        while len(samples) < number_of_samples:
            # Sampled layers for current architectures
            seed = []

            while len(seed) < self.max_len:
                # Pad sequence up to self.max_len
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)

                # Given the previous elements, get probability distribution for the next element
                # Note: Predicting action distribution
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                
                # Extract the probability distribuition
                probab = probab[0][0] # 3 levels of nested lists

                # Sample the next element/action randomly given the distribution
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]
                
                # No dropout layer as input layer
                if next == dropout_id and len(seed) == 0:
                    continue
                # No final layer as input layer
                if next == final_layer_id and len(seed) == 0:
                    continue

                # Terminate sampling as soon as the final layer is sampled
                # (Allows to have variable length architectures, <pad> excluded)
                if next == final_layer_id:
                    seed.append(next)
                    break

                # If the architecture has reached the maximum length (- 1),
                # add the final layer regardless of what has been predicted
                # TODO: Count how many times it stops due to this as opposed to predicting the final layer
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break

                # Ignore <pad> (0) element
                if not next == 0:
                    seed.append(next)

            # Check if the generated sequence has been generated before.
            # If not, add it to NAS' sequences data.
            # TODO: Avoid potentially infinite loop if no new sequences are sampled.
            if seed not in self.seq_data:
                samples.append(seed)
                self.seq_data.append(seed)

        return samples

    def controller_model(self, controller_input_shape, controller_batch_size):
        """
        Generate a simple LSTM controller.
        """ 
        # Cannot pass both shape and shape.
        # Choose accordingly if you want to train in batches.
        if controller_batch_size > 0:
            main_input = Input(shape=controller_input_shape, name='main_input')
        else:
            main_input = Input(batch_shape=(controller_batch_size, *controller_input_shape), name='main_input')

        # LSTM layer that processes input
        X = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)

        # Output single dense layer with softmax from LSTM's output
        main_output = Dense(self.controller_classes, activation='softmax',
                            name='main_output')(X)

        # Return the model with correct sizes
        return Model(inputs=[main_input], outputs=[main_output])

    def train_controller_model(self, model, X_data, y_data, loss_func,
                               controller_batch_size, nb_epochs):
        """
        Train a simple LSTM controller model.

        Parameters
        ----------
        X_data : list
            Sampled architectures' NAS data from previous epoch

        y_data : list
            Last layer for each sample in NAS data

        loss_func : callable
            REINFORCE policy gradient
        """
        if self.controller_optimizer == 'SGD':
            optim = optimizers.SGD(lr=self.controller_lr, decay=self.controller_decay,
                                   momentum=self.controller_momentum, clipnorm=1.0)
        else:
            # Note: Pass the optimizer name as its Keras class, if not SGD.
            optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr,
                            decay=self.controller_decay, clipnorm=1.0)

        # Compile the model and load weights, if any.
        model.compile(optimizer=optim, loss={'main_output': loss_func})
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)

        print("Training controller...")
        model.fit({'main_input': X_data},                                                      # Set named input
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},    # Set named output
                  epochs=nb_epochs, batch_size=controller_batch_size, verbose=0)

        # Save weights for later
        model.save_weights(self.controller_weights)

    def hybrid_controller_model(self, controller_input_shape, controller_batch_size):
        """
        Generate an hybrid LSTM controller with accuracy predictor.
        """ 
        if controller_batch_size > 0:
            main_input = Input(shape=controller_input_shape, name='main_input')
        else:
            main_input = Input(batch_shape=(controller_batch_size, *controller_input_shape), name='main_input')

        # LSTM layer that processes input
        X = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)

        # Predictor single dense layer with sigmoid from LSTM's output
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(X)
        # Output single dense layer with softmax from LSTM's output
        main_output = Dense(self.controller_classes, activation='softmax',
                            name='main_output')(X)

        # Return the model with multiple outputs
        return Model(inputs=[main_input], outputs=[main_output, predictor_output])

    def train_hybrid_model(self, model, x_data, y_data, pred_target, loss_func,
                           controller_batch_size, nb_epochs):
        """
        Train a hybrid LSTM controller with accuracy predictor.

        Parameters
        ----------
        pred_target : list
            Validation accuracies from sampled architectures.
        """
        if self.controller_optimizer == 'SGD':
            optim = optimizers.SGD(lr=self.controller_lr, decay=self.controller_decay,
                                   momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr,
                            decay=self.controller_decay, clipnorm=1.0)

        # Compile the model and load weights, if any.
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1}) # Multi-loss that takes both into account
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)

        print("Training controller...")
        model.fit({'main_input': x_data},                                                      # Set named input
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),     # Set named main output
                   'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)}, # Set named predictor output
                  epochs=nb_epochs, batch_size=controller_batch_size, verbose=0)

        # Save weights for later
        model.save_weights(self.controller_weights)

    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        """
        Gather predictor accuracies on sampled sequences for the hybrid LSTM.
        """
        pred_accuracies = []

        for seq in seqs:
            # Padd sequences and discard last entry ([:, :-1])
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            Xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)

            # Collect predictor accuracies
            (_, pred_accuracy) = [X[0][0] for X in model.predict(Xc)]
            pred_accuracies.append(pred_accuracy[0]) # 3 levels of nested lists

        return pred_accuracies
