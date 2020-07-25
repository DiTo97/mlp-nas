"""
Main parameters used througout the project.
TODO: Move everything to argparse in run.py to make them volatile.
"""

########################################################
#                   NAS parameters                     #
########################################################
# How many epochs to run the controller for
CONTROLLER_SAMPLING_EPOCHS = 10
# How many architectures to sample at each epoch
SAMPLES_PER_CONTROLLER_EPOCH = 10
# How many epochs to train the controller for
# (<= than CONTROLLER_SAMPLING_EPOCHS)
CONTROLLER_TRAINING_EPOCHS = 10
# How many epochs to train each sampled architecture for
ARCHITECTURE_TRAINING_EPOCHS = 10
# Controller's discount factor base
CONTROLLER_LOSS_ALPHA = 0.9

########################################################
#               Controller parameters                  #
########################################################
# LSTM's hidden size
CONTROLLER_LSTM_DIM = 128
# LSTM optimizer of choice
CONTROLLER_OPTIMIZER = 'Adam'
# Controller's optimizer momentum
CONTROLLER_MOMENTUM = 0.0
# Controller's learning rate
CONTROLLER_LEARNING_RATE = 0.01
# Controller's learning rate decay
CONTROLLER_DECAY = 0.1
# Whether to train the LSTM against an adversarial accuracy predictor
CONTROLLER_USE_PREDICTOR = True

########################################################
#                   MLP parameters                     #
########################################################
# Maximum length of architectures' sequences
MAX_ARCHITECTURE_LENGTH = 10
# MLPs' optimizer of choice
# (Internal 10-epoch-long training)
MLP_OPTIMIZER = 'Adam'
# MLPs' learning rate
MLP_LEARNING_RATE = 0.001
# MLPs' learning rate decay
MLP_DECAY = 0.0
# MLPs' optimizer momentum
MLP_MOMENTUM = 0.0
# MLP's dropout layers probability
MLP_DROPOUT = 0.2
# MLPs' loss function
# (binary_crossentropy if 2 classes detected)
MLP_LOSS_FUNCTION = 'categorical_crossentropy'
# Whether to apply one-shot learning when generating MLPs
MLP_ONE_SHOT = True

########################################################
#                Search space definition               #
########################################################
# Possible number of nodes per-layer
NODES = [8, 16, 32, 64, 128, 256, 512]
# Possible activation functions per-layer
ACT_FUNCS = ['sigmoid', 'tanh', 'relu', 'elu']

########################################################
#                 Output folders path                  #
########################################################
# Directort where to store best MLP models
BEST_MODEL_PATH = './MODELS'
# Directory where to store MLPNAS barplots
BARPLOTS_PATH = './BARPLOTS'
