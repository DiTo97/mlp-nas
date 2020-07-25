"""
Utility functions called throughout the MLPNAS code and in run.py.
"""

import os
import shutil
import pickle
import numpy as np

from keras import models
from itertools import groupby
from matplotlib import pyplot as plt
import time

from constants import BEST_MODEL_PATH, BARPLOTS_PATH
from mlp_generator import MLPSearchSpace

########################################################
#                    Model loading                     #
########################################################
def load_best_model(filepath):
    """
    Load best model from file after NAS search.
    """
    return models.load_model(os.path.join(BEST_MODEL_PATH, filepath))

########################################################
#                   Data processing                    #
########################################################
def unison_shuffled_copies(a, b):
    """
    Generate random permutations of a and b. Utilized to shuffle the input X (and its corresponding target y) after each controller epoch to introduce a degree of variance within the sampling episodes of architectures.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

########################################################
#                       Logging                        #
########################################################
def clean_log():
    filelist = os.listdir('LOGS')

    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))

def get_latest_event_id():
    """
    NAS searches are independent from one another as the model always cold starts from scratch. Within the LOGS folder there are subfolders denoted as exploration events, each of which collects the information from a single NAS search.
    """
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS')
                               if os.path.isdir('LOGS/' + d)]

    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))

########################################################
#                 Results processing                   #
########################################################
def load_nas_data():
    """
    Grab NAS data from the most recent event.
    """
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    return data

def sort_search_data(nas_data):
    """
    Sort the architectures within the NAS data by validation accuracy.
    """
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]

    nas_data = [nas_data[x] for x in sorted_idx]

    return nas_data

########################################################
#                Evaluation and plots                  #
########################################################

def get_top_n_architectures(n, num_classes):
    """
    Grab the Top n architectures from the most recent NAS data and decode them into TensorFlow layers.
    """
    data = load_nas_data()
    data = sort_search_data(data)

    # Instantiate a coherent search space
    search_space = MLPSearchSpace(num_classes)
    print('Top {} architectures:'.format(n))

    # Decode architectures into TensorFlow layers
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation accuracy:', seq_data[1])

def plot_history(history):
    """
    Plot accuracies and losses from a History object. Taken from AML labs.
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    
    # Show the figure
    plt.show()

def get_accuracy_distribution(name=None):
    """
    Show a barplot of the architectures' validation accuracies from the most recent NAS data.
    """

    data = load_nas_data()

    # Grab sorted validation accuracies
    accuracies = [x[1]*100. for x in data]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)

    # Group them and plot a barplot
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}

    plt.figure(figsize=(12, 5))
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.xlabel('Validation accuracies %')
    plt.ylabel('N. of architectures')

    # Save and show the figure
    if name is not None:
        name = name + '.jpg' # Force into JPEG format
        plt.savefig(os.path.join(BARPLOTS_PATH, name))

    plt.show()