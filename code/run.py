"""
Main entry poin to the project in order to train a MLPNAS sampling architectures tailored to a classification problem (either binary or multi-class doesn't make a difference) on a given dataset. Refer to the arguments at the bottom to select a different one:

python run.py --dataset=<dataset_name>

Refer also to constants.py for the main hyperparameters of the model.
"""

import pandas as pd
import argparse

from utils import get_top_n_architectures, get_accuracy_distribution
from datautils import load_dataset

# MLPNAS object class
from mlpnas import MLPNAS

# Pre-made MLP classifier to compare against
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def main(args):
    Xtr, ytr, Xte, yte, num_classes = load_dataset(args.dataset)

    nas = MLPNAS(Xtr, ytr, num_classes,
                task_name=args.dataset)

    # Traverse MLPNAS's search space
    # and sample plausible architectures
    data = nas.search()
    get_top_n_architectures(args.n, num_classes) # Display Top n architectures

    # Extract best architecture
    best_model = nas.extract_best_model(data)

    # Fine-tune the best architecture on the training set
    # Xtr for another 50 epochs (unless it stops early)
    #
    # Assign the return to a variable which will
    # hold training's history in case you need to
    #
    # i.e.: best_model_history
    _ = nas.finetune_model(
        best_model, Xtr, ytr, validation_split=0.2, 
        batch_size=64, shuffle=True, epochs=50, save=args.save
    )

    # from utils import plot_history
    # plot_history(best_model_history)

    tloss, tacc = MLPNAS.score(best_model, Xte, yte)
    print(f"NAS-made MLP test loss: {tloss}")
    print(f"NAS-made MLP test accuracy: {tacc}")

    # Test against sklearn MLP classifier
    mlp = MLPClassifier(activation="relu",
                        solver="adam", alpha=0.01).fit(Xtr, ytr)

    print(f"\nScikit-learn MLP test accuracy: {mlp.score(Xte, yte)}")

    # Validation accuracies barplot
    get_accuracy_distribution(args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Move hyperparameters from constants.py to here
    parser.add_argument('--save', default=True, action='store_true',
                         help='Whether to save the best architecture sampled by MLPNAS')
    parser.add_argument('--no-save', dest='save', action='store_false')

    parser.add_argument('--dataset', type=str, default='mnist',
                         choices=('mnist', 'crimes', '10speakers', 'wine'),
                         help='Name of the dataset to run MLPNAS against')

    parser.add_argument('--n', type=int, default=5,
                         help='How many architectures to show after MLPNAS sampling')

    # Parse arguments and run
    args = parser.parse_args()
    main(args)