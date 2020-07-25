"""
Main utility functions to load a dataset of choice from those available to us within the DATASETS folder, from different formats, either CSV or GZip, comprising preprocessing, train/test split and counting of the number of target classes (binary or multi-class).
"""

import pandas as pd
import numpy as np

import preproc

from sklearn.model_selection import train_test_split

def load_dataset(name):
    """
    Load a dataset by passing its corresponding name:
        - mnist
            MNIST datasets for digits' recognition where each feature corresponds to the value in [0, 255] of a pixel from their 28x28 images, as we avoided the use of images as inputs to our MLPs, since as we know CNNs would be much better suited to do that.

        - crimes
            LA crimes dataset since 1934 with 39 classes.

        - 10speakers
            Audio features dataset from 10 speakers for speaker identification.

        - wine
            Wine quality dataset for quality assessment.
            (Note: This is more of dummy dataset, due to its size, to verify the NAS model was working.)
    """

    if name == 'mnist':
        return prepare_dataset(
            'DATASETS/mnist.csv',
             labels_colname="label"
        )   
    elif name == 'crimes':
        return prepare_dataset(
            'DATASETS/crimes.csv', 
             _format="csv", 
             labels_colname="Category", 
             label_type="string", 
             preprocess_func=preproc.crimes_preproc,
             parse_dates=['Dates']
        )
    elif name == '10speakers':
        return prepare_dataset(
            'DATASETS/10speakers.gzip',
             _format="gzip",
             labels_colname='class',
             label_type="int")
    elif name == 'wine':
        return prepare_dataset(
            'DATASETS/wine.csv',
             labels_colname='quality_label',
             label_type="string")
    
    raise ValueError("Dataset '%s' not available" % name)

def read_dataset(file_path: str, _format: str = 'csv',
                 labels_colname='label', label_type="int",
                 preprocess_func=None, **kwargs):
    """
    Read a dataset from file, CSV or GZip, and return it as a NumPy matrix.
    
    Returns
    -------
    X : np.ndarray
        Input training set
        
    y : np.ndarray
        Ground-truth target labels
        
    num_classes : int
        Number of different classes in the target
    """

    def multi_class(_class):
        return _class - 1

    dates_cols = kwargs.get("parse_dates")

    df = pd.DataFrame()

    if _format == 'gzip':
        df = pd.read_pickle(file_path, compression=_format)
    else:
        df = pd.read_csv(file_path, parse_dates=(
            dates_cols if dates_cols is not None else False))

    if preprocess_func is not None:
        df = preprocess_func(df)

    labels = []

    if label_type == "int":
        labels = pd.get_dummies(
            [multi_class(int(l)) for l in df[labels_colname].to_list()],
            dtype='int32'
        ).to_numpy()
    else:
        labels = pd.get_dummies(df[labels_colname]).to_numpy()

    data = df.drop([labels_colname], axis=1)

    return (data.to_numpy(), labels, len(np.unique(
                                     labels, axis=0)))

def multi_class_dataset(file_path: str, _format: str = 'csv',
                        labels_colname='label', label_type="int",
                        preprocess_func=None, **kwargs):

    return read_dataset(file_path, _format, labels_colname,
                        label_type, preprocess_func, **kwargs)

def bin_class_dataset(file_path: str, _format: str = 'csv', target_class=1,
                      labels_colname='label', label_type="int",
                      preprocess_func=None, **kwargs):

    def bin_class(x, target):
        return 1 if x == target else 0

    X, y, num_classes = read_dataset(file_path, _format, labels_colname,
                        label_type, preprocess_func, **kwargs)
    # Binarize the target as either 0s or 1s
    y = np.array([bin_class(l, target_class) for l in y.tolist()])

    return (X, y, num_classes)

def prepare_dataset(filepath,_format="csv", labels_colname='label',
                   label_type="int", is_binary_task=False, target=1,
                   preprocess_func=None, test_size=0.2, **kwargs):

    if _format not in {"csv", "gzip"}:
        raise ValueError("Wrong format: Please, use CSV or GZip.")

    if label_type not in {"string", "int"}:
        raise ValueError("Wrong label type: Please, use 'int' or 'string'.")

    if is_binary_task:
        X, y, _ = bin_class_dataset(
            filepath, _format, target, labels_colname,
            label_type, preprocess_func, **kwargs)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size)

        return (Xtr, ytr, Xte, yte, 2)

    X, y, num_classes = multi_class_dataset(
        filepath, _format, labels_colname,
        label_type, preprocess_func, **kwargs)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size)
    return (Xtr, ytr, Xte, yte, num_classes)
