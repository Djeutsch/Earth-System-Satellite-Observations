"""
Author: Laique Merlin Djeutchouang inspired from Debjani
"""
# Loading of the required packages
try:
    import sys
    import os
    import time
    from termcolor import colored
    import warnings
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.exceptions import DataConversionWarning
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report, accuracy_score
    import rasterio as rio
    import numpy as np
    from typing import List, Tuple, Set, Dict, Any, Optional
    import pandas as pd
    from glob import glob
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from string import ascii_lowercase as asci
except ModuleNotFoundError:
    print('Module import error')
    sys.exit()
else:
    print(colored(
        '\nBingo!!! All libraries properly loaded. Ready to start!!!', 'green'), '\n')

# Disable all warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Warning used to notify implicit data conversions happening in the code.
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')


def find_max_shape(filenames: List[str]) -> Tuple[int, int]:
    """
    This function finds maximum shapes among all the filenames present in files (list).
    :param filenames: list of strings, each string contains the path of the image filenames.
    :return: maximum x shape and maximum y shape of an image.
    """
    max_x = 0
    max_y = 0
    for file in filenames:
        img = rio.open(file)
        img_data = img.read()
        max_x = max(max_x, img_data.shape[1])
        max_y = max(max_y, img_data.shape[2])

    return max_x, max_y


def process_train_data(filenames: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes all the images present in the files, finds maximum shapes along
    x and y direction from all the images, makes a numpy array of zeroes of the shape of
    [maximum_x, maximum_y]. Reads all the images present in the files and stack them one
    by one in the numpy array. Takes a square root of the numpy array and returns the array.
    :param filenames: list of strings, each string contains the path of the image filenames.
    :return: Containing the square root of the images (pixels) present in the filenames.
    """
    max_x, max_y = find_max_shape(filenames)
    train_array = np.zeros((len(filenames), max_x, max_y, 11))
    for i, filename in enumerate(filenames):
        img = rio.open(filename)
        img_data = img.read()
        img_data = np.transpose(img_data, axes = [1, 2, 0])
        train_array[i, 0:img_data.shape[0], 0:img_data.shape[1], :] = img_data
    train_array = np.sqrt(train_array)

    # Normalize the data array for training
    norm_train_array = normalize_data(train_array)

    return train_array, norm_train_array


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Takes a numpy array and normalise it based on its min max values
    :param data: A Numpy array.
    :return: Normalised Numpy array.
    """
    min_array = data.min(axis=0).min(axis=0).min(axis=0)
    max_array = data.max(axis=0).max(axis=0).max(axis=0)
    norm_array = (data - min_array) / (max_array - min_array)

    return norm_array


def process_test_data(filename: str) -> np.ndarray:
    """
    Takes a numpy array and normalise it based on its min max values
    :param filename:
    :return: A normalised Numpy array
    """
    """
    :param data: A Numpy array
    :param min_array: A Numpy array
    :param max_array: A Numpy array
    :return: 
    """
    # Read the test image file and get its data
    img = rio.open(filename)
    img_data = img.read()
    img_data = np.transpose(img_data, axes=[1, 2, 0])
    img_data = np.sqrt(img_data)

    # Normalize data for testing
    min_array = img_data.min(axis=0).min(axis=0)
    max_array = img_data.max(axis=0).max(axis=0)
    norm_img_data = (img_data - min_array) / (max_array - min_array)

    return norm_img_data
