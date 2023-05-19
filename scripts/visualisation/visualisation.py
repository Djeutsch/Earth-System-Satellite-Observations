"""
Author: Laique Merlin Djeutchouang inspired from Debjani
"""
# Loading of the required packages
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from ..utils import utils
from typing import List, Tuple


def plot_satellite_image(filename: str,
                         band_list: List[int],
                         fig_size: Tuple[int]=(6, 6)) -> None:
    """
    Creates image plots for 3 channels images.
    :param filename: path to the tif image file
    :param band_list: contains indices of bands to be used to create 3-channel images
    :param fig_size: contains dimensions of the 2D figure size
    """
    
    assert len(band_list) == 3, "Incorrect number of channels"
    img_data = rio.open(filename).read()
    img_data = np.transpose(img_data, axes=[1, 2, 0])
    rgb_img_data = img_data[:, :, band_list]
    rgb_img_data = np.sqrt(rgb_img_data)
    norm_rgb_img_data = utils.normalize_data(rgb_img_data)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(norm_rgb_img_data[:, :, [0, 1, 2]])
    ax.imshow(norm_rgb_img_data)
    plt.show()

   
    