"""
Author: Laique Merlin Djeutchouang inspired from Debjani
"""
# Loading of the required packages
import os
import numpy as np
from .utils import utils
from typing import Any, List


class TestLandCoverSegmentationModel():
    """
    docstring
    """
    def __init__(self, model: Any, data_path: str,
                 file_name: str, band_list: List[str]) -> None:
        """
        :param model: a trained clustering model such as K-Means and Mini-batch K-Means
        :param data_path: path name of the directory containing the images
        :param file_name: for area specific clustering, file_name will be "area_name_*"
                        otherwise for all dataset clustering
        :param band_list: list of bands on which clustering will be performed
        """
        self.model = model
        self.data_path = data_path
        self.file_name = file_name
        self.band_list = band_list

    def test(self):
        """
        This function will do all the necessary data preparation for testing
        the clustering model on remote sensing data and then will cluster the test data.
        """
        filename: str = os.path.join(self.data_path, self.file_name)
        normalised_data = utils.process_test_data(filename)
        sub_normalised_data = normalised_data[:, :, self.band_list]

        norm_rgb_img = sub_normalised_data[:, :, 0:3]
        new_shape = (-1, sub_normalised_data.shape[2])
        test_data = sub_normalised_data.reshape(new_shape).astype(np.float64)
        test_clusters = self.model.predict(test_data)
        test_clusters = test_clusters.reshape(norm_rgb_img[:, :, 0].shape)

        return norm_rgb_img, test_clusters
