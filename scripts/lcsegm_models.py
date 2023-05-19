"""
Author: Laique Merlin Djeutchouang inspired from Debjani
"""
# Loading of the required packages
import os
import glob
from sklearn import cluster
from .utils import utils
import numpy as np
# import numpy.typing as npt
from typing import Any, List, Tuple


class LandCoverSegmentationModel():
    """
    This class is to perform the land cover segmentation for remote sensing satellite imagery data.
    """
    def __init__(self, file_path: str, file_name: str,
                 band_list: List, n_cluster: int) -> None:
        """
        :param file_path: Path name of the directory which contains the images
        :param file_name: For area specific clustering, file_name will be "area_name_*"
                          otherwise for all dataset clustering file_name will be "*"
        :param band_list: List of bands on which clustering will be performed
        :param n_cluster: Number of cluster to be generated
        """
        self.file_path = file_path
        self.file_name = file_name
        self.band_list = band_list
        self.n_cluster = n_cluster

    def kmeans_clustering(self) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        This function does all the necessary data preparation for training the K-Means algorithm
        and perform the clustering for these remote sensing satellite data.
        :return: model, clusters, and cluster_centers
        """
        # Path of base data directory: it is assumed that the file is kept inside
        data_dir = self.file_path
        data_files = glob.glob((os.path.join(data_dir, self.file_name)))
        train_data, _ = utils.process_train_data(data_files)
        normalised_train_data = utils.normalize_data(train_data)
        sub_train_data = normalised_train_data[:, :, :, self.band_list]

        # Put the data in the required format for clustering
        new_shape = (-1, sub_train_data.shape[3])
        train_x = sub_train_data.reshape(new_shape)

        # Perform Clustering
        k_means = cluster.KMeans(n_clusters=self.n_cluster)
        k_means.fit(train_x)
        x_cluster = k_means.labels_
        cluster_centers = k_means.cluster_centers_
        x_cluster = x_cluster.reshape(sub_train_data[:, :, :, 0].shape)

        return k_means, x_cluster, cluster_centers

