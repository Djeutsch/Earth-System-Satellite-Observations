##### Librarie and Package Requirements, and Testing of them
from utils import *
import pandas as pd

def mbkm_clustering(data: pd.DataFrame, features: list,
                  spindex: str, num_clusters: int) -> pd.DataFrame:
    """
    Uses the unsupervised classification method Mini Batch K-Means to clusters or
    groups the pixel values stored in a dataframe whose columns are the rastars bands and
    spectral signatures of the rasters.
    
    parameters
    ----------
        data: DataFrame whose columns are the rastars bands and spectral
            signatures of the rasters.
        features: list of the main features of interest (11 bands).
        spindex: (str) the Spectral Index of interest.
        num_clusters: (int) number of pixel clusters or groups to create.
        
    Returns:
        A DataFrame (data) with an additional column of clusters (int).
    """
    ### Get none null instances according to the index of interest
    obs = data.loc[:, spindex].notnull().values
    prd = data.loc[:, features].notnull().all(1).values
    trn = obs & prd
    X_data = data.loc[trn, features]

    print(f"{' '*4} MBK-Means({num_clusters}) clustering ...")
    kmn = MiniBatchKMeans(n_clusters=num_clusters,
                          batch_size=4096,
                          random_state=19062018, verbose=0)  # Create the model

    # Fit the created model and predict the labels of clusters
    pred_labels = kmn.fit_predict(X_data)
    # Assign the predicted labels to the original data
    data.loc[trn, "clusters"] = pred_labels.astype(int)

    return data


def lclu_clustering(DATA_DIR: str, IN_DIR: str = "train") -> pd.DataFrame:
    """
    Clusters Sentinel-2 imagery data and assigns classes or land cover types
    using bands information and spectral indices NDWI, NDVI, DBI and NDBI.
    
    parameters
    ----------
        DATA_DIR: main directory of the data
        IN_DIR: folder name of inputs
        
    Returns:
        A DataFrame of grouped pixels of rasters in IN_DIR, where each pixel
        cluster has been assigned a land cover class.
    """
    ###### Get the data for Land Cover Clustering
    src_data, bands = lclu_data_processor(DATA_DIR, IN_DIR, verbose=False)
    spindices = ["NDWI", "NDVI", "DBI", "NDBI"]

    print(f"\n\nProcessed {IN_DIR}ing data clustering:")
    ###### NDWI
    data = src_data.copy()
    spindex = "NDWI"
    num_clusters = 2
    features = bands[:]
    features.append(spindex)

    print(f"\n{' '*2} Decision node(s) based on: {spindex}")
    df_lcc_ndwi = mbkm_clustering(data, features, spindex, num_clusters)

    ### Extract the Water class: where NDWI mean value > 0
    ndwi_mean_val = df_lcc_ndwi[[spindex, "clusters"]
                                ].groupby(by=["clusters"]).mean()
    try:
        cluster = ndwi_mean_val[ndwi_mean_val[spindex] > 0].index[0]
        idx_water = df_lcc_ndwi.clusters == int(cluster)
        water_data = df_lcc_ndwi[idx_water][bands].copy()
        water_data["classes"] = 1  # Assign a class for the Water land cover
        data = df_lcc_ndwi[~idx_water].copy()  # Where NDWI mean value <= 0
    except IndexError:  # Meaning there is no water
        features.append("classes")
        water_data = pd.DataFrame(columns=features)  # Empty data for water
        cluster1 = ndwi_mean_val.index[0]
        cluster2 = ndwi_mean_val.index[1]
        idx_nowater = (df_lcc_ndwi.clusters == int(cluster1)) | (
            df_lcc_ndwi.clusters == int(cluster2))
        data = df_lcc_ndwi[idx_nowater].copy()  # Where NDWI mean value <= 0

    ###### NDVI
    data = df_lcc_ndwi[~idx_water].copy()  # Where NDWI mean value <= 0
    spindex = "NDVI"
    num_clusters = 3
    features = bands[:]
    features.append(spindex)

    print(f"\n{' '*2} Decision node(s) based on: {spindex}")
    df_lcc_ndvi = mbkm_clustering(data, features, spindex, num_clusters)

    ### Get NDVI mean value per cluster
    ndvi_mean_val = df_lcc_ndvi[[spindex, "clusters"]
                                ].groupby(by=["clusters"]).mean()
    ndvi_mean_val = ndvi_mean_val.sort_values(spindex, ascending=False)

    ### Extract the Nat Veg class:  where there is the highest NDVI mean value
    cluster1 = ndvi_mean_val.index[0]
    idx_nat_veg = df_lcc_ndvi.clusters == int(cluster1)
    nat_veg_data = df_lcc_ndvi[idx_nat_veg][bands].copy()
    # Assign a class for the Natural Vegetation land cover
    nat_veg_data["classes"] = 2

    ### Extract the Agri Field class:  where there is the middle NDVI mean value
    cluster2 = ndvi_mean_val.index[1]
    idx_agr_field = df_lcc_ndvi.clusters == int(cluster2)
    agr_field_data = df_lcc_ndvi[idx_agr_field][bands].copy()
    # Assign a class for the Agricultural Field land cover
    agr_field_data["classes"] = 3

    ###### DBI
    clust_ndvi_lowest = ndvi_mean_val.index[2]
    idx_ndvi_lowest = df_lcc_ndvi.clusters == int(clust_ndvi_lowest)
    # Where there is the lowest NDVI mean value
    data = df_lcc_ndvi[idx_ndvi_lowest].copy()
    spindex = "DBI"
    num_clusters = 3
    features = bands[:]
    features.append(spindex)

    print(f"\n{' '*2} Decision node(s) based on: {spindex}")
    df_lcc_dbi = mbkm_clustering(data, features, spindex, num_clusters)

    ### Get DBI mean value per cluster
    dbi_mean_val = df_lcc_dbi[[spindex, "clusters"]
                              ].groupby(by=["clusters"]).mean()
    dbi_mean_val = dbi_mean_val.sort_values(spindex, ascending=False)

    ### Extract the Bare Ground class:  where there is the highest DBI mean value
    cluster = dbi_mean_val.index[0]
    idx_dbi1 = df_lcc_dbi.clusters == int(cluster)
    bare_ground_data1 = df_lcc_dbi[idx_dbi1][bands].copy()
    # Assign a class for the Bare Ground land cover
    bare_ground_data1["classes"] = 4

    ###### NDBI
    cluster1 = dbi_mean_val.index[1]
    cluster2 = dbi_mean_val.index[2]
    idx_dbi = (df_lcc_dbi.clusters == int(cluster1)) | (
        df_lcc_dbi.clusters == int(cluster2))
    data = df_lcc_dbi[idx_dbi].copy()  # Other two classes
    spindex = "NDBI"
    num_clusters = 2
    features = bands[:]
    features.append(spindex)

    print(f"\n{' '*2} Decision node(s) based on: {spindex}")
    df_lcc_ndbi = mbkm_clustering(data, features, spindex, num_clusters)

    ### Get NDBI mean value per cluster
    ndbi_mean_val = df_lcc_ndbi[[spindex, "clusters"]
                                ].groupby(by=["clusters"]).mean()
    ndbi_mean_val = ndbi_mean_val.sort_values(spindex, ascending=False)

    ### Extract the Bare Ground class:  where there is the highest NDBI mean value
    cluster = ndbi_mean_val.index[0]
    idx_dbi2 = df_lcc_ndbi.clusters == int(cluster)
    bare_ground_data2 = df_lcc_ndbi[idx_dbi2][bands].copy()
    # Assign a class for the Bare Ground land cover
    bare_ground_data2["classes"] = 4

    ### Extract the Built-up class:  where there is the lowest NDBI mean value
    cluster = ndbi_mean_val.index[1]
    idx_urban = df_lcc_ndbi.clusters == int(cluster)
    urban_data = df_lcc_ndbi[idx_urban][bands].copy()
    # Assign a class for the Urban/Buitl-up land cover
    urban_data["classes"] = 5

    ##### Merging all layers: the 5 types of land cover
    print(f"\n{' '*2} Merging of all the layers from: {spindices}\n")
    list_class = [water_data, nat_veg_data, agr_field_data,
                  bare_ground_data1, bare_ground_data2, urban_data]
    class_data = pd.concat(list_class, ignore_index=False).sort_index()

    ##### Retrieve the unclassified instances
    unknown_class_idx = []
    for idx in src_data.index:
        if idx not in class_data.index:
            unknown_class_idx.append(idx)

    unclass_data = src_data.iloc[unknown_class_idx, :][bands]
    # Assign a class for the unknown land cover type
    unclass_data["classes"] = 0

    ##### Merge the overall data: known and unknown classes
    merged_df = pd.concat([class_data, unclass_data],
                          ignore_index=False).sort_index()

    df_bands = merged_df[bands].copy()
    df_classes = merged_df[["classes"]]
    df_spindices = src_data[spindices].copy()
    lclu_data = df_bands.join(df_spindices)

    return lclu_data.join(df_classes), bands




