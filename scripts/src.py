##### Librarie and Package Requirements, and Testing of them
try:
    import sys
    import os
    from termcolor import colored
    import warnings
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.exceptions import DataConversionWarning
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report, accuracy_score
    import rasterio as rio
    import numpy as np
    import pandas as pd
    from glob import glob
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from string import ascii_lowercase as asci
except ModuleNotFoundError:
    print(colored('\n\nModule improt error\n', 'red'))
    sys.exit()
else:
    print(colored(
        '\n\nBingo!!! All libraries properly loaded. Ready to start!!!', 'green'), '\n')

##### Disable all warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

##### Warning used to notify implicit data conversions happening in the code.
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')


############### Data Access ###############

##### Raster file loader
def load_raster(input_file: str):
    """
    Returns a raster array which consists of its bands and
    transformation matrix parameters
    ----------
        input_file: str
            path directory to the raster file
    """
    with rio.open(input_file) as src:
        band = src.read()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        profile = src.profile
        raster_img = np.rollaxis(band, 0, 1)

        output = {"band": band,
                  "raster_img": raster_img,
                  "transform": transform,
                  "crs": crs,
                  "shape": shape,
                  "profile": profile}

        return output

##### Raster file writer
def write_raster(raster, crs, transform, output_file):
    """
    Writes a raster array which consists of one band to the disc.
    ----------
        raster:
            raster array
        transform: 
            transformation matrix parameters
        output_file: str
            path directory to write the raster file
    """
    profile = {"driver": "GTiff",
               "compress": "lzw",
               "width": raster.shape[0],
               "height": raster.shape[1],
               "crs": crs,
               "transform": transform,
               "dtype": raster.dtype,
               "count": 1,
               "tiled": False,
               "interleave": 'band',
               "nodata": 0}

    profile.update(dtype=raster.dtype,
                   height=raster.shape[0],
                   width=raster.shape[1],
                   nodata=0,
                   compress="lzw")

    with rio.open(output_file, "w", **profile) as out:
        out.write_band(1, raster)



############### Spectral Indices (SIs) ###############

##### Normalized Difference Vegetation Index (NDVI)
def NDVI(nir, red):
    """
    Calculates NDVI
    
    parameters
    ----------
        nir: NIR band as input
        red: RED band as input
    """
    NDVI = (nir.astype("float") - red.astype("float")) / \
        (nir.astype("float") + red.astype("float"))

    return NDVI

##### Dry Bareness Index (DBI)
def DBI(green, swinr1, ndvi):
    """
    Calculate DBI
    
    parameters
    ----------
        swinr1: SWINR1 band as input
        green: green band as input
    """
    DBI = ((swinr1.astype("float") - green.astype("float")) /
           (swinr1.astype("float") + green.astype("float"))) - ndvi

    return DBI

##### Modified Normalized Difference Water Index (NDWI)
def NDWI(green, swinr1):
    """
    Calculate MNDWI
    
    parameters
    ----------
        swinr1: MINR band as input
        green: GREEN band as input
    """
    NDWI = (green.astype("float") - swinr1.astype("float")) / \
        (green.astype("float") + swinr1.astype("float"))

    return NDWI

##### Normalized Difference Built-up Index (NDBI)
def NDBI(swinr1, nir):
    """
    Calculate NDBI
    
    parameter
    ---------
        swinr: SWINR band as input
        nir: NIR band as input
    """
    NDBI = (swinr1.astype("float") - nir.astype("float")) / \
        (swinr1.astype("float") + nir.astype("float"))

    return NDBI

#####
def spectral_indix(input_file: str, sp_index: str="NDVI", verbose: bool=True):
    """
    Calculate the specified Spectral Index. 
    
    parameters
    ----------
        input_file: str
            path directory to the raster file
        sp_index: spectral indix of interest: NDVI, DBI, NDWI, NDBI
    """
    if verbose:
        print(f"\nThe spectral indix: {sp_index}, is being calculated ...",)

    ### Get the image id from the image_path
    img_id = input_file.split("/")[-1].split(".")[0]
    print(f"{' '*2} Raster image ID: {img_id}")

    ### Load the raster image
    if not input_file.endswith(".tif"):
        return "S\nSorry! The file entered is not for a raster image."
    else:
        raster = load_raster(input_file)

        ### Slice the bands: our data has only 11 bands instead of 13
        # Bands = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12')
        blue = raster["band"][0, :, :]  # represented by B2
        green = raster["band"][1, :, :]  # represented by B3
        red = raster["band"][2, :, :]  # represented by B4
        nir = raster["band"][6, :, :]  # represented by B8
        swinr1 = raster["band"][9, :, :]  # represented by B11
        swinr2 = raster["band"][10, :, :]  # represented by B12

        if sp_index == "NDVI":
            ### Calculate NDVI
            return NDVI(nir, red)

        elif sp_index == "DBI":
            ### Calculate DBI
            ndvi = NDVI(nir, red)
            return DBI(green, swinr1, ndvi)

        elif sp_index == "NDWI":
            ### Calculate NDWI
            return NDWI(green, swinr1)

        elif sp_index == "NDBI":
            ### Calculate NDBI
            return NDBI(swinr1, nir)

        else:
            alert = "\nSorry! The spectral indix is one of these: NDVI, DBI, SAVI, NDWI, NDBI!\n"
            return alert

#####
def spectral_indices(input_file: str, verbose: bool=False) -> tuple:
    """
    Calculate the Spectral Indices: NDVI, DBI, SAVI, NDWI, NDBI. 
    
    parameters
    ----------
        input_file: str
            path directory to the raster file
    """
    ### Get the image id from the image_path
    if verbose:
        img_id = input_file.split("/")[-1].split(".")[0]
        print(f"{' '*2} Spectral Indices from raster image ID: {img_id}")

    ### Load the raster image
    if not input_file.endswith(".tif"):
        return "S\nSorry! The file entered is not for a raster image."
    else:
        raster = load_raster(input_file)

        ### Slice the bands: note our data has only 11 bands instead of 13
        # Bands = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12')
        blue = raster["band"][0, :, :]  # represented by B2
        green = raster["band"][1, :, :]  # represented by B3
        red = raster["band"][2, :, :]  # represented by B4
        nir = raster["band"][6, :, :]  # represented by B8
        swinr = raster["band"][9, :, :]  # represented by B11
        swinr1 = raster["band"][10, :, :]  # represented by B12

        ### Calculate NDWI, NDVI, DBI, NDBI, SAVI
        ndvi = NDVI(nir, red)
        spindices = {"NDWI": NDWI(green, swinr1),
                     "NDVI": ndvi,
                     "DBI": DBI(green, swinr1, ndvi),
                     "NDBI": NDBI(swinr, red, ndvi)}

        RGB = (red, green, blue)

        return spindices, RGB



######### Data Processor for Land Cover Clustering (LCC) ############

#####
def lclu_data_processor(DATA_DIR: str,
                       IN_DIR: str = "train",
                       verbose: bool = True) -> tuple:
    """
    Preprocesses Sentinel-2 imagery data to extract features from the
    raster bands, and the calculated spectral indices for Land Cover
    segmentation and classification.
    
    parameters
    ----------
        DATA_DIR: main directory of the data
        IN_DIR: folder name of inputs
        
    Returns:
        A tuple(DataFrame, list_of_bands)
    """
    print(f"\n\nPreprocessing of {IN_DIR}ing data:\n")
    ### Set up the paths
    input_raster = os.path.join(DATA_DIR, IN_DIR, "*.tif")
    data_paths = sorted(glob(input_raster))

    num_bands = 11
    IMG_arr_list = []
    spindex_arr_dict = {"NDWI": [], "NDVI": [], "DBI": [], "NDBI": []}
    for image_path in data_paths:
        ### Extract and process the feature data from the bands
        if verbose:
            img_id = image_path.split("/")[-1].split(".")[0]
            print(f"{' '*2} Raster image ID: {img_id}")
        raster = rio.open(image_path)
        # Reshape as (num_samples, num_bands)
        img_arr = np.moveaxis(raster.read(), 0, -1).reshape(-1, num_bands)
        IMG_arr_list.append(img_arr)

        ### Extract and process the target data from the spectral index
        for spindex in spindex_arr_dict.keys():
            arr_spindex = spectral_indix(image_path, spindex, verbose=False)
            # Reshape as (num_samples, 1)
            spindex_arr = arr_spindex.reshape(-1, 1)
            spindex_arr_dict[spindex].append(
                spindex_arr)  # Update its list values

    ### Data
    bands = list(raster.descriptions)
    df_bands = pd.DataFrame(data=np.vstack(
        IMG_arr_list), columns=bands)
    spindex_arr_dict = {key: np.vstack(
        spindex_arr_dict[key]).ravel() for key in spindex_arr_dict.keys()}
    spindex_df = pd.DataFrame(data=spindex_arr_dict)

    return df_bands.join(spindex_df), bands







######### Additional Performance Metric for classification
def jaccard_score(y_true, y_pred) -> float:
    """
    Compute Jaccard similarity score to evaluate the accuracy of a classification
    
    parameters:
        y_true: array-like of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred: array-like of shape (n_samples,)
            Estimated targets as returned by a classifier
            
    Returns:
        The calculated metric (float)
    """
    ### y_true, y_pred must be a flatten vector
    try:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
    except:
        pass
    ### Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    ### Compute mean IoU
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(float)
    mean_IoU = np.nanmean(IoU).item()

    return mean_IoU









######### Plotting Tools

##### To annotate plots
def number_figures(axes, pos=None, labels=asci, braces=True, **text_kwargs):

    def depth(L): return isinstance(L, list) and max(map(depth, L)) + 1

    if pos is None:
        pos = [[0.99, 0.93]] * len(axes)
    elif (depth(pos) == 1) & (len(pos) == 2):
        pos = [list(pos)] * len(axes)
    elif (depth(pos) != 2) & (len(pos) != len(axes)):
        raise (Exception, 'check the position is the right format')

    for c, ax in enumerate(axes):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        w = x1 - x0
        h = y1 - y0
        x = x0 + w * pos[c][0]
        y = y0 + h * pos[c][1]

        t = '%s' % labels[c]
        if braces:
            t = '(%s)' % (t)
        ax.text(x, y, t, **text_kwargs)

##### Plot the confusion matrix and feature importance


def plot_cfmatrix_fimp_lgbm(lgbm_clf, test_y, 
                            pred_labels, labels, bands):
    ##### Get the confusion matrix
    cm = confusion_matrix(test_y, pred_labels)
    df_cm = pd.DataFrame(cm, labels, labels)

    ##### Feature importance and Heat Map
    fimp = pd.DataFrame(sorted(zip(lgbm_clf.feature_importance(), bands)),
                        columns=["Value", "Feature"])
    fimp = fimp.set_index("Feature")/lgbm_clf.feature_importance().sum()
    fimp.reset_index(inplace=True)

    fig = plt.figure(figsize=[24, 8])
    grid = mpl.gridspec.GridSpec(1, 2, wspace=0.15, width_ratios=[0.55, 0.45])
    ax = [fig.add_subplot(grid[0, 0]),
          fig.add_subplot(grid[0, 1])]
    ###
    #sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, ax=ax[0], cbar=False)
    ax[0].set_xlabel("\nTarget class")
    ax[0].set_ylabel("Estimated class")
    ax[0].set_title("Confusion Matrix")
    ###
    sns.barplot(x="Value", y="Feature", data=fimp, ax=ax[1], palette="dark")
    ax[1].set_title("Light GBM Features' Importance")
    ax[1].set_xlabel(
        "Importances\n(Usefullness fraction of the feature in the model)")
    ax[1].set_ylabel("Features")

    txt_kwargs = {"fontsize": 20, "c": "k"}
    number_figures(ax, pos=[0.005, 1.015], braces=True, **txt_kwargs)
    plt.show()
    return fig





