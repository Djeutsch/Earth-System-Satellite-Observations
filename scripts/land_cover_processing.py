##### Librarie and Package Requirements

from utils import *
import matplotlib.pyplot as plt



###### Visualization Utils ######
def arr_normalizer(array):
    """
    Normalizes numpy arrays into scale 0.0 - 1.0
    """
    array_min, array_max = array.min(), array.max()

    return ((array - array_min)/(array_max - array_min))


def plot_spectral_index(ax, fig, spindex_arr, spindex_name=""):

    if spindex_name == "NDWI":
        cmap = "YlGnBu"
    if spindex_name == "NDVI":
        cmap = "RdYlGn"
    if spindex_name == "DBI":
        cmap = "cividis"
    if spindex_name == "NDBI":
        cmap = "bone"

    img = ax.imshow(spindex_arr, cmap=cmap)
    ax.set_title(f"\n{spindex_name}")
    ax.axis("off")
    cbar = fig.colorbar(mappable=img, ax=ax, shrink=0.7, pad=0.05,
                        orientation="horizontal", extend="both")
    cbar.set_label(f"{spindex_name} range")
    

def plot_spectral_indices(input_file):
    """
    Plot the true color image (RGB) of the raster and its spectral indices. 
    
    parameters
    ----------
        input_file: str
            path directory to the raster file
    """
    ### Get data to plot
    spindices, RGB = spectral_indices(input_file)

    ### Normalize the bands and get the RGB image array
    red, green, blue = RGB
    redn = arr_normalizer(red)
    greenn = arr_normalizer(green)
    bluen = arr_normalizer(blue)
    rgb = np.dstack((redn, greenn, bluen))

    ##### Plotting
    fig, ax = plt.subplots(2, 3, figsize=(15, 12))
    ax = ax.reshape(-1)

    ### RGB natural color composite
    ax[0].imshow(rgb, cmap="terrain")
    ax[0].set_title("True Color Image (RGB)")
    ax[0].axis("off")

    ### Visualize all spectral indices
    for i, key in enumerate(spindices.keys()):
        plot_spectral_index(ax[i+1], fig, spindices[key], key)

    ax[-1].set_axis_off()
    raster_id = input_file.split("/")[-1].split(".")[0]
    fig.suptitle(
        f"\nSpectral Indeices of Sentinel-2 imagery\nraster ID: {raster_id}", fontsize=15)
    fig.tight_layout()
    plt.show()
    return fig


def plot_raster_bands(input_file):
    """
    Plot the raster bands. 
    
    parameters
    ----------
        input_file: str
            path directory to the raster file
    """
    raster = rio.open(input_file)
    img = raster.read()
    fig, ax = plt.subplots(2, 6, figsize=(20, 10))
    ax = ax.reshape(-1)
    for j in range(raster.count):
        ax[j].imshow(img[j, :, :])
        ax[j].set_title(raster.descriptions[j])
        ax[j].axis("off")

    ax[-1].set_axis_off()
    raster_id = input_file.split("/")[-1].split(".")[0]
    fig.suptitle(
        f"\nRaster bands (11) of Sentinel-2 imagery\nraster ID: {raster_id}", fontsize=15)
    fig.tight_layout()
    plt.show()
    return fig







if __name__ == "__main__":

    # Set up the path
    
    input_file = "../data-in/train/china_train1_0907.tif"
    _ = plot_raster_bands(input_file)
    _ = plot_spectral_indices(input_file)
    print("\nALL DONE!\n")
