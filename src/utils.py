# Util functions used to create SICS dataset

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.features import rasterize

from skimage.morphology import skeletonize
from scipy.spatial import distance

import cv2

import spyndex
import xarray as xr

# Global variables
channels = ['Blue','Green','Red','NIR']

standards = ['B','G','R','N']

def get_index(bands, index):

        """Calculate spectral index using spyndex library"""

        # scale bands to 0-1
        bands = scale_bands(bands)

        da = xr.DataArray(
            bands,
            dims = ("band","x","y"),
            coords = {"band": channels}
        )

        params = {standard: da.sel(band = channel) for standard,channel in zip(standards,channels)}

        idx = spyndex.computeIndex(
            index = index,
            params = params)


        return idx

def stack_10m_bands(safe_dir):
    """Stacks Sentinel-2 10m bands from a given .SAFE directory.
    Returns:
        stacked_array (numpy.ndarray): Stacked Sentinel-2 bands (Bands x Height x Width)
        reference_raster (rasterio.DatasetReader): Rasterio dataset for metadata access
    """
    
    # Define band selection (Sentinel-2 L2A band resolutions)
    band_map = {
        "B02": "R10m",  # Blue
        "B03": "R10m",  # Green
        "B04": "R10m",  # Red
        "B08": "R10m"   # NIR
    }

    granule_path = glob.glob(os.path.join(safe_dir, "GRANULE", "*"))[0]  # Get the GRANULE directory
    img_data_path = os.path.join(granule_path, "IMG_DATA")  # IMG_DATA directory

    stacked_bands = []
    reference_raster = None  # To store a rasterio dataset

    for band, res in band_map.items():
        band_path = glob.glob(os.path.join(img_data_path, res, f"*_{band}_*.jp2"))
        if not band_path:
            print(f"Band {band} not found, skipping...")
            continue
        
        band_path = band_path[0]
        with rasterio.open(band_path) as src:
            band_data = src.read(1)

            stacked_bands.append(band_data)

            # Use the first valid raster file as a reference
            if reference_raster is None:
                reference_raster = rasterio.open(band_path)

    if not stacked_bands:
        raise ValueError("No valid Sentinel-2 bands found in the given SAFE directory.")

    # Stack into a single array (Bands x Height x Width)
    stacked_array = np.stack(stacked_bands, axis=0)
    return stacked_array, reference_raster

def stack_crop_resample(safe_dir, output_tif, utm=None,resampling_factor=1):
    """Stacks Sentinel-2 10m bands, crops using UTM coordinates, resamples and saves as a .tif.
    
    Args:
        safe_dir (str): Path to the Sentinel-2 .SAFE directory.
        output_tif (str): Path to save the output .tif file.
        resampling_factor (float): Resampling factor (default=1, no resampling).
        p1 (tuple): First UTM coordinate (x_min, y_min).
        p2 (tuple): Second UTM coordinate (x_max, y_max).

    Returns:
        None
    """
    
    band_map = {
        "B02": "R10m",  # Blue
        "B03": "R10m",  # Green
        "B04": "R10m",  # Red
        "B08": "R10m"   # NIR
    }

    granule_path = glob.glob(os.path.join(safe_dir, "GRANULE", "*"))[0]
    img_data_path = os.path.join(granule_path, "IMG_DATA")

    stacked_bands = []
    reference_raster = None
    left, top, right, bottom = utm

    for band in ["B04", "B03","B02", "B08"]: # Save as R, G, B, NIR
        res = band_map[band]
        band_path = glob.glob(os.path.join(img_data_path, res, f"*_{band}_*.jp2"))
        if not band_path:
            print(f"Band {band} not found, skipping...")
            continue
        
        band_path = band_path[0]

        with rasterio.open(band_path) as src:

            # Define cropping window
            window = from_bounds(left, bottom, right, top, transform=src.transform)

            # New dimensions after resampling
            height = int(window.height * resampling_factor)
            width = int(window.width * resampling_factor)

            band_data = src.read(1,
                                window=window,
                                out_shape=(1, height, width),
                                resampling=Resampling.bilinear)
     

            # Compute the transform for the cropped image first
            transform = src.window_transform(window)

            # Apply resampling scaling to the transform
            transform = transform * transform.scale(
                (window.width / width),
                (window.height / height)
            )

            stacked_bands.append(band_data)

            # Store reference raster metadata
            if reference_raster is None:
                reference_raster = src
    

    if not stacked_bands:
        raise ValueError("No valid Sentinel-2 bands found.")

    stacked_array = np.stack(stacked_bands, axis=0)

    # Calculate NDWI and NDVI
    """NDWI = get_index(stacked_array, 'NDWI')
    NDVI = get_index(stacked_array, 'NDVI')
    NDWI = NDWI.expand_dims(dim="band")
    NDVI = NDVI.expand_dims(dim="band")

    stacked_array = np.vstack((stacked_array, NDWI, NDVI))"""

    # Save output .tif file
    out_meta = reference_raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": stacked_array.shape[1],
        "width": stacked_array.shape[2],
        "count": stacked_array.shape[0],
        "dtype": stacked_array.dtype,
        "transform": transform
    })

    with rasterio.open(output_tif, "w", **out_meta) as dst:
        for i in range(stacked_array.shape[0]):
            dst.write(stacked_array[i], i + 1)

    print(f"Saved stacked Sentinel-2 bands to {output_tif}")

    return stacked_array

def display_bands(img, downsample=True,scale=10):
    """Visualize all SR bands of a satellite image."""

    band_names = ["Blue", "Green", "Red", "NIR"]
    n = len(band_names)

    fig, axs = plt.subplots(1, n, figsize=(20, 5))

    for i in range(n):
        band = img[i]
        if downsample:
            band = cv2.resize(band, (band.shape[1] // scale, band.shape[0] // scale), interpolation=cv2.INTER_AREA)
        axs[i].imshow(band, cmap="gray")
        axs[i].set_title(band_names[i])
        axs[i].axis("off")

def scale_bands(img,satellite="sentinel"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

def get_rgb(img, r=2,g=1,b=0, contrast=1):
    """Convert a stacked array of bands to RGB"""

    if img.shape[0] > img.shape[2]:
        #  H x W x B -> B x H x W
        img = np.transpose(img, (2,0,1))

    r = img[r]
    g = img[g]
    b = img[b]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb.astype(np.float32)

    rgb = scale_bands(rgb)
    rgb = np.clip(rgb, 0, contrast) / contrast

    # convert to 255
    rgb = (rgb * 255).astype(np.uint8) 

    return rgb


def thin_edge_map(edge_map):
    """
    Post-process a 2D binary edge map:
    1. Apply thinning to reduce edge width to 1 pixel.
    2. Connect nearby disjoint edge segments using interpolation.

    Parameters:
    - edge_map (np.ndarray): 2D binary array (values 0 or 1).
    - connect_distance (int): Max pixel distance between endpoints to connect.

    Returns:
    - np.ndarray: Post-processed binary edge map (values 0 or 1).
    """
    # Step 1: Thinning
    binary = edge_map > 0
    thinned = skeletonize(binary).astype(np.uint8)

    return thinned

def get_line_mask(raster,line):
    """
    Get a mask of the line in the raster
    """
    line_mask = rasterize(
        [line.geometry],
        out_shape=raster.shape,
        transform=raster.transform,
        fill=0,
        dtype='uint16'
    )
    return line_mask