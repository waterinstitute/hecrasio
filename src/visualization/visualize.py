import os
import pathlib as pl
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.crs import CRS
from matplotlib import colors


# Add custom basemaps to folium
basemaps = {
    "Google Maps": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Maps",
        overlay=True,
        control=True,
    ),
    "Google Satellite": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Google Terrain": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Terrain",
        overlay=True,
        control=True,
    ),
    "Google Satellite Hybrid": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Esri Satellite": folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=True,
        control=True,
    ),
}

def list_files(path: str, file_type: str):
    """Create a list in the directory given by the path string"""
    li_files = [
        file for file in os.listdir(path) if os.path.splitext(file)[-1] == file_type
    ]
    li_all_files = list()
    for file in li_files:
        li_all_files.append(os.path.join(path, file))
    return li_all_files


def colorize(array, cmap="terrain"):
    normed_data = (array - array.min()) / (array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data)

from rasterio.crs import CRS



def array_from_tiff(huc:str)-> tuple:
    with rxr.open_rasterio("huc_"+huc+".tif") as array_Ar:
        crs_wgs84 = CRS.from_string("EPSG:4326")
        array_A_wgs84 = array_Ar.rio.reproject(crs_wgs84)
        array_Ar_reproject = np.squeeze(np.ma.getdata(array_A_wgs84.to_masked_array()))
    return array_A_wgs84, array_Ar_reproject
