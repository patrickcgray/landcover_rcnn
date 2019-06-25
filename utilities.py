import rasterio
from rasterio.plot import adjust_band
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.plot import show
from rasterio.windows import Window
import rasterio.features
import rasterio.warp
import rasterio.mask
from shapely.geometry import Polygon
from pyproj import Proj, transform
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def read_txt(filenames):
    pixels = list()
    for file in filenames:
        opened_file = open(file, "r")
        list_px = opened_file.readlines()
        px = list()
        for line in list_px:
            split = line.split()
            index = split[1][-1]
            row = split[0][1:-1]
            col = split[1][:-2]
            px.append(((int(row), int(col)),int(index)))
        pixels.append(px)
    return pixels

def read_windows(rasters, c, r, buffer, tile_size):
    tiles = []
    #only works when rasters are in same projection
    for raster in rasters:
        tile = raster.read(list(np.arange(1, raster.count+1)), window=Window(c-buffer, r-buffer, tile_size, tile_size))
        tiles.append(tile)
    return (*tiles,)

def get_class_count():
    return len(class_names)

def merge_classes(y):
    y[y == 22] = 23
    y[y == 24] = 23
    return y
   
def make_label_mask(landsat, label):
    image_dataset = landsat
    label_proj = Proj(label.crs)
    raster_points = image_dataset.transform * (0, 0), image_dataset.transform * (image_dataset.width, 0), image_dataset.transform * (image_dataset.width, image_dataset.height), image_dataset.transform * (0, image_dataset.height)
    l8_proj = Proj(image_dataset.crs)
    new_raster_points = []
    # convert the raster bounds from landsat into label crs
    for x,y in raster_points:
        x,y = transform(l8_proj,label_proj,x,y)
        # convert from crs into row, col in label image coords
        row, col = label.index(x, y)
        # don't forget row, col is actually y, x so need to swap it when we append
        new_raster_points.append((col, row))
        # turn this into a polygon
    raster_poly = Polygon(new_raster_points)
        # Window.from_slices((row_start, row_stop), (col_start, col_stop))
    masked_label_image = label.read(window=Window.from_slices((int(raster_poly.bounds[1]), int(raster_poly.bounds[3])), (int(raster_poly.bounds[0]), int(raster_poly.bounds[2]))))
    return masked_label_image, raster_poly

def load_data():
    label_dataset = rasterio.open('/deep_data/NLCD/NLCD_2016_Land_Cover_L48_20190424.img')

    l8_image_paths = [
    '/deep_data/processed_landsat/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined.tif',  
    '/deep_data/processed_landsat/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_028010_20170714_20190429_C01_V01_SR_combined.tif'
    ]

    s1_image_paths = [
    '/deep_data/sentinel_sar/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined/aligned-LC08_CU_027012_20170907_20181121_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined/aligned-LC08_CU_028011_20170907_20181130_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined/aligned-LC08_CU_028012_20171002_20171019_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined/aligned-LC08_CU_028012_20171103_20190429_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined/aligned-LC08_CU_029011_20171018_20190429_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028010_20170714_20190429_C01_V01_SR_combined/aligned-LC08_CU_028010_20170714_20190429_C01_V01_SR_combined_SAR.tif'
   ]

    dem_image_paths = [
    '/deep_data/sentinel_sar/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-79.69001_33.95762_-77.7672_35.51886__4500X4631_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-77.7672_35.00779_-75.79042_36.58923__4500X4262_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-79.69001_33.95762_-77.7672_35.51886__4500X4631_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-78.07896_33.69485_-76.14021_35.27466__4500X4248_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-76.14021_34.71847_-74.14865_36.318__4500X4408_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028010_20170714_20190429_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-77.44453_36.318_-75.42829_37.90193__4500X4283_ShowLogo_False_tiff_depth=32f.tiff'
    ]

    landsat_datasets = []
    for fp in l8_image_paths:
        landsat_datasets.append(rasterio.open(fp))
    sentinel_datasets = []
    for fp in s1_image_paths:
        sentinel_datasets.append(rasterio.open(fp))
    dem_datasets = []
    for fp in dem_image_paths:
        dem_datasets.append(rasterio.open(fp))
    return (landsat_datasets, sentinel_datasets, dem_datasets, label_dataset)      
        
def plot_confusion_matrix(y_true, y_pred, classes, class_dict,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # convert class_id to class_name using the class_dict
    cover_names = []
    for cover_class in classes:
        cover_names.append(class_dict[cover_class])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    #print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=cover_names, yticklabels=cover_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax





colors = dict((
    (11, (0,0,255)), #water ~ blue
#(12, (0,0,255)), #snow ~ white
(21, (255,0,0)), #open space developed ~ red
(22, (50,0,0)), # low intensity developed ~ darker red
(23, (50,0,0)), # medium intensity developed ~ darker darker red
(24, (50,0,0)), # high intensity developed ~ darker darker darker red
(31, (153,76,0)), # barren land ~ dark orange
(41, (0,204,0)), # deciduous forest ~ green
(42, (0,153,0)), # evergreen forest ~ darker green
(43, (0,102,0)), # mixed forest ~ darker darker green
(52, (153,0,76)), #schrub ~ dark pink
(71, (255,153,71)), # grass land ~  orange
(81, (204,204,0)),#pasture ~ yellowish
(82, (153,153,0)),#cultivated land ~ darker yellow
(90, (0,255,255)), #woody wetland ~ aqua
(95, (0,102,102)), #emergent herbaceous wetlands ~ darker aqua
))


class_names = dict((
(11, "Water"),
(12, "Snow/Ice"),
(21, "Open Space Developed"),
(22, "Low Intensity Developed"),
(23, "Medium Intensity Developed"),
(24, "High Intensity Developed"),
(31, "Barren Land"),
(41, "Deciduous Forest"),
(42, "Evergreen Forest"),
(43, "Mixed Forest"),
#(51, "Dwarf Scrub/Shrub - ALASKA"),
(52, "Scrub/Shrub"),
(71, "Grassland / Herbaceous"),
#(72, "Sedge / Herbaceous - ALASKA"),
#(73, "Lichen / Herbaceous - ALASKA"),f
#(74, "Moss - ALASKA"),
(81, "Pasture/Hay"),
(82, "Cultivated Land"),
(90, "Woody Wetland"),
(95, "Emergent Herbaceous Wetlands"),
))

class_to_index = dict((
(11, 0),
#(12, 1),
(21, 3),
(22, 1),
(23, 1),
(24, 1),
(31, 4),
(41, 2),
(42, 2),
(43, 2),
(52, 2),
(71, 3),
(81, 3),
(82, 3),
(90, 2),
(95, 5),
))

indexed_dictionary = dict((
(0, "Water"),
(1, "Developed"),
(2, "Forest"),
(3, "Cultivated"),
(4, "Barren"),
(5, "Wetland"),
))


old_indexed_dictionary = dict((
(0, "Water"),
(1, "Snow/Ice"),
(2, "Open Space Developed"),
(3, "Low Intensity Developed"),
(4, "Medium Intensity Developed"),
(5, "High Intensity Developed"),
(6, "Barren Land"),
(7, "Deciduous Forest"),
(8, "Evergreen Forest"),
(9, "Mixed Forest"),
#(51, "Dwarf Scrub/Shrub - ALASKA"),
(10, "Scrub/Shrub"),
(11, "Grassland / Herbaceous"),
#(72, "Sedge / Herbaceous - ALASKA"),
#(73, "Lichen / Herbaceous - ALASKA"),
#(74, "Moss - ALASKA"),
(12, "Pasture/Hay"),
(13, "Cultivated Land"),
(14, "Woody Wetland"),
(15, "Emergent Herbaceous Wetlands"),
))