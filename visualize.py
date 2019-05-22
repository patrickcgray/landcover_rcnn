import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np
from pyproj import Proj, transform
from shapely.geometry import Polygon
from rasterio.windows import Window

class VisualizeData:
    def __init__(self, landsat_datasets, label_dataset):
        self.landsat = landsat_datasets
        self.labels = label_dataset
   
    def view_landsat_tile(self, landsat_index):
        image_dataset = self.landsat[landsat_index]
        full_img = self.landsat[landsat_index].read()
        index = np.array([3, 2, 1])
        colors = full_img[index, :, :].astype(np.float64)
        max_val = 4000
        min_val = 0
        # Enforce maximum and minimum values
        colors[colors[:, :, :] > max_val] = max_val
        colors[colors[:, :, :] < min_val] = min_val
        for b in range(colors.shape[0]):
            colors[b, :, :] = colors[b, :, :] * 1 / (max_val - min_val)
        colors_reshaped = reshape_as_image(colors)
        fig, ax = plt.subplots(figsize=(10, 10))
        # plot with normal matplotlib functions
        ax.imshow(colors_reshaped)
        ax.set_title("RGB in matplotlib imshow")
    
    def view_labels(self, landsat_index):
        label_proj = Proj(self.labels.crs)
        image_dataset = self.landsat[landsat_index]
        raster_points = image_dataset.transform * (0, 0), image_dataset.transform * (image_dataset.width, 0), image_dataset.transform * (image_dataset.width, image_dataset.height), image_dataset.transform * (0, image_dataset.height)
        l8_proj = Proj(image_dataset.crs)
        new_raster_points = []
        # convert the raster bounds from landsat into label crs
        for x,y in raster_points:
            x,y = transform(l8_proj,label_proj,x,y)
            # convert from crs into row, col in label image coords
            row, col = self.labels.index(x, y)
            # don't forget row, col is actually y, x so need to swap it when we append
            new_raster_points.append((col, row))
        # turn this into a polygon
        raster_poly = Polygon(new_raster_points)
        # Window.from_slices((row_start, row_stop), (col_start, col_stop))
        masked_label_image = self.labels.read(window=Window.from_slices((int(raster_poly.bounds[1]), int(raster_poly.bounds[3])), (int(raster_poly.bounds[0]), int(raster_poly.bounds[2]))))
        fig, ax = plt.subplots(figsize=[15,15])
        ax.imshow(masked_label_image[0,:,:])
        
        
        