import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np

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
        