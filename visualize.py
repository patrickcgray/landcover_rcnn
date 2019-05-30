import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np
from pyproj import Proj, transform
from shapely.geometry import Polygon
from rasterio.windows import Window
import random
import math
import utilities as util

class VisualizeData:
    
    def __init__(self, landsat_datasets, label_dataset):
        self.landsat, self.s1, self.dem, self.label = util.load_data()
        self.label_proj = Proj(label_dataset.crs)
        self.open_figs = list()
        self.colors = util.colors
        self.class_names = util.class_names
   
    def view_landsat(self, landsat_index):
        image_dataset = self.landsat[landsat_index]
        full_img = self.landsat[landsat_index].read()
        colors_reshaped = self.__normalize_rgb(full_img)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(colors_reshaped)
        ax.set_title("RGB in matplotlib imshow")
    
    def view_labels(self, landsat_index):
        masked_label_image, raster_poly = util.make_a_label_mask(self.landsat[landsat_index], self.label_proj)
        ax = self.__plot_a_tile(masked_label_image[0,:,:], colors=self.colors)
        
    def print_a_tile(self, landsat_index, class_index, tile_size, middle=False):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        buffer = math.floor(tile_size / 2)
        masked_label_image, raster_poly = util.make_a_label_mask(self.landsat[landsat_index], self.label_proj)
        rows,cols = np.where(masked_label_image[0] == class_index)
        all_locations = list(zip(rows,cols))
        if len(all_locations) == 0:
            return
        landsat_tile = np.zeros(1)
        while np.isnan(landsat_tile).any() == True or -9999 in landsat_tile or np.amax(landsat_tile) == 0:
            tile_loc = all_locations[random.randint(0, len(all_locations)-1)]
            r_label, c_label = tile_loc
            r_landsat, c_landsat = self.__transform_to_l8(landsat_index, r_label, c_label, raster_poly)
            landsat_tile = self.__read_from_raster(self.landsat[landsat_index], c_landsat-buffer, r_landsat-buffer, tile_size) 
        
        label_tile = self.__read_from_raster(self.labels, c_label+raster_poly.bounds[0]-buffer, r_label+raster_poly.bounds[1]-buffer, tile_size)
        landsat_tile_normalized = self.__normalize_rgb(landsat_tile)
        ax_c = self.__plot_a_tile(landsat_tile_normalized)
        ax_c.set_title(self.class_names[class_index])
        ax_l = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors)
        ax_l.set_title(self.class_names[class_index])
        if middle:
            ax_m = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors, middle = True)
            ax_m.set_title(self.class_names[class_index])
            
    
    def __get_landsat_raster(self, r_label, c_label, landsat_index, raster_poly):

        return landsat_tile
        
    #takes landsat row_col
    def print_chosen_tile(self, landsat_index, tile_size, row_col, middle=False):
        buffer = math.ceil(tile_size / 2)
        r_landsat, c_landsat = row_col
        r_label, c_label = self.__transform_to_label(landsat_index, r_landsat, c_landsat)
        class_index = self.__get_label(r_label, c_label)
        label_tile = self.__read_from_raster(self.labels, c_label-buffer, r_label-buffer, tile_size)
        landsat_tile = self.__read_from_raster(self.landsat[landsat_index], c_landsat-buffer, r_landsat-buffer, tile_size)
        landsat_tile_normalized = self.__normalize_rgb(landsat_tile)
        ax_c = self.__plot_a_tile(landsat_tile_normalized)
        ax_c.set_title(self.class_names[class_index])
        ax_l = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors)
        ax_l.set_title(self.class_names[class_index])
        if middle:
            ax_m = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors, middle = True)
            ax_m.set_title(self.class_names[class_index])
        
        
    def __get_label(self, r_label, c_label):
        window = ((r_label, r_label+1), (c_label, c_label+1))
        data = self.labels.read(1, window=window, masked=False, boundless=True)
        label = data[0,0]
        return label
    
    def __read_from_raster(self, raster, column_start, row_start, tile_size):
        w = raster.read(window=Window(column_start, row_start, tile_size, tile_size))
        return w
    
    def __plot_a_tile(self, tile, colors = None, middle = False):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.open_figs.append(fig)
        if colors == None:
             ax.imshow(tile)
        else:
            colored_label_img = np.zeros((tile.shape[0], tile.shape[1], 3))
            center_px = math.floor(tile.shape[0]/2)
            for h in range(tile.shape[0]):
                for w in range(tile.shape[1]):
                    if h == center_px and w == center_px and middle:
                        colored_label_img[h][w] = (0,0,0)
                    elif tile[h][w] not in self.class_names:
                         colored_label_img[h][w] = (0,0,0)
                    else:
                        colored_label_img[h][w] = np.divide(colors[tile[h][w]], 255)
            print("sanity check print middle pixel {}".format(tile[center_px][center_px]))
            ax.imshow(colored_label_img)
        return ax
  
    def close_figs(self):
        for fig in self.open_figs:
            plt.close(fig)
    def __transform_to_l8(self, landsat_index, r_label, c_label, raster_poly):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        x,y = self.labels.xy(r_label+raster_poly.bounds[1],c_label+raster_poly.bounds[0])
        # go from label projection into landsat projection
        x,y = transform(self.label_proj, l8_proj,x,y)
        # convert from landsat geographic space into row col
        r_landsat,c_landsat = self.landsat[landsat_index].index(x,y) 
        return r_landsat, c_landsat
   
    def __transform_to_label(self, landsat_index, r_landsat, c_landsat):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        x,y = self.landsat[landsat_index].xy(r_landsat,c_landsat)
        x,y = transform(l8_proj, self.label_proj,x,y)
        r_label,c_label = self.labels.index(x,y) 
        return r_label, c_label
        
    def __normalize_rgb(self, tile):
        tile = tile[[3, 2, 1],:,:].astype(np.float64)
        max_val = 4000
        min_val = 0
        # Enforce maximum and minimum values
        tile[tile[:, :, :] > max_val] = max_val
        tile[tile[:, :, :] < min_val] = min_val
        for b in range(tile.shape[0]):
            tile[b, :, :] = tile[b, :, :] * 1 / (max_val - min_val)
        tile_reshaped = reshape_as_image(tile)
        return tile_reshaped
        
      
    def count_labels(self,landsat_index):
        buckets = dict()
        label_mask, raster_poly = util.make_a_label_mask(self.landsat[landsat_index], self.label_proj)
        for cls in self.class_names:
            rows,cols = np.where(label_mask[0] == cls)
            all_locations = list(zip(rows,cols))
            buckets[cls] = len(all_locations)
        return buckets
            
        
        
        