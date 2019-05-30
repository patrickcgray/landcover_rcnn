import utilities as util
import rasterio
import numpy as np
import random
import math
import itertools
from rasterio.plot import show
from rasterio.windows import Window
from pyproj import Proj, transform
from tqdm import tqdm
from shapely.geometry import Polygon
from rasterio.plot import reshape_as_raster, reshape_as_image

class tile_gen():
    def __init__(self, landsat, sentinel, dem, label, tile_size, class_count):
        self.landsat = landsat
        self.s1 = sentinel
        self.dem = dem
        self.label = label
        self.tile_length = tile_size
        self.class_count = class_count
    
    def __get_band_counts(self, return_total=False):
        l8_band_count = self.landsat[0].count  
        s1_band_count = self.s1[0].count
        dem_band_count = self.dem[0].count
        total = l8_band_count + s1_band_count + dem_band_count
        if return_total:
            return total
        return (l8_band_count, s1_band_count, dem_band_count, total)
   
    def get_tile_shape(self, reshape=False):
        if reshape:
            tile_shape = (self.__get_band_counts(return_total=True), self.tile_length, self.tile_length)
        else:
            tile_shape = (self.tile_length, self.tile_length, self.__get_band_counts(return_total=True))
        return tile_shape
        
    def tile_generator(self, pixel_locations, batch_size, merge=True):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
        tile_size = self.tile_length
        i = 0
        bad_count = 0
        label_proj = Proj(self.label.crs)
        l8_proj = Proj(self.landsat[0].crs)
        # assuming all images have the same num of bands
        band_count  = self.__get_band_counts(return_total=True)
        class_count = self.class_count
        buffer = math.floor(tile_size / 2)
        while True:
            image_batch = np.zeros((batch_size, tile_size, tile_size, band_count-1)) # take one off because we don't want the QA band
            label_batch = np.zeros((batch_size,class_count))
            b = 0
            while b < batch_size:
                # if we're at the end  of the data just restart
                if i >= len(pixel_locations):
                    i=0
                r, c = pixel_locations[i][0]
                dataset_index = pixel_locations[i][1]
                i += 1
                tiles_to_read = [self.landsat[dataset_index], self.s1[dataset_index], self.dem[dataset_index]]
                tile, s1_tile, dem_tile = util.read_windows(tiles_to_read, c ,r, buffer, tile_size)
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                #reshape from raster format to image format and standardize according to image wide stats
                reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                ### get label data
                # find gps of that pixel within the image
                (x, y) = self.landsat[dataset_index].xy(r, c)
                # convert the point we're sampling from to the same projection as the label dataset if necessary
                if l8_proj != label_proj:
                    x,y = transform(l8_proj,label_proj,x,y)
                    # reference gps in label_image
                row, col = self.label.index(x,y)
                window = ((row, row+1), (col, col+1))
                data = self.label.read(1, window=window, masked=False, boundless=True)
                if merge:
                    data = util.merge_classes(data)
                label = data[0,0]
                if np.isnan(reshaped_s1_tile).any() or np.isinf(reshaped_s1_tile).any():
                    bad_count+=1
                    print(bad_count)
                    pass
                elif np.isnan(reshaped_dem_tile).any() or np.isinf(reshaped_dem_tile).any():
                    bad_count+=1
                    print(bad_count)
                    pass             
                elif label == 0 or np.isnan(label).any() or label not in util.class_to_index:
                    bad_count+=1
                    print(bad_count)
                    pass
                else:                   
                    label = util.class_to_index[label]
                    label_batch[b][label] = 1
                    image_batch[b] = np.dstack((reshaped_tile, reshaped_s1_tile, reshaped_dem_tile))    
                    b += 1
            yield (image_batch, label_batch)     
