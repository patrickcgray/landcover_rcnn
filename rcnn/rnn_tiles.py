import rasterio
import numpy as np
import random
import math
from rasterio.windows import Window
from pyproj import Proj, transform
from rasterio.plot import reshape_as_raster, reshape_as_image
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utilities as util
    
    
class rnn_tile_gen():   
    def __init__(self, landsat, lc_label, canopy_label, tile_size, class_count):
        self.l8_dict = landsat
        self.lc_label = lc_label
        self.canopy_label = canopy_label
        self.tile_length = tile_size
        self.class_count = class_count
  
    def get_tile_shape(self):
        return (self.tile_length, self.tile_length, self.landsat[0].count)
    
    def tile_generator(self, pixel_locations, batch_size):
        ### this is a keras compatible data generator which generates data and labels on the fly 
        ### from a set of pixel locations, a list of image datasets, and a label dataset
        tile_size = self.tile_length
        i = 0
        lc_proj = Proj(self.lc_label.crs)
        canopy_proj = Proj(self.canopy_label.crs)
        l8_proj = Proj(self.l8_dict['028012'][0].crs)
        # assuming all images have the same num of bands
        band_count = self.l8_dict['028012'][0].count - 1
        time_steps = len(self.l8_dict['028012'])
        class_count = self.class_count
        buffer = math.floor(tile_size / 2)
        while True:
            image_batch = np.zeros((batch_size, time_steps, tile_size, tile_size, band_count))
            lc_batch = np.zeros((batch_size, tile_size, tile_size, class_count))
            canopy_batch = np.zeros((batch_size, tile_size, tile_size))
            b = 0
            while b < batch_size:
                # if we're at the end  of the data just restart
                if i >= len(pixel_locations):
                    i=0
                r, c = pixel_locations[i][0]
                tile_num = pixel_locations[i][1]
                i += 1
                tiles_to_read = self.l8_dict[tile_num]
                tiles_read = util.read_windows(tiles_to_read, c ,r, buffer, tile_size)
                reshaped_tiles = []
                for tile in tiles_read:
                    tile = tile[0:7]
                    reshaped_tile = (reshape_as_image(tile))  #- 982.5) / 1076.5
                    reshaped_tiles.append(reshaped_tile)
                ### get label data
                # find gps of that pixel within the image
                (x, y) = self.l8_dict[tile_num][0].xy(r, c) 
                # convert the point we're sampling from to the same projection as the label dataset if necessary
                if l8_proj != lc_proj:
                    lc_x,lc_y = transform(l8_proj,lc_proj,x,y)
                if l8_proj != canopy_proj:
                    canopy_x, canopy_y = transform(l8_proj,canopy_proj,x,y)
                # reference gps in label_image
                lc_row, lc_col = self.lc_label.index(lc_x,lc_y)
                lc_data = self.lc_label.read(1, window=Window(lc_col-buffer, lc_row-buffer, tile_size, tile_size))
                canopy_row, canopy_col = self.canopy_label.index(canopy_x,canopy_y)
                canopy_data = self.canopy_label.read(1, window=Window(canopy_col-buffer, canopy_row-buffer, tile_size, tile_size))
                
                if 0 not in lc_data and np.nan not in lc_data and np.nan not in canopy_data and 255 not in canopy_data and canopy_data.shape == (32, 32):
                    print(canopy_data)
                    lc_label = self.one_hot_encode(lc_data, tile_size, class_count)
                    lc_batch[b] = lc_label
                    canopy_batch[b] = canopy_data
                    total_tile = np.array((*reshaped_tiles,))
                    image_batch[b] = total_tile
                    b += 1
            yield (image_batch, {'landcover': lc_batch, 'canopy': canopy_batch})   
            
    def one_hot_encode(self, data, tile_size, class_count):
        label = np.zeros((tile_size, tile_size, class_count))
        for i in range(tile_size):
            for j in range(tile_size):
                label_index = util.class_to_index[data[i][j]]
                label[i][j][label_index] = 1
        return label     
   