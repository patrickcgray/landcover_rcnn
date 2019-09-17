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
import importlib

importlib.reload(util)
    
    
class rnn_tile_gen():   
    def __init__(self, landsat, lc_label, canopy_label, tile_size, class_count):
        self.l8_dict = landsat
        self.lc_label = lc_label
        self.canopy_label = canopy_label
        self.tile_length = tile_size
        self.class_count = class_count
  
    def get_tile_shape(self):
        return (self.tile_length, self.tile_length, self.landsat[0].count)
    
    def tile_generator(self, pixel_locations, batch_size, flatten=False, canopy=False):
        ### this is a keras compatible data generator which generates data and labels on the fly 
        ### from a set of pixel locations, a list of image datasets, and a label dataset
        bad_tiles = 0
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
            if flatten:
                image_batch = np.zeros((batch_size, time_steps, band_count))
                lc_batch = np.zeros((batch_size, class_count))
                canopy_batch = np.zeros((batch_size, 1))

            else:
                image_batch = np.zeros((batch_size, time_steps, tile_size, tile_size, band_count))
                lc_batch = np.zeros((batch_size, tile_size,tile_size, class_count))
                canopy_batch = np.zeros((batch_size, tile_size, tile_size, 1))
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
                band_avg = [  345.72448081,   352.93755735,   319.34257128,   899.39728239,
         649.46125258,   370.53562465, -1084.8218946 ]
                band_std = [ 661.75737932,  363.32761072,  425.28671553,  959.63442896,
        838.86193201,  501.96992987, 3562.42995552]
                for tile in tiles_read:
                    tile = tile[0:7]
                    reshaped_tile = reshape_as_image(tile).astype(np.float64)
                    for jj in range(7):
                        if flatten:
                            reshaped_tile[0][0][jj] = np.divide(np.subtract(reshaped_tile[0][0][jj],band_avg[jj]),band_std[jj])
                        else:
                            reshaped_tile[:][:][jj] = np.divide(np.subtract(reshaped_tile[:][:][jj],band_avg[jj]),band_std[jj])
                    reshaped_tiles.append(reshaped_tile)
                ### get label data
                # find gps of that pixel within the image
                (x, y) = self.l8_dict[tile_num][0].xy(r, c) 
                # convert the point we're sampling from to the same projection as the label dataset if necessary
                # TODO not working here either
                #if l8_proj != lc_proj:
                #    lc_x,lc_y = transform(l8_proj,lc_proj,x,y)
                lc_x,lc_y = x,y
                # TODO not working here either
                #if l8_proj != canopy_proj:
                #    canopy_x, canopy_y = transform(l8_proj,canopy_proj,x,y)
                canopy_x, canopy_y = x,y
                # reference gps in label_image
                lc_row, lc_col = self.lc_label.index(lc_x,lc_y)
                lc_data = self.lc_label.read(1, window=Window(lc_col-buffer, lc_row-buffer, tile_size, tile_size))
                canopy_row, canopy_col = self.canopy_label.index(canopy_x,canopy_y)
                canopy_data = self.canopy_label.read(1, window=Window(canopy_col-buffer, canopy_row-buffer, tile_size, tile_size))
                lc_label = self.one_hot_encode(lc_data, tile_size, class_count)
                if flatten:
                    lc_batch[b] = lc_label.reshape(class_count)
                    canopy_batch[b] = canopy_data.reshape(1) / 100
                    total_tile = np.array((*reshaped_tiles,))
                    image_batch[b] = total_tile.reshape((4,7))
                else:
                    lc_batch[b] = lc_label #lc_label.reshape(tile_size*tile_size, class_count)      
                    canopy_batch[b] = canopy_data.reshape((tile_size, tile_size, 1)) / 100
                    total_tile = np.array((*reshaped_tiles,))
                    image_batch[b] = total_tile
                b += 1
            if canopy:
                yield (image_batch, {'landcover': lc_batch, 'canopy': canopy_batch})
            else: 
                yield (image_batch, lc_batch)
            
    def one_hot_encode(self, data, tile_size, class_count):
        label = np.zeros((tile_size, tile_size, class_count))
        flag = True
        count = 0 
        for i in range(tile_size):
            for j in range(tile_size):
                label_index = util.class_to_index[data[i][j]]
                label[i][j][label_index] = 1
        return label 
   