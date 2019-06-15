import utilities as util
import rasterio
import numpy as np
import random
import math
import itertools
from rasterio.windows import Window
from pyproj import Proj, transform
from tqdm import tqdm
from shapely.geometry import Polygon
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utilities as util

def gen_pixel_locations(l8_data, pixel_count, tile_size):
    pixels = []
    buffer = math.floor(tile_size/2)
    count_per_dataset = math.ceil(pixel_count / len(l8_data))
    for index, l8_d in l8_data.items():
        #randomly pick `count` num of pixels from each dataset
        img_height, img_width = l8_d[0].shape
        rows = range(0+buffer, img_height-buffer)
        columns = range(0+buffer, img_width-buffer)
        points = random.sample(set(itertools.product(rows, columns)), math.ceil(10*count_per_dataset))
        dataset_index_list = [index] * math.ceil(10*count_per_dataset)
        dataset_pixels = list(zip(points, dataset_index_list))
      #  dataset_pixels = delete_black_tiles(l8_data, tile_size, dataset_pixels, max_size = count_per_dataset)
        pixels += dataset_pixels
    return (pixels)

def delete_bad_tiles(l8_data, pixels, tile_size, max_size=None):
    buffer = math.floor(tile_size / 2)
    cloud_list = [72, 80, 96, 130, 132, 136, 160, 224]
    new_pixels = []
    for pixel in pixels:
        r, c = pixel[0]
        dataset_index = pixel[1]
        tiles_to_read = l8_data[dataset_index]
        tiles_read = util.read_windows(tiles_to_read, c ,r, buffer, tile_size)
        flag = True
        for tile in tiles_read:
            if np.isnan(tile).any() == True or -9999 in tile or tile.size == 0 or np.amax(tile) == 0 or np.isin(tile[7,:,:], cloud_list).any() or tile.shape != (l8_data[dataset_index][0].count, tile_size, tile_size):
                flag = False
                break
        if flag:
            new_pixels.append(pixel)
        if max_size != None and len(new_pixels) == max_size:
            return new_pixels
    return new_pixels    


