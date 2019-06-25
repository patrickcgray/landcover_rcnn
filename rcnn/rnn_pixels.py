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
import importlib
importlib.reload(util)

def make_pixels(tile_size, tile_list):
    points = []
    l8_points = []
    buffer = math.floor(tile_size/2)
    x = np.arange(0+buffer, 5000-buffer, tile_size)
    y = np.arange(0+buffer, 5000-buffer, tile_size)
    for row in x:
        for col in y:
            point = (row, col)
            points.append(point)
    #tile_list = ['028012', '029011','028011']
    for tile in tile_list:
        for point in points:
            l8_points.append((point, tile))
    random.shuffle(l8_points)
    return l8_points


#def gen_pixel_locations(l8_data, pixel_count, tile_size):
#    pixels = []
#    buffer = math.floor(tile_size/2)
#    count_per_dataset = math.ceil(pixel_count / len(l8_data))
#    for index, l8_d in l8_data.items():
        #randomly pick `count` num of pixels from each dataset
#        img_height, img_width = l8_d[0].shape
#        rows = range(0+buffer, img_height-buffer)
#        columns = range(0+buffer, img_width-buffer)
#        points = random.sample(set(itertools.product(rows, columns)), math.ceil(10*count_per_dataset))
#        dataset_index_list = [index] * math.ceil(10*count_per_dataset)
#        dataset_pixels = list(zip(points, dataset_index_list))
      #  dataset_pixels = delete_black_tiles(l8_data, tile_size, dataset_pixels, max_size = count_per_dataset)
 #       pixels += dataset_pixels
 #   return (pixels)

def train_val_test_split(pixels, train_val_ratio, val_test_ratio):
    random.shuffle(pixels)
    train_px = pixels[:int(len(pixels)*train_val_ratio)]
    val_pixels = pixels[int(len(pixels)*train_val_ratio):]
    val_px = val_pixels[:int(len(val_pixels)*val_test_ratio)]
    test_px = val_pixels[int(len(val_px)*val_test_ratio):]
    print("train:{} val:{} test:{}".format(len(train_px), len(val_px), len(test_px)))
    return (train_px, val_px, test_px)
    
def delete_bad_tiles(l8_data, lc_label, canopy_label, pixels, tile_size):
    buffer = math.floor(tile_size / 2)
    cloud_list = [72, 80, 96, 130, 132, 136, 160, 224]
    new_pixels = []
    l8_proj = Proj(l8_data['028012'][0].crs)
    lc_proj = Proj(lc_label.crs)
    canopy_proj = Proj(canopy_label.crs)
    for pixel in pixels:
        r, c = pixel[0]
        dataset_index = pixel[1]
        tiles_to_read = l8_data[dataset_index]
        tiles_read = util.read_windows(tiles_to_read, c ,r, buffer, tile_size)
        (x, y) = l8_data[dataset_index][0].xy(r, c) 
        # convert the point we're sampling from to the same projection as the label dataset if necessary
        if l8_proj != lc_proj:
            lc_x,lc_y = transform(l8_proj,lc_proj,x,y)
        if l8_proj != canopy_proj:
            canopy_x, canopy_y = transform(l8_proj,canopy_proj,x,y)
         # reference gps in label_image
        lc_row, lc_col = lc_label.index(lc_x,lc_y)
        lc_data = lc_label.read(1, window=Window(lc_col-buffer, lc_row-buffer, tile_size, tile_size))
        canopy_row, canopy_col = canopy_label.index(canopy_x,canopy_y)
        canopy_data = canopy_label.read(1, window=Window(canopy_col-buffer, canopy_row-buffer, tile_size, tile_size))
        flag = True
        if 0 in lc_data or np.nan in lc_data or np.nan in canopy_data or 255 in canopy_data or canopy_data.shape != (tile_size, tile_size) or len(np.unique(lc_data)) == 1:
            flag = False
        for tile in tiles_read:
            if np.isnan(tile).any() == True or -9999 in tile or tile.size == 0 or np.amax(tile) == 0 or np.isin(tile[7,:,:], cloud_list).any() or tile.shape != (l8_data[dataset_index][0].count, tile_size, tile_size):
                flag = False
                break
        if flag:
            new_pixels.append(pixel)
    return new_pixels    

#        if max_size != None and len(new_pixels) == max_size:
#           return new_pixels# max_size=None


