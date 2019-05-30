import utilities as util
import rasterio
import numpy as np
import random
import math
import itertools
from rasterio.plot import show
from rasterio.windows import Window
import rasterio.features
import rasterio.warp
import rasterio.mask
from pyproj import Proj, transform
from tqdm import tqdm
from shapely.geometry import Polygon

class pixel_gen():
    
    def __init__(self, landsat, sentinel, dem, label, tile_size, class_count):
        self.landsat = landsat
        self.s1 = sentinel
        self.dem = dem
        self.label = label
        self.tile_length = tile_size
        self.balance = np.zeros(class_count)
        self.class_count = class_count
        self.pixels = list()
        self.train = list()
        self.val = list()
        self.test = list()
    
    def gen_pixels(self, pixel_count, balanced = True, merge = True, index = None):
        l8_data, s1_data, dem_data = self.__get_tiles_to_use(index)
        if balanced:
            self.pixels = self.__gen_balanced_pixel_locations(l8_data, s1_data, dem_data, self.label, pixel_count, self.tile_length, self.class_count, merge=merge)
        else:
            self.pixels = self.__gen_pixel_locations(l8_data, s1_data, dem_data, pixel_count, self.tile_length)
        print("pixels generated {}".format(len(self.pixels)))

    
    def __gen_balanced_pixel_locations(self, l8_data, s1_data, dem_data, label_dataset, pixel_count, tile_size, num_classes, merge=True):
        label_proj = Proj(label_dataset.crs)
        pixels = []
        pixel_count_per_dataset = math.ceil(pixel_count / len(l8_data))
        for index, l8_d in enumerate(tqdm(l8_data)):
            l8_proj = Proj(l8_d.crs)
            points_per_class = pixel_count_per_dataset // num_classes
            masked_label_image, raster_poly = util.make_label_mask(l8_d, label_dataset)
            if merge:
                masked_label_image = util.merge_classes(masked_label_image)
            all_points_per_image = []
            for cls in util.class_names:
                cls = int(cls)
                rows,cols = np.where(masked_label_image[0] == cls)
                all_locations = list(zip(rows,cols))
                random.shuffle(all_locations)
                l8_points = []
                if len(all_locations)!=0:
                    for r,c in all_locations[:math.ceil(7*points_per_class)]:
                        x,y = label_dataset.xy(r+raster_poly.bounds[1],c+raster_poly.bounds[0])
                        x,y = transform(label_proj, l8_proj, x, y)
                        r,c = l8_d.index(x,y)
                        l8_points.append((r,c))
                    class_px_index = [index] * len(l8_points)
                    class_px = list(zip(l8_points, class_px_index))
                    l8_points = self.__delete_black_tiles(l8_data, s1_data, dem_data, tile_size, class_px, max_size = points_per_class)
                    self.balance[util.class_to_index[cls]] += len(l8_points[:points_per_class])
                    all_points_per_image += l8_points[:points_per_class]
            pixels += all_points_per_image
        random.shuffle(pixels)
        return (pixels)

    def print_balance():
        for i in range(len(util.indexed_dictionary)):
            print("{}:{}".format(util.indexed_dictionary[i], self.balance[i]))
    
    
    def __gen_pixel_locations(self, l8_data, s1_data, dem_data, pixel_count, tile_size):
        pixels = []
        buffer = math.floor(tile_size/2)
        count_per_dataset = math.ceil(pixel_count / len(l8_datasets))
        for index, l8_data in enumerate(l8_datasets):
            #randomly pick `count` num of pixels from each dataset
            img_height, img_width = l8_data.shape
            rows = range(0+buffer, img_height-buffer)
            columns = range(0+buffer, img_width-buffer)
            points = random.sample(set(itertools.product(rows, columns)), math.ceil(10*count_per_dataset))
            dataset_index_list = [index] * count_per_dataset
            dataset_pixels = list(zip(points, dataset_index_list))
            dataset_pixels = self.__delete_black_tiles(l8_data, s1_data, dem_data, tile_size, dataset_pixels, max_size = count_per_dataset)
            pixels += dataset_pixels
        return (pixels)

    
    def __delete_black_tiles(self, l8_data, s1_data, dem_data, tile_size, pixels, max_size = None):
        buffer = math.floor(tile_size / 2)
        cloud_list = [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352]
        new_pixels = []
        l8_band_count = l8_data[0].count
        for pixel in pixels:
            r, c = pixel[0]
            dataset_index = pixel[1]
            tiles_to_read = [l8_data[dataset_index], s1_data[dataset_index], dem_data[dataset_index]]
            tile, s1_tile, dem_tile = util.read_windows(tiles_to_read, c ,r, buffer, tile_size)
            if np.isnan(tile).any() == True or -9999 in tile or tile.size == 0 or np.amax(tile) == 0 or np.isin(tile[7,:,:], cloud_list).any() or tile.shape != (l8_band_count, tile_size, tile_size):
                pass
            elif np.isnan(s1_tile).any() or np.isnan(dem_tile).any():
                pass
            else:
                new_pixels.append(pixel)
                if max_size != None and len(new_pixels) == max_size:
                    return new_pixels
        return new_pixels    
        
    def __get_tiles_to_use(self, index):
        if index == None:
            l8_data, s1_data, dem_data = (self.landsat, self.s1, self.dem)
        else:
            l8_data, s1_data, dem_data = (self.landsat[index:index+1], self.s1[index:index+1], self.dem[index:index+1])
        return (l8_data, s1_data, dem_data)
            
        



