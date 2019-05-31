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
        total = l8_band_count + s1_band_count + dem_band_count - 1  # take one off because we don't want the QA band
        if return_total:
            return total
        return (l8_band_count, s1_band_count, dem_band_count, total)
   
    def get_tile_shape(self, reshape=False):
        if reshape:
            tile_shape = (self.__get_band_counts(return_total=True), self.tile_length, self.tile_length)
        else:
            tile_shape = (self.tile_length, self.tile_length, self.__get_band_counts(return_total=True))
        return tile_shape
        
    def tile_generator(self, pixel_locations, batch_size, fcn = False, merge=True):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
        tile_size = self.tile_length
        i = 0
        label_proj = Proj(self.label.crs)
        l8_proj = Proj(self.landsat[0].crs)
        # assuming all images have the same num of bands
        band_count  = self.__get_band_counts(return_total=True)
        class_count = self.class_count
        buffer = math.floor(tile_size / 2)
        while True:
            image_batch = np.zeros((batch_size, tile_size, tile_size, band_count))
            if fcn:
                label_batch = np.zeros((batch_size,tile_size, tile_size, class_count))
            else:
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
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                ### get label data
                # find gps of that pixel within the image
                (x, y) = self.landsat[dataset_index].xy(r, c)
                # convert the point we're sampling from to the same projection as the label dataset if necessary
                if l8_proj != label_proj:
                    x,y = transform(l8_proj,label_proj,x,y)
                    # reference gps in label_image
                row, col = self.label.index(x,y)
                flag = True
                if fcn:
                    data = self.label.read(1, window=Window(col-buffer, row-buffer, tile_size, tile_size))
                    data = util.merge_classes(data)
                    if 0 not in data and np.nan not in data:
                        label = self.one_hot_encode(data, tile_size, class_count)
                        label_batch[b] = label
                        flag = False
                else:
                    data = self.label.read(1, window=((row, row+1), (col, col+1)), masked=False, boundless=True)
                    data = util.merge_classes(data)
                    if 0 not in data and np.nan not in data:
                        label = data[0,0]                       
                        label = util.class_to_index[label]
                        label_batch[b][label] = 1
                        flag = False
                if not flag:
                    image_batch[b] = np.dstack((reshaped_tile, reshaped_s1_tile, reshaped_dem_tile))    
                    b += 1
            yield (image_batch, label_batch)   
            
    def one_hot_encode(self, data, tile_size, class_count):
        label = np.zeros((tile_size, tile_size, class_count))
        for i in range(tile_size):
            for j in range(tile_size):
                label_index = util.class_to_index[data[i][j]]
                label[i][j][label_index] = 1
        return label

    def evaluate_cnn(self, model, pixels, batch_size=64, merge=True):
        predictions = model.predict_generator(generator = self.tile_generator(pixels, batch_size=batch_size, merge=merge), steps=len(pixels) // batch_size, verbose=1)
        eval_generator = self.tile_generator(pixels, batch_size=1)
        labels = np.empty(predictions.shape)
        count = 0
        while count < len(labels):
            image_b, label_b = next(eval_generator)
            labels[count] = label_b
            count += 1
        label_index = np.argmax(labels, axis=1)     
        pred_index = np.argmax(predictions, axis=1)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        util.plot_confusion_matrix(label_index, pred_index, classes=np.array(list(util.indexed_dictionary)),
                      class_dict=util.indexed_dictionary)
        # Plot normalized confusion matrix
        util.plot_confusion_matrix(label_index, pred_index, classes=np.array(list(util.indexed_dictionary)),
                      class_dict=util.indexed_dictionary,
                      normalize=True)
        count = 0
        for i in range(len(label_index)):
            if(label_index[i] == pred_index[i]):
                count+=1
        print("Accuracy is {}".format(count/len(label_index)))
        