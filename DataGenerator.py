import numpy as np
from rasterio.plot import adjust_band
from rasterio.windows import Window
from pyproj import Proj, transform
import random
import math
import itertools

class DataGenerator:
    def tile_generator(image_datasets, label_dataset, tile_height, tile_width, pixel_locations, batch_size):
        ### this is a keras compatible data generator which generates data and labels on the fly 
        ### from a set of pixel locations, a list of image datasets, and a label dataset
        
        # pixel locations looks like [r, c, dataset_index]
        label_image = label_dataset.read()
        label_image[label_image == 255] = 1
    
        c = r = 0
        i = 0
        
        outProj = Proj(label_dataset.crs)
    
        # assuming all images have the same num of bands
        band_count = image_datasets[0].count
        class_count = len(np.unique(label_image))
        buffer = math.ceil(tile_height / 2)
      
        while True:
            image_batch = np.zeros((batch_size, tile_height, tile_width, band_count-1)) # take one off because we don't want the QA band
            label_batch = np.zeros((batch_size,class_count))
            b = 0
            while b < batch_size:
                # if we're at the end  of the data just restart
                if i >= len(pixel_locations):
                    i=0
                c, r = pixel_locations[i][0]
                dataset_index = pixel_locations[i][1]
                i += 1
                tile = image_datasets[dataset_index].read(list(np.arange(1, band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                if np.amax(tile) == 0: # don't include if it is part of the image with no pixels
                    pass
                elif np.isnan(tile).any() == True or -9999 in tile: 
                    # we don't want tiles containing nan or -999 this comes from edges
                    # this also takes a while and is inefficient
                    pass
                elif tile.shape != (band_count, tile_width, tile_height):
                    print('wrong shape')
                    print(tile.shape)
                    # somehow we're randomly getting tiles without the correct dimensions
                    pass
                elif np.isin(tile[7,:,:], [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352]).any() == True:
                    # make sure pixel doesn't contain clouds
                    # this is probably pretty inefficient but only checking width x height for each tile
                    # read more here: https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1873_US_Landsat_ARD_DFCB_0.pdf
                    #print('Found some cloud.')
                    #print(tile[7,:,:])
                    pass
                else:
                    tile = adjust_band(tile[0:7])
                    # reshape from raster format to image format
                    reshaped_tile = reshape_as_image(tile)
    
                    # find gps of that pixel within the image
                    (x, y) = image_datasets[dataset_index].xy(r, c)
    
                    # convert the point we're sampling from to the same projection as the label dataset if necessary
                    inProj = Proj(image_datasets[dataset_index].crs)
                    if inProj != outProj:
                        x,y = transform(inProj,outProj,x,y)
    
                    # reference gps in label_image
                    row, col = label_dataset.index(x,y)
    
                    # find label
                    label = label_image[:, row, col]
                    # if this label is part of the unclassified area then ignore
                    if label == 0 or np.isnan(label).any() == True:
                        pass
                    else:
                        # add label to the batch in a one hot encoding style
                        label_batch[b][label] = 1
                        image_batch[b] = reshaped_tile
                        b += 1
            yield (image_batch, label_batch)
    
    def gen_pixel_locations(image_datasets, train_count, val_count, tile_size):
        ### this function pulls out a train_count + val_count number of random pixels from a list of raster datasets
        ### and returns a list of training pixel locations and image indices 
        ### and a list of validation pixel locations and indices
        
        ## future improvements could make this select classes evenly
        train_pixels = []
        val_pixels = []
        
        buffer = math.ceil(tile_size/2)
        
        train_count_per_dataset = math.ceil(train_count / len(image_datasets))
        val_count_per_dataset = math.ceil(val_count / len(image_datasets))
       
        total_count_per_dataset = train_count_per_dataset + val_count_per_dataset
        for index, image_dataset in enumerate(image_datasets):
            #randomly pick `count` num of pixels from each dataset
            img_height, img_width = image_dataset.shape
            
            rows = range(0+buffer, img_height-buffer)
            columns = range(0+buffer, img_width-buffer)
            #rows_sub, columns_sub = zip(*random.sample(list(zip(rows, columns)), total_count))
            
            points = random.sample(set(itertools.product(rows, columns)), total_count_per_dataset)
            
            dataset_index_list = [index] * total_count_per_dataset
            
            dataset_pixels = list(zip(points, dataset_index_list))
            
            train_pixels += dataset_pixels[:train_count_per_dataset]
            val_pixels += dataset_pixels[train_count_per_dataset:]
            
            
        return (train_pixels, val_pixels)