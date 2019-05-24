import rasterio
import numpy as np
import random
import math
import itertools

from rasterio.plot import adjust_band
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.plot import show
from rasterio.windows import Window
import rasterio.features
import rasterio.warp
import rasterio.mask

from pyproj import Proj, transform
from tqdm import tqdm
from shapely.geometry import Polygon

class_names = dict((
(11, "Water"),
(12, "Snow/Ice"),
(21, "Open Space Developed"),
(22, "Low Intensity Developed"),
(23, "Medium Intensity Developed"),
(24, "High Intensity Developed"),
(31, "Barren Land"),
(41, "Deciduous Forest"),
(42, "Evergreen Forest"),
(43, "Mixed Forest"),
#(51, "Dwarf Scrub/Shrub - ALASKA"),
(52, "Scrub/Shrub"),
(71, "Grassland / Herbaceous"),
#(72, "Sedge / Herbaceous - ALASKA"),
#(73, "Lichen / Herbaceous - ALASKA"),
#(74, "Moss - ALASKA"),
(81, "Pasture/Hay"),
(82, "Cultivated Land"),
(90, "Woody Wetland"),
(95, "Emergent Herbaceous Wetlands"),
))


def merge_classes(y):
    # medium intensity and high intensity
    y[y == 3] = 2

    # open space developed, cultivated land, and pasture hay
    y[y == 5] = 6
    y[y == 7] = 6

    # decidious and mixed
    y[y == 9] = 11

    # evergreen and scrub shrub
    y[y == 12] = 10

    # pal wetland and pal scrub shrub
    y[y == 14] = 15

    # est forest and est scrub shrub
    y[y == 17] = 16
    
    return(y)


def gen_balanced_pixel_locations(image_datasets, train_count, label_dataset, merge=False):
    ### this function pulls out a train_count + val_count number of random pixels from a list of raster datasets
    ### and returns a list of training pixel locations and image indices 
    ### and a list of validation pixel locations and indices
    
    label_proj = Proj(label_dataset.crs)
    num_classes = len(class_names)
    
    train_pixels = []
    
    train_count_per_dataset = math.ceil(train_count / len(image_datasets))
    for index, image_dataset in enumerate(tqdm(image_datasets)):
        # how many points from each class
        points_per_class = train_count_per_dataset // num_classes
        
        # get landsat boundaries in this image
        # create approx dataset mask in geographic coords
        # this fcn maps pixel locations in (row, col) coordinates to (x, y) spatial positions
        raster_points = image_dataset.transform * (0, 0), image_dataset.transform * (image_dataset.width, 0), image_dataset.transform * (image_dataset.width, image_dataset.height), image_dataset.transform * (0, image_dataset.height)
        l8_proj = Proj(image_dataset.crs)
        new_raster_points = []
        # convert the raster bounds from landsat into label crs
        for x,y in raster_points:
            x,y = transform(l8_proj,label_proj,x,y)
            # convert from crs into row, col in label image coords
            row, col = label_dataset.index(x, y)
            # don't forget row, col is actually y, x so need to swap it when we append
            new_raster_points.append((col, row))
        # turn this into a polygon
        raster_poly = Polygon(new_raster_points)
        # Window.from_slices((row_start, row_stop), (col_start, col_stop))
        masked_label_image = label_dataset.read(window=Window.from_slices((int(raster_poly.bounds[1]), int(raster_poly.bounds[3])), (int(raster_poly.bounds[0]), int(raster_poly.bounds[2]))))
        if merge:
            masked_label_image = merge_classes(masked_label_image)
        # loop for each class
        all_points_per_image = []
        for cls in class_names:
            cls = int(cls)
            # mask the label subset image to each class
            # pull out the indicies where the mask is true
            rows,cols = np.where(masked_label_image[0] == cls)
            all_locations = list(zip(rows,cols))
       
            # shuffle all locations
            random.shuffle(all_locations)
            # now convert to landsat image crs
            # TODO need to time this to see if it is slow, can probably optimize
            l8_points = []
            # TODO Will probably need to catch this for classes smaller than the ideal points per class
            if len(all_locations)!=0:
                for r,c in all_locations[:points_per_class]:
                # convert label row and col into label geographic space
                    x,y = label_dataset.xy(r+raster_poly.bounds[1],c+raster_poly.bounds[0])
                # go from label projection into landsat projection
                    x,y = transform(label_proj, l8_proj,x,y)
                # convert from landsat geographic space into row col
                    r,c = image_dataset.index(x,y)
                    l8_points.append((r,c))
                all_points_per_image += l8_points

        dataset_index_list = [index] * len(all_points_per_image)

        dataset_pixels = list(zip(all_points_per_image, dataset_index_list))
        train_pixels += dataset_pixels
    random.shuffle(train_pixels)
    return (train_pixels)


def gen_pixel_locations(image_datasets, train_count, val_count, tile_size):
    #This function takes in a list of raster datasets and randomly samples `train_count` and `val_count` random pixels from each dataset.
    # It doesn't sample within tile_size / 2 of the edge in order to avoid missing data.
    
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
    
def tile_generator(l8_image_datasets, s1_image_datasets, dem_image_datasets, ndvi_image_datasets, label_dataset, tile_height, tile_width, pixel_locations, batch_size, merge=False):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
    
    class_to_index = dict((
        (11, 0),
        (12, 1),
        (21, 2),
        (22, 3),
        (23, 4),
        (24, 5),
        (31, 6),
        (41, 7),
        (42, 8),
        (43, 9),
        (52, 10),
        (71, 11),
        (81, 12),
        (82, 13),
        (90, 14),
        (95, 15),
        ))

    c = r = 0
    i = 0
    
    label_proj = Proj(label_dataset.crs)
    l8_proj = Proj(l8_image_datasets[0].crs)
    s1_proj = Proj(s1_image_datasets[0].crs)
    ndvi_proj = Proj(ndvi_image_datasets[0].crs)

    # assuming all images have the same num of bands
    l8_band_count = l8_image_datasets[0].count  
    s1_band_count = s1_image_datasets[0].count
    dem_band_count = dem_image_datasets[0].count
    ndvi_band_count = ndvi_image_datasets[0].count
    band_count = l8_band_count + s1_band_count + dem_band_count + ndvi_band_count
    class_count = len(class_names)
    buffer = math.ceil(tile_height / 2)
  
    while True:
        image_batch = np.zeros((batch_size, tile_height, tile_width, band_count-1)) # take one off because we don't want the QA band
        label_batch = np.zeros((batch_size,class_count))
        b = 0
        while b < batch_size:
            # if we're at the end  of the data just restart
            if i >= len(pixel_locations):
                i=0
            r, c = pixel_locations[i][0]
            dataset_index = pixel_locations[i][1]
            i += 1
            tile = l8_image_datasets[dataset_index].read(list(np.arange(1, l8_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
            if tile.size == 0:
                pass
            elif np.amax(tile) == 0: # don't include if it is part of the image with no pixels
                pass
            elif np.isnan(tile).any() == True or -9999 in tile: 
                # we don't want tiles containing nan or -999 this comes from edges
                # this also takes a while and is inefficient
                pass
            elif tile.shape != (l8_band_count, tile_width, tile_height):
                #print('wrong shape')
                #print(tile.shape)
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
                # set medium developed to high dev
                #tile[tile == 3] = 2
                
                # taking off the QA band
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                
                # L8, S1, and DEM are all the same projection and area otherwise this wouldn't work
                # read in the sentinel-1 data 
                s1_tile = s1_image_datasets[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
               
                # read in the DEM data 
                dem_tile = dem_image_datasets[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                
                # read in the NDVI data 
                ndvi_tile = ndvi_image_datasets[dataset_index].read(list(np.arange(1, ndvi_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                
                if np.isnan(s1_tile).any() == True:
                    pass
                elif np.isnan(dem_tile).any() == True:
                    pass
                elif np.isnan(ndvi_tile).any() == True:
                    pass
                elif np.amax(ndvi_tile) == 0: # don't include if it is part of the image with no pixels
                    pass
                else:
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                    # I don't think we need to normalize since it is already normalized to -1 to 1
                    reshaped_ndvi_tile = reshape_as_image(ndvi_tile)
                    
                    ### get label data
                    # find gps of that pixel within the image
                    (x, y) = l8_image_datasets[dataset_index].xy(r, c)

                    # convert the point we're sampling from to the same projection as the label dataset if necessary
                    if l8_proj != label_proj:
                        x,y = transform(l8_proj,label_proj,x,y)

                    # reference gps in label_image
                    row, col = label_dataset.index(x,y)

                    # find label
                    # image is huge so we need this to just get a single position
                    window = ((row, row+1), (col, col+1))
                    data = label_dataset.read(1, window=window, masked=False, boundless=True)
                    label = data[0,0]
                    # if this label is part of the unclassified area then ignore
                    if label == 0 or np.isnan(label).any() == True:
                        pass
                    else:                   
                        # add label to the batch in a one hot encoding style
                        label = class_to_index[label]
                        label_batch[b][label] = 1
                        image_batch[b] = np.dstack( ( reshaped_tile, reshaped_s1_tile, reshaped_dem_tile, reshaped_ndvi_tile ) )
                        b += 1
        yield (image_batch, label_batch)


    
    
def pixel_generator(l8_image_datasets, s1_image_datasets, dem_image_datasets, ndvi_image_datasets, label_dataset, pixel_locations, batch_size, merge=False):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
    
    class_to_index = dict((
        (11, 0),
        (12, 1),
        (21, 2),
        (22, 3),
        (23, 4),
        (24, 5),
        (31, 6),
        (41, 7),
        (42, 8),
        (43, 9),
        (52, 10),
        (71, 11),
        (81, 12),
        (82, 13),
        (90, 14),
        (95, 15),
        ))

    c = r = 0
    i = 0
    
    label_proj = Proj(label_dataset.crs)
    l8_proj = Proj(l8_image_datasets[0].crs)
    s1_proj = Proj(s1_image_datasets[0].crs)
    ndvi_proj = Proj(ndvi_image_datasets[0].crs)

    # assuming all images have the same num of bands
    l8_band_count = l8_image_datasets[0].count  
    s1_band_count = s1_image_datasets[0].count
    dem_band_count = dem_image_datasets[0].count
    ndvi_band_count = ndvi_image_datasets[0].count
    band_count = l8_band_count + s1_band_count + dem_band_count + ndvi_band_count
    class_count = len(class_names)
  
    while True:
        image_batch = np.zeros((batch_size, band_count-1)) # take one off because we don't want the QA band
        label_batch = np.zeros((batch_size,class_count))
        b = 0
        while b < batch_size:
            # if we're at the end  of the data just restart
            if i >= len(pixel_locations):
                i=0
            r, c = pixel_locations[i][0]
            dataset_index = pixel_locations[i][1]
            i += 1
            tile = l8_image_datasets[dataset_index].read(list(np.arange(1, l8_band_count+1)), window=Window(c, r, 1, 1))
            if tile.size == 0:
                pass
            elif np.amax(tile) == 0: # don't include if it is part of the image with no pixels
                pass
            elif np.isnan(tile).any() == True or -9999 in tile: 
                # we don't want tiles containing nan or -999 this comes from edges
                # this also takes a while and is inefficient
                pass
            elif np.isin(tile[7,:,:], [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352]).any() == True:
                # make sure pixel doesn't contain clouds
                # this is probably pretty inefficient but only checking width x height for each tile
                # read more here: https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1873_US_Landsat_ARD_DFCB_0.pdf
                #print('Found some cloud.')
                #print(tile[7,:,:])
                pass
            else:
                # set medium developed to high dev
                #tile[tile == 3] = 2
                
                # taking off the QA band
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                
                # L8, S1, and DEM are all the same projection and area otherwise this wouldn't work
                # read in the sentinel-1 data 
                s1_tile = s1_image_datasets[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c, r, 1, 1))
               
                # read in the DEM data 
                dem_tile = dem_image_datasets[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c, r, 1, 1))
                # read in the NDVI data 
                ndvi_tile = ndvi_image_datasets[dataset_index].read(list(np.arange(1, ndvi_band_count+1)), window=Window(c, r, 1, 1))
                
                
                if np.isnan(s1_tile).any() == True:
                    pass
                elif np.isnan(dem_tile).any() == True:
                    pass
                elif np.isnan(ndvi_tile).any() == True:
                    pass
                elif np.amax(ndvi_tile) == 0: # don't include if it is part of the image with no pixels
                    pass
                else:
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                    # I don't think we need to normalize since it is already normalized to -1 to 1
                    reshaped_ndvi_tile = reshape_as_image(ndvi_tile)
                    
                    ### get label data
                    # find gps of that pixel within the image
                    (x, y) = l8_image_datasets[dataset_index].xy(r, c)

                    # convert the point we're sampling from to the same projection as the label dataset if necessary
                    if l8_proj != label_proj:
                        x,y = transform(l8_proj,label_proj,x,y)

                    # reference gps in label_image
                    row, col = label_dataset.index(x,y)

                    # find label
                    # image is huge so we need this to just get a single position
                    window = ((row, row+1), (col, col+1))
                    data = label_dataset.read(1, window=window, masked=False, boundless=True)
                    label = data[0,0]
                    # if this label is part of the unclassified area then ignore
                    if label == 0 or np.isnan(label).any() == True:
                        pass
                    else:                   
                        # add label to the batch in a one hot encoding style
                        label = class_to_index[label]
                        label_batch[b][label] = 1
                        image_batch[b] = np.dstack( ( reshaped_tile, reshaped_s1_tile, reshaped_dem_tile, reshaped_ndvi_tile ) )
                        b += 1
        return (image_batch, label_batch)
    
  
        
        
def sk_tile_generator(l8_image_datasets, s1_image_datasets, dem_image_datasets, label_dataset, tile_height, tile_width, pixel_locations, batch_size, merge=False):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
    
    # pixel locations looks like [r, c, dataset_index]
    label_image = label_dataset.read()
    # merge some of the labels
    if merge:
        label_image = merge_classes(label_image)

    c = r = 0
    i = 0
    
    label_proj = Proj(label_dataset.crs)
    l8_proj = Proj(l8_image_datasets[0].crs)
    s1_proj = Proj(s1_image_datasets[0].crs)

    # assuming all images have the same num of bands
    l8_band_count = l8_image_datasets[0].count  
    s1_band_count = s1_image_datasets[0].count
    dem_band_count = dem_image_datasets[0].count
    band_count = l8_band_count + s1_band_count + dem_band_count
    class_count = len(class_names)
    buffer = math.ceil(tile_height / 2)
  
    while True:
        image_batch = np.zeros((batch_size, tile_height * tile_width * (band_count-1)))
        label_batch = np.zeros((batch_size,class_count))
        b = 0
        while b < batch_size:
            # if we're at the end  of the data just restart
            if i >= len(pixel_locations):
                i=0
            r,c = pixel_locations[i][0]
            dataset_index = pixel_locations[i][1]
            i += 1
            tile = l8_image_datasets[dataset_index].read(list(np.arange(1, l8_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
            if np.amax(tile) == 0: # don't include if it is part of the image with no pixels
                pass
            elif np.isnan(tile).any() == True or -9999 in tile: 
                # we don't want tiles containing nan or -999 this comes from edges
                # this also takes a while and is inefficient
                pass
            elif tile.shape != (l8_band_count, tile_width, tile_height):
                #print('wrong shape')
                #print(tile.shape)
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
                # set medium developed to high dev
                #tile[tile == 3] = 2
                
                # taking off the QA band and adjusting the rest
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                
                # L8, S1, and DEM are all the same projection and area otherwise this wouldn't work
                # read in the sentinel-1 data 
                s1_tile = s1_image_datasets[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
               
                # read in the DEM data 
                dem_tile = dem_image_datasets[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                
                if np.isnan(s1_tile).any() == True:
                    pass
                elif np.isnan(dem_tile).any() == True:
                    pass
                else:
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                    ### get label data
                    
                    # find gps of that pixel within the image
                    (x, y) = l8_image_datasets[dataset_index].xy(r, c)

                    # convert the point we're sampling from to the same projection as the label dataset if necessary
                    if l8_proj != label_proj:
                        x,y = transform(l8_proj,label_proj,x,y)

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
                        image_batch[b] = np.dstack( ( reshaped_tile, reshaped_s1_tile, reshaped_dem_tile ) ).flatten()
                        
                        b += 1
        return (image_batch, label_batch)
        

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes, class_dict,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    # convert class_id to class_name using the class_dict
    cover_names = []
    for cover_class in classes:
        cover_names.append(class_dict[cover_class])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    #print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=cover_names, yticklabels=cover_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

"""
TO Run: 

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(np.argmax(sk_label_batch_val, axis=1), pred_index, classes=np.array(list(class_names)),
                      class_dict=class_names)

# Plot normalized confusion matrix
plot_confusion_matrix(np.argmax(sk_label_batch_val, axis=1), pred_index, classes=np.array(list(class_names)),
                      class_dict=class_names,
                      normalize=True)

plt.show()
"""
