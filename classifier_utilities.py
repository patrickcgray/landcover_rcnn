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


indexed_dictionary = dict((
(0, "Water"),
(1, "Snow/Ice"),
(2, "Open Space Developed"),
(3, "Low Intensity Developed"),
(4, "Medium Intensity Developed"),
(5, "High Intensity Developed"),
(6, "Barren Land"),
(7, "Deciduous Forest"),
(8, "Evergreen Forest"),
(9, "Mixed Forest"),
#(51, "Dwarf Scrub/Shrub - ALASKA"),
(10, "Scrub/Shrub"),
(11, "Grassland / Herbaceous"),
#(72, "Sedge / Herbaceous - ALASKA"),
#(73, "Lichen / Herbaceous - ALASKA"),
#(74, "Moss - ALASKA"),
(12, "Pasture/Hay"),
(13, "Cultivated Land"),
(14, "Woody Wetland"),
(15, "Emergent Herbaceous Wetlands"),
))


      
def pixel_balance(pixels, landsat_datasets, label_dataset,merge=True):
    predictions = np.zeros(len(indexed_dictionary))
    label_proj = Proj(label_dataset.crs)
    l8_proj = Proj(landsat_datasets[0].crs)
    for pixel in pixels:
        r, c = pixel[0]
        dataset_index = pixel[1]
        (x, y) = landsat_datasets[dataset_index].xy(r, c)
        # convert the point we're sampling from to the same projection as the label dataset if necessary
        if l8_proj != label_proj:
            x,y = transform(l8_proj,label_proj,x,y)
        # reference gps in label_image
        row, col = label_dataset.index(x,y)
        # find label
        # image is huge so we need this to just get a single position
        window = ((row, row+1), (col, col+1))
        data = label_dataset.read(1, window=window, masked=False, boundless=True)
        if merge:
            data = merge_classes(data)
        label = data[0,0]
        label = class_to_index[label]
        predictions[label] +=1
    return predictions

def train_val_test_split(pixels, train_val_ratio, val_test_ratio):
    random.shuffle(pixels)
    train_px = pixels[:int(len(pixels)*train_val_ratio)]
    val_pixels = pixels[int(len(pixels)*train_val_ratio):]
    val_px = val_pixels[:int(len(val_pixels)*val_test_ratio)]
    test_px = val_pixels[int(len(val_px)*val_test_ratio):]
    return (train_px, val_px, test_px)

def merge_classes(y):
    # medium intensity and high intensity
    #y[y == 2] = 4
    y[y == 22] = 23
    y[y == 24] = 23
    return(y)


def gen_balanced_pixel_locations(image_datasets, train_count, tile_size, label_dataset, merge=True):
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
                for r,c in all_locations[:math.ceil(7*points_per_class)]:
                # convert label row and col into label geographic space
                    x,y = label_dataset.xy(r+raster_poly.bounds[1],c+raster_poly.bounds[0])
                # go from label projection into landsat projection
                    x,y = transform(label_proj, l8_proj,x,y)
                # convert from landsat geographic space into row col
                    r,c = image_dataset.index(x,y)
                    l8_points.append((r,c))
                class_px_index = [index] * len(l8_points)
                class_px = list(zip(l8_points, class_px_index))
                l8_points = delete_black_tiles(image_datasets, tile_size, class_px, max_size = points_per_class)
                all_points_per_image += l8_points[:points_per_class]
        train_pixels += all_points_per_image
    random.shuffle(train_pixels)
    return (train_pixels)


def gen_pixel_locations(image_datasets, pixel_count, tile_size):
    pixels = []
    buffer = math.ceil(tile_size/2)
    count_per_dataset = math.ceil(pixel_count / len(image_datasets))
    for index, image_dataset in enumerate(image_datasets):
        #randomly pick `count` num of pixels from each dataset
        img_height, img_width = image_dataset.shape
        rows = range(0+buffer, img_height-buffer)
        columns = range(0+buffer, img_width-buffer)
        points = random.sample(set(itertools.product(rows, columns)), math.ceil(10*count_per_dataset))
        dataset_index_list = [index] * count_per_dataset
        dataset_pixels = list(zip(points, dataset_index_list))
        dataset_pixels = delete_black_tiles(image_datasets, tile_size, dataset_pixels, max_size = count_per_dataset)
        pixels += dataset_pixels
    return (pixels)
        
def tile_generator(l8_image_datasets, s1_image_datasets, dem_image_datasets, label_dataset, tile_height, tile_width, pixel_locations, batch_size, merge=True, reshape=False, verbose = False):
    ### this is a keras compatible data generator which generates data and labels on the fly 
    ### from a set of pixel locations, a list of image datasets, and a label dataset
    c = r = 0
    i = 0   
    label_proj = Proj(label_dataset.crs)
    l8_proj = Proj(l8_image_datasets[0].crs)
    s1_proj = Proj(s1_image_datasets[0].crs)

    # assuming all images have the same num of bands
    l8_band_count = l8_image_datasets[0].count  
    s1_band_count = s1_image_datasets[0].count
    dem_band_count = dem_image_datasets[0].count
    band_count = l8_band_count# + s1_band_count + dem_band_count
    class_count = len(class_names)
    buffer = math.floor(tile_height / 2)
  
    while True:
        image_batch = np.zeros((batch_size, tile_height, tile_width, band_count-1)) # take one off because we don't want the QA band
        label_batch = np.zeros((batch_size,class_count))
        b = 0
        while b < batch_size:
            # if we're at the end  of the data just restart
            if i >= len(pixel_locations):
                i=0
            r, c = pixel_locations[i][0]
            if verbose:
                print("({},{})".format(r,c))
            dataset_index = pixel_locations[i][1]
            i += 1
            tile = l8_image_datasets[dataset_index].read(list(np.arange(1, l8_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                # L8, S1, and DEM are all the same projection and area otherwise this wouldn't work
                # read in the sentinel-1 data 
            s1_tile = s1_image_datasets[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
                # read in the DEM data 
            dem_tile = dem_image_datasets[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
            if np.isnan(s1_tile).any():
                print("u need to check before")
                pass
            elif np.isnan(dem_tile).any():
                print("u need to check before")
                pass
            else:
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
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
                window = ((row, row+1), (col, col+1))
                data = label_dataset.read(1, window=window, masked=False, boundless=True)
                if merge:
                    data = merge_classes(data)
                label = data[0,0]
                if np.isnan(reshaped_s1_tile).any() or np.isinf(reshaped_s1_tile).any():
                    print("s1 reshaped")
                    pass
                elif np.isnan(reshaped_dem_tile).any() or np.isinf(reshaped_dem_tile).any():
                    print("dem reshaped")
                    pass             
                elif label == 0 or np.isnan(label).any() or label not in class_to_index:
                    print("label:{}".format(label))
                    pass
                else:                   
                    label = class_to_index[label]
                    label_batch[b][label] = 1
                    image_batch[b] = reshaped_tile #np.dstack((reshaped_tile, reshaped_s1_tile, reshaped_dem_tile))    
                    b += 1
        yield (image_batch, label_batch) 


def load_data():
    label_dataset = rasterio.open('/deep_data/NLCD/NLCD_2016_Land_Cover_L48_20190424.img')

    l8_image_paths = [
    '/deep_data/processed_landsat/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined.tif',  
    '/deep_data/processed_landsat/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined.tif',
    '/deep_data/processed_landsat/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined.tif'
    ]

    s1_image_paths = [
    '/deep_data/sentinel_sar/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined/aligned-LC08_CU_027012_20170907_20181121_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined/aligned-LC08_CU_028011_20170907_20181130_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined/aligned-LC08_CU_028012_20171002_20171019_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined/aligned-LC08_CU_028012_20171103_20190429_C01_V01_SR_combined_SAR.tif',
    '/deep_data/sentinel_sar/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined/aligned-LC08_CU_029011_20171018_20190429_C01_V01_SR_combined_SAR.tif',
   ]

    dem_image_paths = [
    '/deep_data/sentinel_sar/LC08_CU_027012_20170907_20181121_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-79.69001_33.95762_-77.7672_35.51886__4500X4631_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028011_20170907_20181130_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-77.7672_35.00779_-75.79042_36.58923__4500X4262_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171002_20171019_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-79.69001_33.95762_-77.7672_35.51886__4500X4631_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_028012_20171103_20190429_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-78.07896_33.69485_-76.14021_35.27466__4500X4248_ShowLogo_False_tiff_depth=32f.tiff',
    '/deep_data/sentinel_sar/LC08_CU_029011_20171018_20190429_C01_V01_SR_combined_dem/aligned-wms_DEM_EPSG4326_-76.14021_34.71847_-74.14865_36.318__4500X4408_ShowLogo_False_tiff_depth=32f.tiff',
    ]


    landsat_datasets = []
    for fp in l8_image_paths:
        landsat_datasets.append(rasterio.open(fp))
    sentinel_datasets = []
    for fp in s1_image_paths:
        sentinel_datasets.append(rasterio.open(fp))
    dem_datasets = []
    for fp in dem_image_paths:
        dem_datasets.append(rasterio.open(fp))
    return (landsat_datasets, sentinel_datasets, dem_datasets, label_dataset)      
        
        
        
        
def delete_black_tiles(landsat, s1, dem tile_size, pixels, max_size = None):
    buffer = math.floor(tile_size / 2)
    l8_band_count = landsat[0].count  
    new_pixels = []
    for pixel in pixels:
        r, c = pixel[0]
        dataset_index = pixel[1]
        tile = landsat[dataset_index].read(list(np.arange(1, l8_band_count+1)),window=Window(c-buffer, r-buffer, tile_size, tile_size))
        s1_tile = s1[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
        dem_tile = dem[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
        if np.isnan(tile).any() == True or -9999 in tile or tile.size == 0 or np.amax(tile) == 0 or np.isin(tile[7,:,:], [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352]).any() == True or tile.shape != (l8_band_count, tile_size, tile_size):
            pass
        elif np.isnan(s1_tile).any():
            pass
        elif np.isnan(dem_tile).any():
            pass
        else:
            new_pixels.append(pixel)
            if max_size != None and len(new_pixels) == max_size:
                return new_pixels
    return new_pixels    
        

def evaluate_cnn(model, landsat_datasets, sentinel_datasets, dem_datasets, label_dataset, tile_size, pixels, batch_size=64, old=False):
    if old:
        predictions = model.predict_generator(generator= nw.tile_generator(landsat_datasets, sentinel_datasets, dem_datasets, label_dataset, tile_size, tile_size, pixels, batch_size), steps=len(pixels) // batch_size, verbose=1)
        eval_generator = nw.tile_generator(landsat_datasets, sentinel_datasets, dem_datasets, label_dataset, tile_size, tile_size, pixels, batch_size=1)
    else:
        predictions = model.predict_generator(generator= tile_generator(landsat_datasets, sentinel_datasets, dem_datasets, label_dataset, tile_size, tile_size, pixels, batch_size), steps=len(pixels) // batch_size, verbose=1)
        eval_generator = tile_generator(landsat_datasets, sentinel_datasets, dem_datasets, label_dataset, tile_size, tile_size, pixels, batch_size=1, verbose = False)
    labels = np.empty(predictions.shape)
    count = 0
    while count < len(labels):
        image_b, label_b = next(eval_generator)
        #print(np.argmax(label_b))
        labels[count] = label_b
        count += 1
    label_index = np.argmax(labels, axis=1)     
    pred_index = np.argmax(predictions, axis=1)
    print(label_index)
    print(pred_index)
    #print(pred_index)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(label_index, pred_index, classes=np.array(list(indexed_dictionary)),
                      class_dict=indexed_dictionary)
   # Plot normalized confusion matrix
    plot_confusion_matrix(label_index, pred_index, classes=np.array(list(indexed_dictionary)),
                      class_dict=indexed_dictionary,
                      normalize=True)
    count = 0
    for i in range(len(label_index)):
        if(label_index[i] == pred_index[i]):
            count+=1
    print("ACCURACY")
    print(count/len(label_index))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
def pixel_generator(l8_image_datasets, s1_image_datasets, dem_image_datasets, label_dataset, pixel_locations, batch_size, merge=False):
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
                
                # taking off the QA band and adjusting the rest
                tile = tile[0:7]
                # reshape from raster format to image format and standardize according to image wide stats
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5
                                
                # L8, S1, and DEM are all the same projection and area otherwise this wouldn't work
                # read in the sentinel-1 data 
                s1_tile = s1_image_datasets[dataset_index].read(list(np.arange(1, s1_band_count+1)), window=Window(c,r,1,1))
               
                # read in the DEM data 
                dem_tile = dem_image_datasets[dataset_index].read(list(np.arange(1, dem_band_count+1)), window=Window(c,r,1,1))
                
                if np.isnan(s1_tile).any() == True:
                    pass
                elif np.isnan(dem_tile).any() == True:
                    pass
                else:
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_s1_tile = (reshape_as_image(s1_tile)  - 0.10) / 0.088
                    # reshape from raster format to image format and standardize according to image wide stats
                    reshaped_dem_tile = (reshape_as_image(dem_tile)  - 31) / 16.5
                    
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
                        image_batch[b] = np.dstack( ( reshaped_tile, reshaped_s1_tile, reshaped_dem_tile ) )
                        
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
