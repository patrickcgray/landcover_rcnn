import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np
from pyproj import Proj, transform
from shapely.geometry import Polygon
from rasterio.windows import Window
import random
import math

class VisualizeData:
    
    def __init__(self, landsat_datasets, label_dataset):
        self.landsat = landsat_datasets
        self.labels = label_dataset
        self.label_proj = Proj(label_dataset.crs)
        self.open_figs = list()
        self.colors = dict((
    (11, (0,0,255)), #water ~ blue
(12, (0,0,255)), #snow ~ white
(21, (255,0,0)), #open space developed ~ red
(22, (50,0,0)), # low intensity developed ~ darker red
(23, (50,0,0)), # medium intensity developed ~ darker darker red
(24, (50,0,0)), # high intensity developed ~ darker darker darker red
(31, (153,76,0)), # barren land ~ dark orange
(41, (0,204,0)), # deciduous forest ~ green
(42, (0,153,0)), # evergreen forest ~ darker green
(43, (0,102,0)), # mixed forest ~ darker darker green
(52, (153,0,76)), #schrub ~ dark pink
(71, (255,153,71)), # grass land ~  orange
(81, (204,204,0)),#pasture ~ yellowish
(82, (153,153,0)),#cultivated land ~ darker yellow
(90, (0,255,255)), #woody wetland ~ aqua
(95, (0,102,102)), #emergent herbaceous wetlands ~ darker aqua
))
        self.class_names = dict((
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
   
    def view_landsat(self, landsat_index):
        image_dataset = self.landsat[landsat_index]
        full_img = self.landsat[landsat_index].read()
        colors_reshaped = self.__normalize_rgb(full_img)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(colors_reshaped)
        ax.set_title("RGB in matplotlib imshow")
    
    def view_labels(self, landsat_index):
        masked_label_image, raster_poly = self.__make_a_label_mask(landsat_index)
        ax = self.__plot_a_tile(masked_label_image[0,:,:], colors=self.colors)
        
    def print_a_tile(self, landsat_index, class_index, tile_size, middle=False):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        buffer = math.ceil(tile_size / 2)
        masked_label_image, raster_poly = self.__make_a_label_mask(landsat_index)
        rows,cols = np.where(masked_label_image[0] == class_index)
        all_locations = list(zip(rows,cols))
        if len(all_locations) == 0:
            return
        loc_index = random.randint(0, len(all_locations)-1)
        tile_loc = all_locations[loc_index]
        #r, c is in label mask pixel locations
        r_label, c_label = tile_loc
        r_landsat, c_landsat = self.__transform_to_l8(landsat_index, r_label, c_label, raster_poly)
        label_tile = self.__read_from_raster(self.labels, c_label+raster_poly.bounds[0]-buffer, r_label+raster_poly.bounds[1]-buffer, tile_size)
        landsat_tile = self.__read_from_raster(self.landsat[landsat_index], c_landsat-buffer, r_landsat-buffer, tile_size)
        landsat_tile_normalized = self.__normalize_rgb(landsat_tile)
        ax_c = self.__plot_a_tile(landsat_tile_normalized)
        ax_c.set_title(self.class_names[class_index])
        ax_l = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors)
        ax_l.set_title(self.class_names[class_index])
        if middle:
            ax_m = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors, middle = True)
            ax_m.set_title(self.class_names[class_index])
            
            
    #takes landsat row_col
    def print_chosen_tile(self, landsat_index, tile_size, row_col, middle=False):
        buffer = math.ceil(tile_size / 2)
        r_landsat, c_landsat = row_col
        r_label, c_label = self.__transform_to_label(landsat_index, r_landsat, c_landsat)
        class_index = self.__get_label(r_label, c_label)
        label_tile = self.__read_from_raster(self.labels, c_label-buffer, r_label-buffer, tile_size)
        landsat_tile = self.__read_from_raster(self.landsat[landsat_index], c_landsat-buffer, r_landsat-buffer, tile_size)
        landsat_tile_normalized = self.__normalize_rgb(landsat_tile)
        ax_c = self.__plot_a_tile(landsat_tile_normalized)
        ax_c.set_title(self.class_names[class_index])
        ax_l = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors)
        ax_l.set_title(self.class_names[class_index])
        if middle:
            ax_m = self.__plot_a_tile(label_tile[0,:,:], colors = self.colors, middle = True)
            ax_m.set_title(self.class_names[class_index])
        
        
    def __get_label(self, r_label, c_label):
        window = ((r_label, r_label+1), (c_label, c_label+1))
        data = self.labels.read(1, window=window, masked=False, boundless=True)
        label = data[0,0]
        return label
    def __read_from_raster(self, raster, column_start, row_start, tile_size):
        w = raster.read(window=Window(column_start, row_start, tile_size, tile_size))
        return w
    def __plot_a_tile(self, tile, colors = None, middle = False):
        fig, ax = plt.subplots(figsize=(10, 10))
        self.open_figs.append(fig)
        if colors == None:
             ax.imshow(tile)
        else:
            colored_label_img = np.zeros((tile.shape[0], tile.shape[1], 3))
            center_px = math.ceil(tile.shape[0]/2)
            for h in range(tile.shape[0]):
                for w in range(tile.shape[1]):
                    if h == center_px and w == center_px and middle:
                        colored_label_img[h][w] = (0,0,0)
                    if tile[h][w] not in self.class_names:
                         colored_label_img[h][w] = (0,0,0)
                    else:
                        colored_label_img[h][w] = np.divide(colors[tile[h][w]], 255)
            print("sanity check print middle pixel {}".format(tile[center_px][center_px]))
            ax.imshow(colored_label_img)
        return ax
    
    
    
    def close_figs(self):
        for fig in self.open_figs:
            plt.close(fig)
    def __transform_to_l8(self, landsat_index, r_label, c_label, raster_poly):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        x,y = self.labels.xy(r_label+raster_poly.bounds[1],c_label+raster_poly.bounds[0])
        # go from label projection into landsat projection
        x,y = transform(self.label_proj, l8_proj,x,y)
        # convert from landsat geographic space into row col
        r_landsat,c_landsat = self.landsat[landsat_index].index(x,y) 
        return r_landsat, c_landsat
   
    def __transform_to_label(self, landsat_index, r_landsat, c_landsat):
        l8_proj = Proj(self.landsat[landsat_index].crs)
        x,y = self.landsat[landsat_index].xy(r_landsat,c_landsat)
        x,y = transform(l8_proj, self.label_proj,x,y)
        r_label,c_label = self.labels.index(x,y) 
        return r_label, c_label
        
    def __normalize_rgb(self, tile):
        tile = tile[[3, 2, 1],:,:].astype(np.float64)
        max_val = 4000
        min_val = 0
        # Enforce maximum and minimum values
        tile[tile[:, :, :] > max_val] = max_val
        tile[tile[:, :, :] < min_val] = min_val
        for b in range(tile.shape[0]):
            tile[b, :, :] = tile[b, :, :] * 1 / (max_val - min_val)
        tile_reshaped = reshape_as_image(tile)
        return tile_reshaped
        
    def __make_a_label_mask(self, landsat_index):
        image_dataset = self.landsat[landsat_index]
        raster_points = image_dataset.transform * (0, 0), image_dataset.transform * (image_dataset.width, 0), image_dataset.transform * (image_dataset.width, image_dataset.height), image_dataset.transform * (0, image_dataset.height)
        l8_proj = Proj(image_dataset.crs)
        new_raster_points = []
        # convert the raster bounds from landsat into label crs
        for x,y in raster_points:
            x,y = transform(l8_proj,self.label_proj,x,y)
            # convert from crs into row, col in label image coords
            row, col = self.labels.index(x, y)
            # don't forget row, col is actually y, x so need to swap it when we append
            new_raster_points.append((col, row))
        # turn this into a polygon
        raster_poly = Polygon(new_raster_points)
        # Window.from_slices((row_start, row_stop), (col_start, col_stop))
        masked_label_image = self.labels.read(window=Window.from_slices((int(raster_poly.bounds[1]), int(raster_poly.bounds[3])), (int(raster_poly.bounds[0]), int(raster_poly.bounds[2]))))
        return masked_label_image, raster_poly
        
        
        
        
        