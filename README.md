## Temporally Generalizable Land Cover Classification: A Recurrent Convolutional Neural Network Unveils Major Coastal Change through Time

## Overview

This repository has the code needed to go along with the Remote Sensing of Environment paper "Temporally Generalizable Land Cover Classification: A Recurrent Convolutional Neural Network Unveils Major Coastal Change through Time." available at https://www.mdpi.com/2072-4292/13/19/3953. This work uses an RCNN for landcover classification of Landsat imagery across 20 years in eastern North Carolina, USA, and uncovers major change in low lying areas.

All data, both preprocessed and original raw data, necessary to reproduce this work is available on the Duke Digital Repository at DOI: TBD

## Installation

Environmental setup on Azure is here: https://docs.google.com/document/d/1IMttkI-rjG1J9-lv65mQDXFch5rYjkrMUsIifOWs6Ig/edit

## Code Notebooks

#### To process and visualize data:
* rcnn/rnn_data_processing.ipynb
  * this takes the raw landsat data and processes it, stacks the bands, and puts it in the format needed for model development
  * this notebook also visualizes cloud masks and processed images
* labeling_data_prep.ipynb
  * this notebook generates the data that will be used for train, validation, and test 
  * the output and validated train/val/test is available in the shapefile/ directory
* cluster_viz_inspect.ipynb
  * this notebook takes the processed data and visualizes the distribution, clusters the imagery, and generates spectral signature figures


#### To train the model:
* rcnn/model_search_framework.ipynb
  * this trains all deep learning models
* rcnn/model_search_framework_scikit.ipynb
  * this trains all traditional machine learning models

#### To inspect models and predict (using the pretrained model or your own version):
* rcnn/full_tile_prediction.ipynb
  * this uses the trained model to predict on an entire stacked landsat tile
* rcnn/inspecting_model_accuracy_results.ipynb
  * this inspects modela accuracy and generates a number of eval figures
* rcnn/lc_classification_analysis
  * this notebook analyzes the change that occured between 1989 and 2001


