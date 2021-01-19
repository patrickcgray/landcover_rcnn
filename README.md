# Recurrrent Convolutional Neural Network for Long term Ecosystem Dynamics

## Overview

This repository has the code needed to go along with the Remote Sensing of Environment paper "Temporally Generalizable Land Cover Classification: A Recurrent Convolutional Neural Network Unveils Major Coastal Change through Time." available at DOI: TBD. This work uses an RCNN for landcover classification of Landsat imagery across 20 years in eastern North Carolina, USA, and uncovers major change in low lying areas.

All data, both preprocessed and original raw data, necessary to reproduce this work is available on the Duke Digital Repository at DOI: TBD

## Installation

Environmental setup on Azure is here: https://docs.google.com/document/d/1IMttkI-rjG1J9-lv65mQDXFch5rYjkrMUsIifOWs6Ig/edit

## Code Notebooks

To process and visualize data:
* rcnn/rnn_data_processing.ipynb
* cluster_viz_inspect.ipynb
* labeling_data_prep.ipynb

To train the model:
* rcnn/model_search_framework.ipynb
* rcnn/model_search_framework_scikit.ipynb
* 

To inspect models and predict (using the pretrained model or your own version):
* rcnn/full_tile_prediction.ipynb
* rcnn/inspecting_model_accuracy_results.ipynb
* rcnn/lc_classification_analysis


