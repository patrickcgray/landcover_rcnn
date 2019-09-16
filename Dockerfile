# FROM defines the base image
FROM tensorflow/tensorflow:1.13.0rc2-gpu-py3
MAINTAINER Patrick Gray <pgrayobx@gmail.com>

# Disable interactive interface
ARG DEBIAN_FRONTEND=noninteractive

# RUN executes a shell command
# You can chain multiple commands together with &&
# A \ is used to split long lines to help with readability

# Install keras dependencies and geospatial libraries not included in
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install -r requirements.txt

# Enable the widgetsnbextension for notebook
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Install keras
RUN pip --no-cache-dir install --no-deps keras


# Set keras backend to tensorflow by default
ENV KERAS_BACKEND tensorflow

