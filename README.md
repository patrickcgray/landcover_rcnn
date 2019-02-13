# florence_mapping

Code for mapping coastal in order to detect storm driven changes in landcover.

Highly recommended to use the Dockerfile to set up your environment.

In order to build your docker image:

`docker build -t <image name> .`

Running this image:

`docker run --name <container name> --runtime=nvidia -p 8888:8888 -p 6006:6006 -v ~/:/host -v /srv/deepcoast:/deep_data -it <image name>`

Running jupyter in this docker container:

`jupyter notebook --allow-root /host --ip 0.0.0.0`
