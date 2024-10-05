#!/bin/bash

# Check if sufficient arguments are provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <container_name> <image_name>"
  exit 1
fi

# Assign arguments to variables
CONTAINER_NAME=$1
IMAGE_NAME=$2

# Stop the container if it's running
sudo docker stop $CONTAINER_NAME

# Remove the container
sudo docker rm $CONTAINER_NAME

# Remove the specified image
sudo docker rmi -f $IMAGE_NAME

# Build new image
sudo docker build -t ai-server-tool .

# Run the new container
sudo docker run -d -p 4200:4200 ai-server-tool

echo "Rebuilt and restarted container with image $IMAGE_NAME"
