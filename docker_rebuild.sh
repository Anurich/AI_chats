#!/bin/bash

# Assign arguments to variables (optional)
CONTAINER_NAME=$1
IMAGE_NAME=$2
GIT_REPO_URL="https://github.com/Anurich/AI_chats"

# Hardcoded credentials
GIT_USERNAME="Anurich"
GIT_TOKEN="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"  # Replace with your actual token

# If container name is provided, stop and remove the container
if [ ! -z "$CONTAINER_NAME" ]; then
  echo "Stopping and removing container: $CONTAINER_NAME"
  sudo docker stop $CONTAINER_NAME
  sudo docker rm $CONTAINER_NAME
else
  echo "No container name provided. Skipping container stop and removal."
fi

# If image name is provided, remove the image
if [ ! -z "$IMAGE_NAME" ]; then
  echo "Removing image: $IMAGE_NAME"
  sudo docker rmi -f $IMAGE_NAME
else
  echo "No image name provided. Skipping image removal."
fi

# Build new image
echo "Building new image: ai-server-tool"
sudo docker build -t ai-server-tool /path/to/your/repo

# Run the new container
echo "Running new container on port 4200"
sudo docker run -d -p 4200:4200 ai-server-tool

echo "Script completed."
