#!/bin/bash

# Docker image name
IMAGE_NAME="aicomp-pandoc-report"

# Check if Docker image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Docker image '$IMAGE_NAME' not found. Building it now..."
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build Docker image."
        exit 1
    fi
    echo "Docker image built successfully."
else
    echo "Docker image '$IMAGE_NAME' already exists."
fi

# Create _build directory if it doesn't exist
mkdir -p _build

# Run pandoc in Docker container
echo "Running pandoc to generate report.pdf..."
docker run --rm \
    -v "/${PWD}:/root" \
    --entrypoint pandoc \
    $IMAGE_NAME \
    src/report.md --defaults defaults.yaml -o _build/report.pdf

if [ $? -eq 0 ]; then
    echo "Success! Report generated at _build/report.pdf"
else
    echo "Error: Failed to generate report."
    exit 1
fi

