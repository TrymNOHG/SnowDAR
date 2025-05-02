#!/bin/sh

# Script to process the datasets before training the models.
# By Nicolai H. B. and Trym H. G.

# Exits the script immediately on an error
set -e

# Ensure data is in the expected directories.
if [ ! -d "Poles/rgb" ]; then
  echo "Error: Directory Poles/rgb/ not found."
  exit 1
fi

if [ ! -d "Poles/lidar" ]; then
  echo "Error: Directory Poles/lidar/ not found."
  exit 1
fi

echo "--- Datasets found --- "

# rename Poles/lidar/combined_color to Poles/lidar/images for regularity
mv "Poles/lidar/combined_color" "Poles/lidar/images"


echo "Cropping datasets: process/crop_img.py ..."
python3 process/crop_img.py

echo "Cuting images in dataset in half: process/crop_img.py ..."
python3 process/cut_datasets.py

echo "DEBUG: Drawing bounding boxes: viz/draw_bb.py..."
python3 viz/draw_bb.py

# Now we must rename the datasets to images/ and labels/ as that is what YOLO expects
mv "Poles/new_lidar/images" "Poles/new_lidar/images_old"
mv "Poles/new_lidar/labels" "Poles/new_lidar/label_old"
mv "Poles/new_lidar/split_images" "Poles/new_lidar/images"
mv "Poles/new_lidar/split_labels" "Poles/new_lidar/labels"

mv "Poles/new_rgb/images" "Poles/new_rgb/images_old"
mv "Poles/new_rgb/labels" "Poles/new_rgb/label_old"
mv "Poles/new_rgb/split_images" "Poles/new_rgb/images"
mv "Poles/new_rgb/split_labels" "Poles/new_rgb/labels"

echo "Processed datasets now under Poles/new_rgb/ and Poles/new_lidar/"


./generate_yaml.sh

echo "--- Dataset processing complete --- "
