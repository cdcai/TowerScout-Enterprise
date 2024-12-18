# Databricks notebook source
import zipfile  # ZIP files
import os  # os
import shutil  # file operations
import random  # random numbers
from math import floor  #
import xml.etree.ElementTree as ET  # XML parsing

# COMMAND ----------

# Paths
zip_file_path = '/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/raw-training-data/towerscout-training-data.zip'  # Path to zip file
extracted_dir = '/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data'  # Temporary directory for extraction

# COMMAND ----------

# Output directories for images and labels
dataset_dir = '/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_data'
image_train_dir = os.path.join(dataset_dir, 'images', 'train')
image_val_dir = os.path.join(dataset_dir, 'images', 'val')
image_test_dir = os.path.join(dataset_dir, 'images', 'test')
label_train_dir = os.path.join(dataset_dir, 'labels', 'train')
label_val_dir = os.path.join(dataset_dir, 'labels', 'val')
label_test_dir = os.path.join(dataset_dir, 'labels', 'test')

# COMMAND ----------

# Create directories if they don't exist
os.makedirs(image_train_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(image_test_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)
os.makedirs(label_test_dir, exist_ok=True)

# COMMAND ----------

# Step 1: Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
