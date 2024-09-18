# Databricks notebook source
# List the files in the towerscout_wheel directory to find the wheel file
files = dbutils.fs.ls("/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_wheel/")

# Search for a .whl file in the directory
wheel_file = None
for file in files:
    if file.name.endswith(".whl"):
        wheel_file = file.path
        print(f"Found wheel file: {file.name}")
        break

# Store the DBFS path for the wheel file in a variable for later use
if wheel_file:
    wheel_dbfs_path = f"{wheel_file[5:]}"
    print(f"Wheel file path: {wheel_dbfs_path}")
else:
    print("No wheel file found in the directory.")




# COMMAND ----------

# Use the wheel_dbfs_path variable in the %pip install command
# Make sure that this is run in a new cell
%pip install {wheel_dbfs_path}

# COMMAND ----------

# Import the tsdb package to ensure it is installed and accessible
import tsdb

# Print a confirmation message indicating successful import of the tsdb package
print(f"Package {tsdb.__name__} imported successfully!")

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Users/ac84@cdc.gov/TowerScout/requirements.txt
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from tsdb.preprocessing.utils import calculate_square_root

num = 16
result = calculate_square_root(num)
print(f"The square root of {num} is {result}")

# COMMAND ----------


# Perform the import
from tsdb import preprocessing
print(f"{preprocessing} imported successfuly")



