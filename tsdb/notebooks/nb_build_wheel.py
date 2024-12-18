# Databricks notebook source
# Purpose: Check if the global view 'global_temp_towerscout_configs' exists and extract configuration values from it. 
# If the view does not exist, exit the notebook with an error message.

# Check if the global view exists
if spark.catalog._jcatalog.tableExists("global_temp.global_temp_towerscout_configs"):
    # Query the global temporary view and collect the first row
    result = spark.sql("SELECT * FROM global_temp.global_temp_towerscout_configs").collect()[0]
    
    # Extract values from the result row
    env = result['env']
    catalog = result['catalog_name']
    schema = result['schema_name']
    debug_mode = result['debug_mode'] == "true"
    unit_test_mode = result['unit_test_mode'] == "true"
else:
    # Exit the notebook with an error message if the global view does not exist
    dbutils.notebook.exit("Global view 'global_temp_towerscout_configs' does not exist, make sure to run the utils notebook")

# COMMAND ----------

env = "dev"
catalog = "edav_dev_csels"
schema = "towerscout"
debug_mode = False
unit_test_mode = False

# COMMAND ----------

from tsdb.utils.fs import package_and_move_wheel

# COMMAND ----------

source_path = "/Volumes/edav_dev_csels/towerscout/misc/yolov5"
target_path = "/Volumes/edav_dev_csels/towerscout/wheel/"
package_and_move_wheel(source_path, target_path, dbutils, overwrite=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Old Version - Delete when done

# COMMAND ----------

!python /Workspace/Users/ac84@cdc.gov/TowerScout/setup.py bdist_wheel --dist-dir /tmp/
import os
# Step 2: Find the dynamically generated wheel file in /tmp
wheel_files = [f for f in os.listdir('/tmp') if f.endswith('.whl')]
if len(wheel_files) != 1:
    raise RuntimeError("Expected exactly one wheel file in /tmp/ directory.")
wheel_file = wheel_files[0]
if debug_mode:
    print(f"Generated wheel file: {wheel_file}")


# COMMAND ----------

# Step 3: Copy or move the wheel file from /tmp to DBFS
try:
    dbutils.fs.cp(
        f"file:/tmp/{wheel_file}",
        f"dbfs:/Volumes/edav_dev_csels/{schema}/towerscout_wheel/{wheel_file}"
    )
    print(f"Copied {wheel_file} to DBFS.")
except Exception as e:
    if debug_mode:
        print(f"Error while copying file to DBFS: {e}")

# COMMAND ----------

# Step 4: Verify the file exists in DBFS
dbutils.fs.ls(f"/Volumes/edav_dev_csels/{schema}/towerscout_wheel")

# Step 5: Delete the local wheel file from /tmp
!rm /tmp/{wheel_file}
if debug_mode:
    print(f"Deleted local wheel file: {wheel_file}")
