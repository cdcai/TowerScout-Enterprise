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

from pathlib import Path
import subprocess
import shutil


def package_and_move_wheel(source_path: str, target_volume_path: str, overwrite: bool=False):
    """
    Packages the Python code at `source_path` and moves the resulting wheel file to `target_volume_path`.

    Args:
      source_path (str): The path to the Python code to package.
      target_volume_path (str): The path to the volume to move the wheel file to.
    """
    # Step 1.1: Ensure temp directory on driver node is empty
    temp_dir = Path("/tmp/yolov5_package")
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
    
    # Step 1.2: Copy files to temp directory on driver node
    dbutils.fs.cp(f"dbfs:{source_path}", f"file:{temp_dir}", recurse=True)

    # Confirm files are copied
    if not temp_dir.exists():
      raise FileNotFoundError(f"Temporary directory {temp_dir} not found after copy.")

    # Step 2: Build the wheel in the temporary directory
    subprocess.run(["python", "setup.py", "bdist_wheel"], check=True, cwd=temp_dir)

    # Step 3: Locate the wheel file
    dist_path = temp_dir / "dist"
    wheel_files = list(dist_path.glob("*.whl"))
    if not wheel_files:
      raise FileNotFoundError("No .whl file found in the dist directory.")
    
    wheel_name = wheel_files[0].name

    # Step 4: Check if wheel file already exists in target volume
    target_file = f"{target_volume_path}/{wheel_name}"
    if not overwrite and Path(target_file).exists():
      raise FileExistsError(f"Wheel file {wheel_name} already exists in target volume.")
    
    # Step 5: Move wheel file from local driver node to target volume
    dbutils.fs.cp(
        f"file:{dist_path}/{wheel_name}",
        f"dbfs:{target_volume_path}/{wheel_name}"
    )

# COMMAND ----------

source_path = "/Volumes/edav_dev_csels/towerscout/misc/yolov5"
target_path = "/Volumes/edav_dev_csels/towerscout/wheel/"
package_and_move_wheel(source_path, target_path, overwrite=True)

# COMMAND ----------

Path(f"/Volumes/edav_dev_csels/towerscout/wheel/tsdb-0.1.0-py3-none-any.whl").exists()

# COMMAND ----------

temp_dir = Path("/tmp/yolov5_package/")
temp_dir.exists()

# COMMAND ----------

f"file:{temp_dir}"

# COMMAND ----------

import subprocess

# COMMAND ----------

source_path = "/Volumes/edav_dev_csels/towerscout/misc/yolov5/"
temp_dir = "/tmp/yolov5_package"
dbutils.fs.cp(f"dbfs:{source_path}", f"file:{temp_dir}", recurse=True)
subprocess.run(["python", "setup.py", "bdist_wheel"], check=True, cwd=temp_dir)

# COMMAND ----------

from pathlib import Path

# COMMAND ----------

dist_path = Path(temp_dir) / "dist"

wheel_file_name = [f.name for f in dist_path.glob("*.whl")][0]
wheel_path = dist_path / wheel_file_name

# COMMAND ----------

dbutils.fs.cp(
    f"file:{wheel_path}",
    f"dbfs:/Volumes/edav_dev_csels/{schema}/wheel/{wheel_file_name}"
)

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
