# Databricks notebook source
# Install necessary tools
%pip install wheel setuptools

# COMMAND ----------

!python setup.py bdist_wheel --dist-dir /tmp/
import os
# Step 2: Find the dynamically generated wheel file in /tmp
wheel_files = [f for f in os.listdir('/tmp') if f.endswith('.whl')]
if len(wheel_files) != 1:
    raise RuntimeError("Expected exactly one wheel file in /tmp/ directory.")
wheel_file = wheel_files[0]
print(f"Generated wheel file: {wheel_file}")


# COMMAND ----------

# Step 3: Copy or move the wheel file from /tmp to DBFS
try:
    dbutils.fs.cp(
        f"file:/tmp/{wheel_file}",
        f"dbfs:/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_wheel/{wheel_file}"
    )
    print(f"Copied {wheel_file} to DBFS.")
except Exception as e:
    print(f"Error while copying file to DBFS: {e}")

# COMMAND ----------

# Step 4: Verify the file exists in DBFS
dbutils.fs.ls("/Volumes/edav_dev_csels/towerscout_test_schema/towerscout_wheel")

# Step 5: Delete the local wheel file from /tmp
!rm /tmp/{wheel_file}
print(f"Deleted local wheel file: {wheel_file}")
