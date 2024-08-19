# Databricks notebook source
import sys

git_folder_path = "/Workspace/Users/tqi6@cdc.gov/TowerScout/"

if git_folder_path not in sys.path:
    sys.path.append(git_folder_path)

# COMMAND ----------

# MAGIC %reload_ext autoreload 
# MAGIC %autoreload 2

# COMMAND ----------

import tsdb.utilities.test as utt
import tsdb.notebooks.mytwo as mytwo

# COMMAND ----------

utt.yeet()

# COMMAND ----------

tsdb.notebooks.mytwo()

# COMMAND ----------



# COMMAND ----------


