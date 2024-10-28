# Databricks notebook source
# MAGIC %sql
# MAGIC -- Purpose: Create table schema
# MAGIC -- Author: Amanuel Anteneh
# MAGIC -- Date: 10/28/2024
# MAGIC
# MAGIC USE CATALOG edav_prd_csels;
# MAGIC USE SCHEMA towerscout;
# MAGIC
# MAGIC
# MAGIC CREATE TABLE  IF NOT EXISTS gold_towerscout_image_reviewed (
# MAGIC   image_path STRING,
# MAGIC   annotations ARRAY<STRUCT<label: INT, xmin: FLOAT, ymin: FLOAT, xmax: FLOAT, ymax: FLOAT>>,
# MAGIC   dataset_type STRING)
# MAGIC USING delta
# MAGIC PARTITIONED BY (dataset_type)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.checkpoint.writeStatsAsJson' = 'false',
# MAGIC   'delta.checkpoint.writeStatsAsStruct' = 'true',
# MAGIC   'delta.enableDeletionVectors' = 'true',
# MAGIC   'delta.feature.deletionVectors' = 'supported',
# MAGIC   'delta.minReaderVersion' = '3',
# MAGIC   'delta.minWriterVersion' = '7');
