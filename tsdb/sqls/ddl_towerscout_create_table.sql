-- Purpose: Create table schema
-- Author: Amanuel Anteneh
-- Date: 10/28/2024

USE CATALOG edav_prd_csels;
USE SCHEMA towerscout;


CREATE TABLE  IF NOT EXISTS image_inference_silver (
  user_id STRING,
  request_id STRING,
  uuid STRING,
  processing_time TIMESTAMP,
  bboxes ARRAY<STRUCT<label: INT, x1: FLOAT, y1: FLOAT, x2: FLOAT, y2: FLOAT, conf: FLOAT>>,
  image_hash INT,
  image_path STRING,
  model_version STRUCT<yolo_model: STRING, yolo_model_version: STRING, efficientnet_model: STRING, efficientnet_model_version: STRING>,
  image_metadata STRUCT<lat: FLOAT, long: FLOAT, width: INT, height: INT>,
  map_provider STRING
  )
  USING delta
  PARTITIONED BY(user_id, request_id)
  LOCATION 'abfss://ddphss-csels@edavsynapsedatalake.dfs.core.windows.net/towerscout/silver/image_inference_silver';



CREATE TABLE  IF NOT EXISTS image_reviewed_gold (
  user_id STRING,
  request_id STRING,
  uuid STRING,
  reviewed_time TIMESTAMP,
  bboxes ARRAY<STRUCT<label: INT, x1: FLOAT, y1: FLOAT, x2: FLOAT, y2: FLOAT, conf: FLOAT>>,
  image_hash INT,
  image_path STRING,
  split_label STRING
  )
  USING delta
  LOCATION 'abfss://ddphss-csels@edavsynapsedatalake.dfs.core.windows.net/towerscout/gold/image_reviewed_gold';
