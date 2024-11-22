# Databricks notebook source
!pip install aiofiles
!pip install shapely
!pip install geopandas
!pip install efficientnet_pytorch

# COMMAND ----------

# request(callbackUrl).send(towerResults)

# COMMAND ----------

!pip install requests

import requests

# Example of pulling down image and saving it to the tmp directory in the test_volume: 
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg"

# Temporary local path on the driver node
local_image_path = "/tmp/image.jpg"

# Target path in DBFS or a mounted volume
volume_image_path = "dbfs:/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/tmp/image.jpg"

# Fetch the image
response = requests.get(image_url)

# Ensure the request was successful
if response.status_code == 200:
    # Save the image to the temporary local path
    with open(local_image_path, "wb") as file:
        file.write(response.content)
    
    # Use dbutils.fs.cp to copy the file from local storage to the volume
    dbutils.fs.cp(f"file:{local_image_path}", volume_image_path)
    print(f"Image successfully downloaded and saved to {volume_image_path}")
else:
    print(f"Failed to download the image. Status code: {response.status_code}")

# COMMAND ----------

# Import DBUtils
from pyspark.dbutils import DBUtils

# Initialize dbutils
dbutils = DBUtils(spark)

# # Create widgets with default values if they don't exist
# dbutils.widgets.text("bounds", "...", "Bounds")
# dbutils.widgets.text("engine", "bing", "Engine")
# dbutils.widgets.text("provider", "bing", "Provider")
# dbutils.widgets.text("polygons", "...", "Polygons")
# dbutils.widgets.text("estimate", "no", "Estimate")

# Get the task parameters with defaults
bounds = dbutils.widgets.get("bounds")
engine = dbutils.widgets.get("engine")
provider = dbutils.widgets.get("provider")
polygons = dbutils.widgets.get("polygons")
isEstimate = dbutils.widgets.get("estimate") == "yes"

print(bounds)

# COMMAND ----------



# COMMAND ----------

# Load API keys from Databricks Secrets API
towerscout_secrets_scope = "dbs-scope-DDPHSS-CSELS-PD-TOWERSCOUT"
bing_api_key = dbutils.secrets.get(scope=towerscout_secrets_scope, key="bing_api_key")
azure_api_key = dbutils.secrets.get(scope=towerscout_secrets_scope, key="azure_api_key")
google_api_key = dbutils.secrets.get(scope=towerscout_secrets_scope, key="google_api_key")

# Define MAX_TILES, MAX_TILES_SESSION, and any other constants here
print(bing_api_key)
print(azure_api_key)
print(google_api_key)
print("The Bing_API_KEY is set: " + str(bing_api_key != "actual_bing_api_key_value"))  #This key will be updated to an actual key
print("The Azure_API_KEY is set: " + str(azure_api_key.endswith("31JHPM"))) #This key will be updated to an actual key
print("The Google_API_KEY is set: " + str(google_api_key != "actual_google_api_key_value")) #This key will be updated to an actual key



# COMMAND ----------



# COMMAND ----------

# Import necessary libraries
import sys
sys.path.append('/Volumes/edav_dev_csels/towerscout_test_schema/test_volume/scripts')
import json
import tempfile
import os
import uuid
import threading
import ts_maps
import ts_imgutil
from ts_bmaps import BingMap
from ts_gmaps import GoogleMap
from ts_zipcode import Zipcode_Provider
from pyspark.dbutils import DBUtils
import uuid
from ts_yolov5 import YOLOv5_Detector
import time
from ts_en import EN_Classifier
import asyncio
from functools import reduce

MAX_TILES = 1000

# Initialize dbutils
dbutils = DBUtils(spark)

# Get the task parameters with defaults
bounds = dbutils.widgets.get("bounds") or "default_bounds"
engine = dbutils.widgets.get("engine") or "default_engine"
provider = dbutils.widgets.get("provider") or "default_provider"
polygons = dbutils.widgets.get("polygons") or "default_polygons"
isEstimate = (dbutils.widgets.get("estimate") or "no") == "yes"
print(bounds)

print("incoming detection request:")
print(" bounds:", bounds)
print(" engine:", engine)
print(" map provider:", provider)
print(" polygons:", polygons)
# Add any other necessary imports here

process_status = {}
engines = {}

engine_default = None
engine_lock = threading.Lock()
loop = asyncio.get_event_loop()
# EfficientNet secondary classifier
secondary_en = EN_Classifier()

def get_engine(e):
    if e is None:
        e = engine_default

    with engine_lock:
        # take all the other ones out of play
        for engine in engines:
            # print(engine)
            if engines[engine]['id'] != e:
                engines[engine]['engine'] = None

        if engines[e]['engine'] is None:
            print(" loading model:", engines[e]['name'])
            engines[e]['engine'] = YOLOv5_Detector(
                'model_params/yolov5/'+engines[e]['file'])

        return engines[e]['engine']

def process_objects_task(bounds, engine, map, polygons, tiles, process_id, tmpdirname, tmpfilename):
    # start time, get params
    start = time.time()
    results = []
    # get the proper detector
    det = get_engine(engine)
    # main processing:

    # retrieve tiles and metadata if available
    meta = map.get_sat_maps(tiles, loop, tmpdirname, tmpfilename)
    # session['metadata'] = meta
    print(" asynchronously retrieved", len(tiles), "files")

    # augment tiles with retrieved filenames
    for i, tile in enumerate(tiles):
        tile['filename'] = tmpdirname+"/"+tmpfilename+str(i)+".jpg"

    # detect all towers
    results_raw = det.detect(tiles, None, id(process_id), crop_tiles=True, secondary=secondary_en)


    # read metadata if present
    for tile in tiles:
        if meta:
            filename = tmpdirname+"/"+tmpfilename+str(tile['id'])+".meta.txt"
            with open(filename) as f:
                tile['metadata'] = map.get_date(f.read())
                # print(" metadata: "+tile['metadata'])
                f.close
        else:
            tile['metadata'] = ""

    # record some results in session for later saving if desired
    # session['detections'] = make_persistable_tile_results(tiles)

    # post-process the results
    results = []
    for result, tile in zip(results_raw, tiles):
        # adjust xyxy normalized results to lat, long pairs
        for i, object in enumerate(result):
            # object['conf'] *= map.checkCutOffs(object) # used to do this before we started cropping
            object['x1'] = tile['lng'] - 0.5*tile['w'] + object['x1']*tile['w']
            object['x2'] = tile['lng'] - 0.5*tile['w'] + object['x2']*tile['w']
            object['y1'] = tile['lat'] + 0.5*tile['h'] - object['y1']*tile['h']
            object['y2'] = tile['lat'] + 0.5*tile['h'] - object['y2']*tile['h']
            object['tile'] = tile['id']
            object['id_in_tile'] = i
            object['selected'] = object['secondary'] >= 0.35

            # print(" output:",str(object))
        results += result

    # mark results out of bounds or polygon
    for o in results:
        o['inside'] = ts_imgutil.resultIntersectsPolygons(o['x1'], o['y1'], o['x2'], o['y2'], polygons) and \
            ts_maps.check_bounds(o['x1'], o['y1'], o['x2'], o['y2'], bounds)
        #print("in " if o['inside'] else "out ", end="")

    # sort the results by lat, long, conf
    results.sort(key=lambda x: x['y1']*2*180+2*x['x1']+x['conf'])

    # coaslesce neighboring (in list) towers that are closer than 1 m for x1, y1
    if len(results) > 1:
        i = 0
        while i < len(results)-1:
            if ts_maps.get_distance(results[i]['x1'], results[i]['y1'],
                                    results[i+1]['x1'], results[i+1]['y1']) < 1:
                print(" removing 1 duplicate result")
                results.remove(results[i+1])
            else:
                i += 1

    # prepend a pseudo-result for each tile, for debugging
    tile_results = []
    for tile in tiles:
        tile_results.append({
            'x1': tile['lng'] - 0.5*tile['w'],
            'y1': tile['lat'] + 0.5*tile['h'],
            'x2': tile['lng'] + 0.5*tile['w'],
            'y2': tile['lat'] - 0.5*tile['h'],
            'class': 1,
            'class_name': 'tile',
            'conf': 1,
            'metadata': tile['metadata'],
            'url': tile['url'],
            'selected': True
        })

    # all done
    selected = str(reduce(lambda a,e: a+(e['selected']),results, 0))
    print(" request complete," + str(len(results)) +" detections (" + selected +" selected), elapsed time: ", (time.time()-start))
    results = tile_results+results
    print()

    results = json.dumps(results)
    process_status[process_id] = { 'status': 'Completed', 'results': results }

# COMMAND ----------

# Define utility functions if any
# For example, function to process objects, check tile against bounds, etc.

# COMMAND ----------

# Main logic for loading map tiles and storing them
def load_map_tiles(bounds, engine, provider, polygons):
    # Create a map provider object
    map = None
    if provider == "bing":
        map = BingMap(bing_api_key)
    # elif provider == "azure": 
        # map = AzureMap(azure_api_key)
    elif provider == "google":
        map = GoogleMap(google_api_key)
    if map is None:
        print("Could not instantiate map provider:", provider)
        return

    # divide the map into 640x640 parts
    tiles, nx, ny, meters, h, w = map.make_tiles(bounds, crop_tiles=True)
    print(f" {len(tiles)} tiles, {nx} x {ny}, {meters} x {meters} m")
    # print(" Tile centers:")
    # for c in tiles:
    #   print("  ",c)

    tiles = [t for t in tiles if ts_maps.check_tile_against_bounds(t, bounds)]
    tiles = [t for t in tiles if ts_imgutil.tileIntersectsPolygons(t, polygons)]
    for i, tile in enumerate(tiles):
        tile['id'] = i
    print(" tiles left after viewport and polygon filter:", len(tiles))
    if isEstimate:
        print(" returning number of tiles")
        print()
        return str(len(tiles))

    if len(tiles) > MAX_TILES:
        print(" ---> request contains too many tiles")
        return "[]"
    
    # Create a temporary directory in the Databricks File System (DBFS)
    tmpdirname = f"/tmp/{uuid.uuid4()}"
    dbutils.fs.mkdirs(tmpdirname)

    # Get the filename from the temporary directory
    tmpfilename = dbutils.fs.ls(tmpdirname)[0].name

    print(f"Creating temporary directory: {tmpdirname}")

    # Process the objects task
    process_id = str(uuid.uuid4())
    process_status[process_id] = 'Running'
    process_objects_task(bounds, engine, map, polygons, tiles, process_id, tmpdirname, tmpfilename)

    print(f"Task started: {process_id}")
    

load_map_tiles(bounds, engine, provider, polygons, isEstimate)
    

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------


