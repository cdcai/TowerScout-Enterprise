#
# TowerScout
# A tool for identifying cooling towers from satellite and aerial imagery
#
# TowerScout Team:
# Karen Wong, Gunnar Mein, Thaddeus Segura, Jia Lu
#
# Licensed under CC-BY-NC-SA-4.0
# (see LICENSE.TXT in the root of the repository for details)
#

#
# the provider-independent part of maps
#
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import aiofiles
import requests
import mapbox_vector_tile
import json
import math
from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO
import base64
import asyncio
import aiohttp
import aiofiles
import ssl
from collections import deque
import time
import uuid
from datetime import datetime
from azure.identity import DefaultAzureCredential

class Map:

    def __init__(self):
        self.has_metadata = False

    def get_sat_maps(self, tiles, loop, dir, fname):
        ssl._create_default_https_context = ssl._create_unverified_context
        urls = []
        for tile in tiles:
            # ask provider for this specific url
            url = self.get_url(tile)
            urls.append(url)
            tile['url'] = url
            # print(urls[-1])
            if self.has_metadata:
                urls.append(self.get_meta_url(tile))
        # execute
        loop.run_until_complete(gather_urls(urls, dir, fname, self.has_metadata))
        return self.has_metadata

    #
    # adapted from https://stackoverflow.com/users/6099211/anton-ovsyannikov
    # correct for both bing and GMaps
    #

    def get_static_map_wh(self, lat=None, lng=None, zoom=19, sx=640, sy=640, crop_tiles=False):
        # lat, lng - center
        # sx, sy - map size in pixels

        sy_cropped = int(sy*0.96) if crop_tiles else sy # cut off bottom 4% if cropping requested

        # common factor based on latitude
        lat_factor = math.cos(lat*math.pi/180.)

        # determine degree size
        globe_size = 256 * 2 ** zoom  # total earth map size in pixels at current zoom
        d_lng = sx * 360. / globe_size  # degrees/pixel
        d_lat = sy_cropped * 360. * lat_factor / globe_size  # degrees/pixel
        d_lat_for_url = sy * 360. * lat_factor / globe_size  # degrees/pixel
 
        # determine size in meters
        ground_resolution = 156543.04 * lat_factor / (2 ** zoom)  # meters/pixel
        d_x = sx * ground_resolution
        d_y = sy_cropped * ground_resolution

        #print("d_lat", d_lat, "d_lng", d_lng)
        return (d_lat, d_lat_for_url, d_lng, d_y, d_x)

    #
    # make_map_list:
    #
    # takes a center and radius, or bounds
    # returns a list of centers for zoom 19 scale 2 images
    #

    def make_tiles(self, bounds, overlap_percent=5, crop_tiles=False):
        south, west, north, east = [float(x) for x in bounds.split(",")]

        # width and height of total map
        w = abs(west-east)
        h = abs(south-north)
        lng = (east+west)/2.0
        lat = (north+south)/2.0

        # width and height of a tile as degrees, also get the meters
        h_tile, h_for_url, w_tile, meters, meters_x = self.get_static_map_wh(
            lng=lng, lat=lat, crop_tiles=crop_tiles)
        print(" tile: w:", w_tile, "h:", h_tile)

        # how many tiles horizontally and vertically?
        nx = math.ceil(w/w_tile/(1-overlap_percent/100.))
        ny = math.ceil(h/h_tile/(1-overlap_percent/100.))

        # now make a list of centerpoints of the tiles for the map
        tiles = []
        for row in range(ny):
            for col in range(nx):
                tiles.append({
                    'lat': north - (0.5+row) * h_tile * (1-overlap_percent/100.),
                    'lat_for_url':north - (0.5 * h_for_url + row * h_tile) * (1-overlap_percent/100.),
                    'lng': west + (col+0.5) * w_tile * (1-overlap_percent/100.),
                    'h':h_tile, 
                    'w': w_tile,
                    'id':len(tiles),
                    'col': col,
                    'row': row
                })

        return tiles, nx, ny, meters, h_tile, w_tile


#
#  async file download helpers
#

def vector_tile_to_geojson(vector_tile_data):
    # Decode the vector tile
    tile = mapbox_vector_tile.decode(vector_tile_data)
    
    features = []
    for layer_name, layer in tile.items():
        for feature in layer['features']:
            # Convert each feature to GeoJSON
            geojson = mapbox_vector_tile.feature_to_geojson(feature, layer_name)
            features.append(geojson)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }
def convert_to_data_uri(image_content):
    image = Image.open(BytesIO(image_content))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

async def gather_urls(urls, dir, fname, metadata):
    # execute
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, urls, dir, fname, metadata)

async def rate_limited_fetch(session, url, dir, fname, i, index):
    await asyncio.sleep(index * (1 / 3))
    await fetch(session, url, dir, fname, i)

async def fetch(session, url, dir, fname, i):

    meta = False
    if url.endswith(" (meta)"):
        url = url[0:-7]
        meta = True
 
    async with session.get(url) as response:
        if response.status != 200:
            print(f"Error: HTTP status code {response.status}")
            error_text = await response.text()
            print(f"Error response: {error_text}")
            # response.raise_for_status()

        # write the file
        filename = dir+"/"+fname+str(i)+(".meta.txt" if meta else ".jpg")
        blobname = fname+str(i)+(".meta.txt" if meta else ".jpg")
        
        
        if (response.headers.get('Content-Type', '').lower() == "application/vnd.mapbox-vector-tile"):
            print("Printing Content type " + response.headers.get('Content-Type', '').lower())
            tile_data = await response.read()
            tile = mapbox_vector_tile.decode(tile_data)
            # print(json.dumps(tile, indent=2))
            json_data = {}
            for layer_name, layer in tile.items():
                json_data[layer_name] = {
                    'features': []
                }
                for feature in layer['features']:
                    json_data[layer_name]['features'].append({
                'geometry': feature['geometry'],
                'properties': feature['properties']
            })
            with open(filename, 'w') as f:
                json.dump(json_data, f)
                f.close()
        else:
            # Code to write to Temp directory - This is required for now as detections are made using those images
            # Need to change the code to read from the AIX Team's container later
            async with aiofiles.open(filename, mode='wb') as f:
                await f.write(await response.read())
                await f.close()
            # Create a unique container
            unique_directory = generate_unique_directory_name('dirazuremaptiles')
            directoryname = 'PD/TowerScout/Unstructured/maps/bronze/'+ unique_directory
            # Adding code to write to the AIX container directory
            blob_url=uploadImagetodirUnqFileName(await response.read(),'ddphss-csels',directoryname,blobname)

        
# limit 3 requests per 1 second period (10 requests per 1 second period would hit 429 rate limitted responses)
async def fetch_all(session, urls, dir, fname, metadata):
    tasks = []
    for (i, url) in enumerate(urls):
        suffix = i//2 if metadata else i
        task = rate_limited_fetch(session, url, dir, fname, suffix, i)
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results





#
# radian conversion and Haversine distance
#

def rad(x):
    return x * math.pi / 180.


def get_distance(x1, y1, x2, y2):
    R = 6378137.
    # Earth’s mean radius in meters
    dLat = rad(abs(y2 - y1))
    dLong = rad(abs(x2-x1))
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(rad(y1)) * math.cos(rad(y2)) * \
        math.sin(dLong / 2) * math.sin(dLong / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d
    # returns the distance in meters

#
# bounds checking
#


def check_bounds(x1, y1, x2, y2, bounds):
    south, west, north, east = [float(x) for x in bounds.split(",")]
    return not (y1 < south or y2 > north or x2 < west or x1 > east)


def check_tile_against_bounds(t, bounds):
    south, west, north, east = [float(x) for x in bounds.split(",")]
    x1 = t['lng']-t['w']/2
    x2 = t['lng']+t['w']/2
    y1 = t['lat']+t['h']/2
    y2 = t['lat']-t['h']/2

    return not (y1 < south or y2 > north or x2 < west or x1 > east)

def uploadImage(blbname):
    
    # Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"
    blob_name = blbname
    file_path = "path_to_your_local_file"

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)


    local_file_path = "C:/TowerScout/Testing/Bing/BingTempfolders/tmpg2scritl/tmpg2scritl1.jpg"
    # blob_client = container_client.get_blob_client(blob=local_file_path.split("/")[-1]) 
    blob_client = container_client.get_blob_client(blob_name)
    # Upload the file
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite = True)
        return blob_client.url

async def readimagecontent(local_file_path):
    async with aiofiles.open(local_file_path, 'wb') as file:
     uploadimagecontent = await file.read()
     await file.close()
    return uploadimagecontent

def uploadImageUnqFileName(blobcontent,filename):
    
    # Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"
    
    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = "my-images-directory/"
    
    # Generate a unique file name using UUID
    unique_file_name = f"{uuid.uuid4()}_{filename}"
    blob_name = f"{unique_file_name}"
    blob_client = container_client.get_blob_client(blob_name)
   
    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url
   
    
def uploadImagetodirUnqFileName(blobcontent,containername,directoryname,filename):
    
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net/",
    credential=credential
)
    
    container_name = containername
    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = directoryname
    
    # Generate a unique file name using UUID
    unique_file_name = f"{uuid.uuid4()}_{filename}"
    # Create a BlobClient for the unique file in the directory
    blob_name = f"{directory_name}{unique_file_name}"
    blob_client = container_client.get_blob_client(blob_name)
   
    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url
   
    
def createUnqDir(containername):
    
    # Azure Storage account details
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net/",
    credential=credential
)
    
    
    # Define the directory where the file will be uploaded
    base_name = "dirazuremaptiles"
    unique_directory_name = generate_unique_directory_name(base_name)

    container_name = containername
    # Create a blob client to upload a placeholder file
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{unique_directory_name}placeholder.txt")

    # Upload a zero-byte blob to create the directory
    blob_client.upload_blob(b'', overwrite=True)    
    
    return unique_directory_name
   
def generate_unique_container_name(base_name):
    # Generate a unique name by appending the current timestamp and a UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
    return f"{base_name}-{timestamp}-{unique_id}"

def generate_unique_directory_name(base_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Take first 8 characters of UUID
    return f"{base_name}_{timestamp}-{unique_id}/"

def uploadImage(blobcontent,blbname):
    
    # Azure Storage account details
    connect_str = "DefaultEndpointsProtocol=https;AccountName=tsstorageaccountvssubscr;AccountKey=l7FCoM2QC+3Xljbb9EkwdZXYxtHBkH7GQX6ta5aHZf2i6N9ZCLUSufGaVIUcD693A8cp1QIHzQDO+AStjbtSQQ==;EndpointSuffix=core.windows.net"
    container_name = "firstcontainerunderroot"
    blob_name = blbname

    # Create a BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    container_client = blob_service_client.get_container_client(container_name)

    # Create a BlobClient object for the blob
    blob_client = container_client.get_blob_client(blob_name)
    # # Upload the file

    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url
   
    
def uploadImagetoUnqDir(blobcontent,containername,directoryname,filename):
    
    # Get the token using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Initialize the BlobServiceClient
    storage_account_name = "davsynapseanalyticsdev"
    blob_service_client = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net/",
    credential=credential
)
    
    container_name = containername
    container_client = blob_service_client.get_container_client(container_name)

    # Define the directory where the file will be uploaded
    directory_name = directoryname
    
    # Create a BlobClient for the file in the directory
    blob_name = f"{directory_name}{filename}"
    blob_client = container_client.get_blob_client(blob_name)
   
    blob_client.upload_blob(blobcontent, overwrite=True)
    return blob_client.url
   